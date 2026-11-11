//! Bounded background jobs for ingestion and other long operations.
//!
//! The HTTP API must not run model inference while holding a request thread
//! for an unbounded time. This module provides a small queue with a fixed
//! number of worker threads, a bounded pending count, deterministic job ids,
//! cooperative cancellation, and a bounded retained-history map.
//!
//! A job task is a `JobTask` object. The queue owns the task, records its
//! lifecycle, and passes a `CancelFlag` so the task can stop between expensive
//! phases. Cancellation is cooperative: a kernel already running inside a
//! model forward pass cannot be interrupted by this module. The server checks
//! the flag before starting a task and again before committing storage.
//!
//! No third-party crate is used; the queue is built from `std::thread`,
//! `std::sync::Mutex`, `std::sync::Condvar`, and atomics.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{Error, Result};

/// Maximum number of jobs that may be pending (queued or running).
pub const MAX_PENDING_JOBS: usize = 4_096;
/// Maximum number of worker threads.
pub const MAX_JOB_WORKERS: usize = 64;
/// Maximum number of finished jobs retained for status queries.
pub const MAX_RETAINED_JOBS: usize = 1_024;

/// Stable positive job identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct JobId(u64);

impl JobId {
    /// Creates a positive job identifier.
    pub fn new(value: u64) -> Result<Self> {
        if value == 0 {
            return Err(Error::usage("job id must be positive"));
        }
        Ok(Self(value))
    }

    /// Returns the raw identifier.
    pub fn get(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for JobId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// Lifecycle state of one job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobStatus {
    /// Waiting for a worker.
    Queued,
    /// A worker is executing the task.
    Running,
    /// The task completed successfully.
    Succeeded,
    /// The task returned an error.
    Failed,
    /// The task was cancelled before or during execution.
    Cancelled,
}

impl JobStatus {
    /// Stable name used in JSON responses.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    /// Returns true when the job has reached a terminal state.
    pub fn is_finished(self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed | Self::Cancelled)
    }
}

/// Cooperative cancellation flag shared by the queue and a task.
#[derive(Debug, Clone)]
pub struct CancelFlag(Arc<AtomicBool>);

impl CancelFlag {
    /// Creates a flag that is not cancelled.
    pub fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    /// Returns true when cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }

    /// Requests cancellation.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

impl Default for CancelFlag {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of one successfully completed job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobOutcome {
    /// Number of documents written.
    pub documents: usize,
    /// Number of chunks written.
    pub chunks: usize,
    /// Optional human-readable summary.
    pub message: Option<String>,
}

impl JobOutcome {
    /// Builds an outcome with no message.
    pub fn new(documents: usize, chunks: usize) -> Self {
        Self {
            documents,
            chunks,
            message: None,
        }
    }

    /// Adds a human-readable summary.
    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = Some(message.into());
        self
    }
}

/// One unit of work executed by a job worker.
pub trait JobTask: Send {
    /// Stable task kind used in job status responses.
    fn kind(&self) -> &'static str;

    /// Executes the task, checking `cancel` between expensive phases.
    ///
    /// # Errors
    ///
    /// Returns a typed error when the work cannot be completed. The queue
    /// records the error message and marks the job failed unless cancellation
    /// was requested, in which case it is marked cancelled.
    fn run(&mut self, cancel: &CancelFlag) -> Result<JobOutcome>;
}

/// Immutable snapshot of one job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobSnapshot {
    /// Job identifier.
    pub id: JobId,
    /// Task kind.
    pub kind: String,
    /// Current status.
    pub status: JobStatus,
    /// Creation time in Unix milliseconds.
    pub created_ms: u64,
    /// Start time in Unix milliseconds, when the job started.
    pub started_ms: Option<u64>,
    /// Finish time in Unix milliseconds, when the job finished.
    pub finished_ms: Option<u64>,
    /// Number of documents written.
    pub documents: usize,
    /// Number of chunks written.
    pub chunks: usize,
    /// Error or summary message.
    pub message: Option<String>,
}

/// Counters exposed by the job queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct JobStats {
    /// Jobs waiting for a worker.
    pub queued: usize,
    /// Jobs currently running.
    pub running: usize,
    /// Successfully completed jobs retained in history.
    pub succeeded: usize,
    /// Failed jobs retained in history.
    pub failed: usize,
    /// Cancelled jobs retained in history.
    pub cancelled: usize,
}

/// Result of a cancellation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelOutcome {
    /// Cancellation was requested for a queued or running job.
    Requested,
    /// The job had already reached a terminal state.
    AlreadyFinished,
    /// No job with this id exists.
    NotFound,
}

/// Bounded job queue with fixed worker threads.
#[derive(Debug)]
pub struct JobQueue {
    inner: Arc<Inner>,
    workers: Mutex<Vec<JoinHandle<()>>>,
}

impl std::fmt::Debug for Inner {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("JobQueueInner")
            .field("capacity", &self.capacity)
            .field("pending", &self.pending.load(Ordering::Relaxed))
            .field("shutdown", &self.shutdown.load(Ordering::Relaxed))
            .finish()
    }
}

struct Inner {
    capacity: usize,
    queue: Mutex<VecDeque<QueuedJob>>,
    records: Mutex<BTreeMap<JobId, Record>>,
    next_id: AtomicU64,
    pending: AtomicUsize,
    shutdown: AtomicBool,
    available: Condvar,
}

struct QueuedJob {
    id: JobId,
    task: Box<dyn JobTask>,
}

#[derive(Debug)]
struct Record {
    kind: String,
    status: JobStatus,
    created_ms: u64,
    started_ms: Option<u64>,
    finished_ms: Option<u64>,
    documents: usize,
    chunks: usize,
    message: Option<String>,
    cancel: CancelFlag,
}

impl Record {
    fn snapshot(&self, id: JobId) -> JobSnapshot {
        JobSnapshot {
            id,
            kind: self.kind.clone(),
            status: self.status,
            created_ms: self.created_ms,
            started_ms: self.started_ms,
            finished_ms: self.finished_ms,
            documents: self.documents,
            chunks: self.chunks,
            message: self.message.clone(),
        }
    }
}

impl JobQueue {
    /// Creates a queue and starts the requested number of workers.
    ///
    /// # Errors
    ///
    /// Returns a usage error for a zero capacity or worker count, a limit error
    /// for a value above the documented cap, or an I/O error if a worker thread
    /// cannot be started.
    pub fn new(capacity: usize, workers: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(Error::usage("job queue capacity must be positive"));
        }
        if capacity > MAX_PENDING_JOBS {
            return Err(Error::limit_exceeded(format!(
                "job queue capacity {capacity} is above the {MAX_PENDING_JOBS}-job cap"
            )));
        }
        if workers == 0 {
            return Err(Error::usage("job worker count must be positive"));
        }
        if workers > MAX_JOB_WORKERS {
            return Err(Error::limit_exceeded(format!(
                "job worker count {workers} is above the {MAX_JOB_WORKERS}-worker cap"
            )));
        }

        let inner = Arc::new(Inner {
            capacity,
            queue: Mutex::new(VecDeque::new()),
            records: Mutex::new(BTreeMap::new()),
            next_id: AtomicU64::new(1),
            pending: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            available: Condvar::new(),
        });

        let mut handles = Vec::with_capacity(workers);
        for index in 0..workers {
            let worker_inner = Arc::clone(&inner);
            match thread::Builder::new()
                .name(format!("seldon-job-{index}"))
                .spawn(move || worker_loop(worker_inner))
            {
                Ok(handle) => handles.push(handle),
                Err(error) => {
                    inner.shutdown.store(true, Ordering::SeqCst);
                    inner.available.notify_all();
                    for handle in handles {
                        let _ = handle.join();
                    }
                    return Err(Error::Io(error));
                }
            }
        }

        Ok(Self {
            inner,
            workers: Mutex::new(handles),
        })
    }

    /// Submits a task and returns its job id.
    ///
    /// # Errors
    ///
    /// Returns a limit error when the pending-job cap or retained-history cap
    /// is reached, and an internal error if a queue lock is poisoned.
    pub fn submit(&self, task: Box<dyn JobTask>) -> Result<JobId> {
        if self.inner.shutdown.load(Ordering::SeqCst) {
            return Err(Error::usage("job queue is shutting down"));
        }

        let current = self.inner.pending.fetch_add(1, Ordering::SeqCst);
        if current >= self.inner.capacity {
            self.inner.pending.fetch_sub(1, Ordering::SeqCst);
            return Err(Error::limit_exceeded(format!(
                "job queue already has {} pending jobs, the documented cap",
                self.inner.capacity
            )));
        }

        let id_value = self.inner.next_id.fetch_add(1, Ordering::SeqCst);
        let id = match JobId::new(id_value) {
            Ok(id) => id,
            Err(error) => {
                self.inner.pending.fetch_sub(1, Ordering::SeqCst);
                return Err(error);
            }
        };
        let kind = task.kind().to_owned();
        let cancel = CancelFlag::new();
        let record = Record {
            kind,
            status: JobStatus::Queued,
            created_ms: now_ms(),
            started_ms: None,
            finished_ms: None,
            documents: 0,
            chunks: 0,
            message: None,
            cancel,
        };

        {
            let mut records = self
                .inner
                .records
                .lock()
                .map_err(|_| Error::internal("job record lock is poisoned"))?;
            if let Err(error) = prune_finished(&mut records, MAX_RETAINED_JOBS) {
                self.inner.pending.fetch_sub(1, Ordering::SeqCst);
                return Err(error);
            }
            records.insert(id, record);
        }

        {
            let mut queue = self
                .inner
                .queue
                .lock()
                .map_err(|_| Error::internal("job queue lock is poisoned"))?;
            queue.push_back(QueuedJob { id, task });
        }
        self.inner.available.notify_one();
        Ok(id)
    }

    /// Returns a snapshot of one job, or None when it does not exist.
    ///
    /// # Errors
    ///
    /// Returns an internal error if the record lock is poisoned.
    pub fn status(&self, id: JobId) -> Result<Option<JobSnapshot>> {
        let records = self
            .inner
            .records
            .lock()
            .map_err(|_| Error::internal("job record lock is poisoned"))?;
        Ok(records.get(&id).map(|record| record.snapshot(id)))
    }

    /// Requests cancellation of a queued or running job.
    ///
    /// # Errors
    ///
    /// Returns an internal error if the record lock is poisoned.
    pub fn cancel(&self, id: JobId) -> Result<CancelOutcome> {
        let mut records = self
            .inner
            .records
            .lock()
            .map_err(|_| Error::internal("job record lock is poisoned"))?;
        let Some(record) = records.get_mut(&id) else {
            return Ok(CancelOutcome::NotFound);
        };
        if record.status.is_finished() {
            return Ok(CancelOutcome::AlreadyFinished);
        }
        record.cancel.cancel();
        Ok(CancelOutcome::Requested)
    }

    /// Returns queue counters.
    ///
    /// # Errors
    ///
    /// Returns an internal error if the record lock is poisoned.
    pub fn stats(&self) -> Result<JobStats> {
        let records = self
            .inner
            .records
            .lock()
            .map_err(|_| Error::internal("job record lock is poisoned"))?;
        let mut stats = JobStats::default();
        for record in records.values() {
            match record.status {
                JobStatus::Queued => stats.queued += 1,
                JobStatus::Running => stats.running += 1,
                JobStatus::Succeeded => stats.succeeded += 1,
                JobStatus::Failed => stats.failed += 1,
                JobStatus::Cancelled => stats.cancelled += 1,
            }
        }
        Ok(stats)
    }

    /// Stops accepting work, cancels pending jobs, and joins every worker.
    ///
    /// Running tasks are asked to cancel cooperatively; the call waits for
    /// them to return. It is safe to call more than once.
    pub fn shutdown(&self) {
        if self.inner.shutdown.swap(true, Ordering::SeqCst) {
            return;
        }
        if let Ok(mut records) = self.inner.records.lock() {
            for record in records.values_mut() {
                if !record.status.is_finished() {
                    record.cancel.cancel();
                }
            }
        }
        self.inner.available.notify_all();
        let handles = match self.workers.lock() {
            Ok(mut workers) => std::mem::take(&mut *workers),
            Err(poisoned) => std::mem::take(&mut *poisoned.into_inner()),
        };
        for handle in handles {
            let _ = handle.join();
        }
    }
}

impl Drop for JobQueue {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn worker_loop(inner: Arc<Inner>) {
    loop {
        let queued = {
            let mut queue = match inner.queue.lock() {
                Ok(queue) => queue,
                Err(_) => return,
            };
            loop {
                if inner.shutdown.load(Ordering::SeqCst) {
                    while let Some(job) = queue.pop_front() {
                        finish_cancelled(&inner, job.id);
                        inner.pending.fetch_sub(1, Ordering::SeqCst);
                    }
                    return;
                }
                if let Some(job) = queue.pop_front() {
                    break job;
                }
                queue = match inner.available.wait(queue) {
                    Ok(queue) => queue,
                    Err(_) => return,
                };
            }
        };

        let id = queued.id;
        let mut task = queued.task;
        let cancel = match inner.records.lock() {
            Ok(records) => match records.get(&id) {
                Some(record) => record.cancel.clone(),
                None => {
                    inner.pending.fetch_sub(1, Ordering::SeqCst);
                    continue;
                }
            },
            Err(_) => {
                inner.pending.fetch_sub(1, Ordering::SeqCst);
                return;
            }
        };

        mark_running(&inner, id);
        if cancel.is_cancelled() {
            finish_cancelled(&inner, id);
            inner.pending.fetch_sub(1, Ordering::SeqCst);
            continue;
        }

        let result = task.run(&cancel);
        match result {
            Ok(outcome) => finish_succeeded(&inner, id, outcome),
            Err(_) if cancel.is_cancelled() => finish_cancelled(&inner, id),
            Err(error) => finish_failed(&inner, id, error.to_string()),
        }
        inner.pending.fetch_sub(1, Ordering::SeqCst);
    }
}

fn prune_finished(records: &mut BTreeMap<JobId, Record>, max: usize) -> Result<()> {
    while records.len() >= max {
        let oldest = records
            .iter()
            .find(|(_, record)| record.status.is_finished())
            .map(|(id, _)| *id);
        match oldest {
            Some(id) => {
                records.remove(&id);
            }
            None => {
                return Err(Error::limit_exceeded(format!(
                    "job history holds {max} unfinished jobs; cannot retain another"
                )));
            }
        }
    }
    Ok(())
}

fn mark_running(inner: &Inner, id: JobId) {
    if let Ok(mut records) = inner.records.lock()
        && let Some(record) = records.get_mut(&id)
    {
        record.status = JobStatus::Running;
        record.started_ms = Some(now_ms());
    }
}

fn finish_succeeded(inner: &Inner, id: JobId, outcome: JobOutcome) {
    if let Ok(mut records) = inner.records.lock()
        && let Some(record) = records.get_mut(&id)
    {
        record.status = JobStatus::Succeeded;
        record.finished_ms = Some(now_ms());
        record.documents = outcome.documents;
        record.chunks = outcome.chunks;
        record.message = outcome.message;
    }
}

fn finish_failed(inner: &Inner, id: JobId, message: String) {
    if let Ok(mut records) = inner.records.lock()
        && let Some(record) = records.get_mut(&id)
    {
        record.status = JobStatus::Failed;
        record.finished_ms = Some(now_ms());
        record.message = Some(message);
    }
}

fn finish_cancelled(inner: &Inner, id: JobId) {
    if let Ok(mut records) = inner.records.lock()
        && let Some(record) = records.get_mut(&id)
    {
        record.status = JobStatus::Cancelled;
        record.finished_ms = Some(now_ms());
        record.message = Some("job cancelled".to_owned());
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    struct CountingTask {
        counter: Arc<AtomicUsize>,
    }

    impl JobTask for CountingTask {
        fn kind(&self) -> &'static str {
            "test"
        }

        fn run(&mut self, _cancel: &CancelFlag) -> Result<JobOutcome> {
            self.counter.fetch_add(1, Ordering::SeqCst);
            Ok(JobOutcome::new(1, 1))
        }
    }

    struct BlockingTask {
        started: Arc<AtomicBool>,
        release: Arc<AtomicBool>,
    }

    impl JobTask for BlockingTask {
        fn kind(&self) -> &'static str {
            "blocking"
        }

        fn run(&mut self, cancel: &CancelFlag) -> Result<JobOutcome> {
            self.started.store(true, Ordering::SeqCst);
            while !self.release.load(Ordering::SeqCst) {
                if cancel.is_cancelled() {
                    return Err(Error::usage("cancelled"));
                }
                thread::sleep(std::time::Duration::from_millis(2));
            }
            Ok(JobOutcome::new(1, 1))
        }
    }

    fn wait_for<F>(mut predicate: F)
    where
        F: FnMut() -> bool,
    {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !predicate() {
            assert!(
                std::time::Instant::now() < deadline,
                "condition did not become true"
            );
            thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    #[test]
    fn runs_a_task_and_records_the_outcome() {
        let queue = JobQueue::new(4, 1).expect("queue");
        let counter = Arc::new(AtomicUsize::new(0));
        let id = queue
            .submit(Box::new(CountingTask {
                counter: Arc::clone(&counter),
            }))
            .expect("submit");
        wait_for(|| {
            queue
                .status(id)
                .expect("status")
                .is_some_and(|job| job.status == JobStatus::Succeeded)
        });
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        let job = queue.status(id).expect("status").expect("job");
        assert_eq!(job.documents, 1);
        assert_eq!(job.chunks, 1);
        queue.shutdown();
    }

    #[test]
    fn cancels_a_queued_job() {
        let queue = JobQueue::new(2, 1).expect("queue");
        let started = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let first = queue
            .submit(Box::new(BlockingTask {
                started: Arc::clone(&started),
                release: Arc::clone(&release),
            }))
            .expect("first");
        wait_for(|| started.load(Ordering::SeqCst));
        let second = queue
            .submit(Box::new(CountingTask {
                counter: Arc::new(AtomicUsize::new(0)),
            }))
            .expect("second");
        assert_eq!(
            queue.cancel(second).expect("cancel"),
            CancelOutcome::Requested
        );
        release.store(true, Ordering::SeqCst);
        wait_for(|| {
            queue
                .status(first)
                .expect("status")
                .is_some_and(|job| job.status.is_finished())
                && queue
                    .status(second)
                    .expect("status")
                    .is_some_and(|job| job.status == JobStatus::Cancelled)
        });
        queue.shutdown();
    }

    #[test]
    fn enforces_the_pending_cap() {
        let queue = JobQueue::new(1, 1).expect("queue");
        let started = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let _first = queue
            .submit(Box::new(BlockingTask {
                started: Arc::clone(&started),
                release: Arc::clone(&release),
            }))
            .expect("first");
        wait_for(|| started.load(Ordering::SeqCst));
        let error = queue
            .submit(Box::new(CountingTask {
                counter: Arc::new(AtomicUsize::new(0)),
            }))
            .expect_err("cap");
        assert!(matches!(error, Error::LimitExceeded(_)));
        release.store(true, Ordering::SeqCst);
        queue.shutdown();
    }
}
