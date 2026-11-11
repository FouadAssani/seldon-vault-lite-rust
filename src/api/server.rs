//! Native HTTP/1.1 server, route handlers, and bounded job integration.
//!
//! The server binds to `127.0.0.1` by default. Binding to a non-loopback
//! address requires an access token at startup; the token is checked on every
//! request with a constant-time comparison. There is no TLS implementation, so
//! the documentation explicitly forbids exposing the service directly to the
//! Internet.
//!
//! One request is handled per connection. The accept loop is non-blocking so a
//! shutdown flag can be observed without a signal handler. Each connection has
//! a bounded read/write timeout, the total connection count is bounded, and
//! every request body is bounded before allocation. Ingestion is delegated to
//! the bounded job queue in `crate::jobs`; searches open a lock-free read-only
//! store view so they do not hold the writer lock.

use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::api::base64::{self, MAX_MEDIA_BYTES};
use crate::api::engine::{Engine, ImageFormat};
use crate::api::http::{self, HttpError, Method, Request, Response};
use crate::error::{Error, Result};
use crate::index::{Filters, Index, SearchMode, SearchRequest};
use crate::jobs::{CancelFlag, CancelOutcome, JobId, JobOutcome, JobQueue, JobSnapshot, JobTask};
use crate::json::{self, Limits, Number, Object, Value};
use crate::media::preprocess::PreprocessOptions;
use crate::model::prompt::Prompt;
use crate::storage::model::{ChunkInput, Document, DocumentId, Metadata, Modality};
use crate::storage::{ReadOnlyStore, Store};
use crate::{MODEL_CACHE_DIR, MODEL_REVISION, PROGRAM_NAME, TARGET_TRIPLE, VERSION};

type HttpResult<T> = std::result::Result<T, HttpError>;

/// Default bind address.
pub const DEFAULT_BIND: &str = "127.0.0.1:8080";
/// Default maximum number of concurrent connections.
pub const DEFAULT_MAX_CONNECTIONS: usize = 32;
/// Default request read/write timeout in milliseconds.
pub const DEFAULT_REQUEST_TIMEOUT_MS: u64 = 10_000;
/// Default maximum request body size in bytes.
pub const DEFAULT_MAX_BODY_BYTES: usize = 32 * 1024 * 1024;
/// Default bounded ingestion queue capacity.
pub const DEFAULT_QUEUE_CAPACITY: usize = 64;
/// Default number of job workers.
pub const DEFAULT_JOB_WORKERS: usize = 1;
/// Maximum accepted connection count.
pub const MAX_CONNECTIONS: usize = 1_024;
/// Maximum accepted request timeout.
pub const MAX_REQUEST_TIMEOUT: Duration = Duration::from_secs(300);
/// Minimum accepted request timeout.
pub const MIN_REQUEST_TIMEOUT: Duration = Duration::from_millis(100);
/// Maximum accepted request body size.
pub const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;
/// Minimum accepted request body size.
pub const MIN_BODY_BYTES: usize = 1024;
/// Maximum number of token ids accepted in one request.
pub const MAX_TOKEN_IDS: usize = 8_192;
/// Maximum text input size in bytes.
pub const MAX_TEXT_INPUT_BYTES: usize = 1024 * 1024;
/// Maximum search limit.
pub const MAX_SEARCH_LIMIT: usize = 1_000;
/// Maximum search offset.
pub const MAX_SEARCH_OFFSET: usize = 1_000_000;
/// Maximum bytes drained from a rejected connection before the 503 response.
pub const MAX_REJECT_DRAIN_BYTES: usize = 64 * 1024;
/// Maximum access-token length in bytes.
pub const MAX_TOKEN_BYTES: usize = 512;
/// Minimum access-token length for a non-loopback bind.
pub const MIN_LAN_TOKEN_BYTES: usize = 16;

/// Runtime configuration of the HTTP service.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Bind address. The default is `127.0.0.1:8080`.
    pub bind: SocketAddr,
    /// Data directory holding the journal and snapshots.
    pub data_dir: PathBuf,
    /// Model directory containing config.json and model.safetensors.
    pub model_dir: PathBuf,
    /// Embedding dimension used when a request omits it.
    pub dimension: usize,
    /// Visual token budget for image requests.
    pub vision_budget: usize,
    /// Optional bearer token required on every request.
    pub access_token: Option<String>,
    /// Maximum number of concurrent connections.
    pub max_connections: usize,
    /// Read and write timeout per connection.
    pub request_timeout: Duration,
    /// Maximum request body size in bytes.
    pub max_body_bytes: usize,
    /// Maximum number of queued or running jobs.
    pub queue_capacity: usize,
    /// Number of job worker threads.
    pub job_workers: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: DEFAULT_BIND
                .parse()
                .expect("the documented default bind address is valid"),
            data_dir: PathBuf::from("data"),
            model_dir: Path::new(MODEL_CACHE_DIR).join(MODEL_REVISION),
            dimension: 768,
            vision_budget: crate::media::preprocess::DEFAULT_MAX_SOFT_TOKENS,
            access_token: None,
            max_connections: DEFAULT_MAX_CONNECTIONS,
            request_timeout: Duration::from_millis(DEFAULT_REQUEST_TIMEOUT_MS),
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            queue_capacity: DEFAULT_QUEUE_CAPACITY,
            job_workers: DEFAULT_JOB_WORKERS,
        }
    }
}

impl ServerConfig {
    /// Validates the configuration before any model or storage work starts.
    ///
    /// # Errors
    ///
    /// Returns a usage error for an invalid limit, an invalid token, or a
    /// non-loopback bind without a token of at least 16 bytes.
    pub fn validate(&self) -> Result<()> {
        if self.max_connections == 0 || self.max_connections > MAX_CONNECTIONS {
            return Err(Error::usage(format!(
                "max_connections must be in 1..={MAX_CONNECTIONS}"
            )));
        }
        if self.request_timeout < MIN_REQUEST_TIMEOUT || self.request_timeout > MAX_REQUEST_TIMEOUT
        {
            return Err(Error::usage(format!(
                "request_timeout must be in {}..={} ms",
                MIN_REQUEST_TIMEOUT.as_millis(),
                MAX_REQUEST_TIMEOUT.as_millis()
            )));
        }
        if self.max_body_bytes < MIN_BODY_BYTES || self.max_body_bytes > MAX_BODY_BYTES {
            return Err(Error::usage(format!(
                "max_body_bytes must be in {MIN_BODY_BYTES}..={MAX_BODY_BYTES}"
            )));
        }
        if self.queue_capacity == 0 || self.queue_capacity > crate::jobs::MAX_PENDING_JOBS {
            return Err(Error::usage(format!(
                "queue_capacity must be in 1..={}",
                crate::jobs::MAX_PENDING_JOBS
            )));
        }
        if self.job_workers == 0 || self.job_workers > crate::jobs::MAX_JOB_WORKERS {
            return Err(Error::usage(format!(
                "job_workers must be in 1..={}",
                crate::jobs::MAX_JOB_WORKERS
            )));
        }
        PreprocessOptions::new(self.vision_budget)?;
        if let Some(token) = &self.access_token {
            if token.is_empty() {
                return Err(Error::usage("access token must not be empty"));
            }
            if token.len() > MAX_TOKEN_BYTES {
                return Err(Error::limit_exceeded(format!(
                    "access token is {} bytes, above the {MAX_TOKEN_BYTES}-byte cap",
                    token.len()
                )));
            }
            if token.chars().any(char::is_control) {
                return Err(Error::usage(
                    "access token must not contain control characters",
                ));
            }
        }
        if !self.bind.ip().is_loopback() {
            match &self.access_token {
                None => {
                    return Err(Error::usage(
                        "binding to a non-loopback address requires --token",
                    ));
                }
                Some(token) if token.len() < MIN_LAN_TOKEN_BYTES => {
                    return Err(Error::usage(format!(
                        "a non-loopback bind requires an access token of at least {MIN_LAN_TOKEN_BYTES} bytes"
                    )));
                }
                Some(_) => {}
            }
        }
        Ok(())
    }

    /// Returns true when the configured bind address is loopback.
    pub fn is_loopback(&self) -> bool {
        self.bind.ip().is_loopback()
    }
}

/// A bound but not yet running HTTP server.
#[derive(Debug)]
pub struct Server {
    listener: TcpListener,
    state: Arc<ServerState>,
    local_addr: SocketAddr,
}

impl Server {
    /// Binds the listener, loads the model, and opens the store.
    ///
    /// The store's exclusive writer lock is acquired before the listener is
    /// bound, so a second process cannot start on the same data directory.
    ///
    /// # Errors
    ///
    /// Returns a typed error for an invalid configuration, a model failure, a
    /// locked data directory, or a bind failure.
    pub fn bind(config: ServerConfig) -> Result<Self> {
        config.validate()?;
        let engine = Engine::load(&config.model_dir, config.vision_budget)?;
        engine.validate_dimension(config.dimension)?;
        let store = Store::open(&config.data_dir)?;
        let state = Arc::new(ServerState::new(config.clone(), engine, store)?);
        let listener = TcpListener::bind(config.bind)?;
        listener.set_nonblocking(true)?;
        let local_addr = listener.local_addr()?;
        Ok(Self {
            listener,
            state,
            local_addr,
        })
    }

    /// Actual bound address. Useful when port zero was requested by tests.
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Runs the accept loop until shutdown is requested.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the listener fails unexpectedly. A shutdown
    /// request is not an error.
    pub fn run(self) -> Result<()> {
        self.run_inner()
    }

    /// Starts the accept loop on a dedicated thread.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the server thread cannot be started.
    pub fn spawn(self) -> Result<RunningServer> {
        let addr = self.local_addr;
        let state = Arc::clone(&self.state);
        let handle = thread::Builder::new()
            .name("seldon-http".to_owned())
            .spawn(move || self.run())?;
        Ok(RunningServer {
            addr,
            state,
            join: Some(handle),
        })
    }

    fn run_inner(self) -> Result<()> {
        let state = Arc::clone(&self.state);
        let result = self.accept_loop(&state);
        wait_for_connections(&state);
        state.jobs.shutdown();
        result
    }

    fn accept_loop(&self, state: &Arc<ServerState>) -> Result<()> {
        while !state.shutdown.load(Ordering::SeqCst) {
            match self.listener.accept() {
                Ok((stream, peer)) => {
                    if !acquire_connection(state) {
                        state
                            .metrics
                            .rejected_connections
                            .fetch_add(1, Ordering::Relaxed);
                        reject_busy(stream, state);
                        continue;
                    }
                    let connection_state = Arc::clone(state);
                    let spawned = thread::Builder::new()
                        .name("seldon-http-connection".to_owned())
                        .spawn(move || {
                            let _guard = ConnectionGuard {
                                state: Arc::clone(&connection_state),
                            };
                            handle_connection(stream, peer, &connection_state);
                        });
                    if spawned.is_err() {
                        state.active_connections.fetch_sub(1, Ordering::SeqCst);
                        state.metrics.spawn_failures.fetch_add(1, Ordering::Relaxed);
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(10));
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(error) => return Err(Error::Io(error)),
            }
        }
        Ok(())
    }
}

/// A server running on a dedicated thread.
#[derive(Debug)]
pub struct RunningServer {
    addr: SocketAddr,
    state: Arc<ServerState>,
    join: Option<JoinHandle<Result<()>>>,
}

impl RunningServer {
    /// Actual bound address.
    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }

    /// Requests a graceful shutdown.
    pub fn shutdown(&self) {
        self.state.shutdown.store(true, Ordering::SeqCst);
    }

    /// Waits for the accept loop and job workers to stop.
    ///
    /// # Errors
    ///
    /// Returns the server thread's error or an internal error if the thread
    /// panicked or was already joined.
    pub fn join(mut self) -> Result<()> {
        let handle = self
            .join
            .take()
            .ok_or_else(|| Error::internal("server thread was already joined"))?;
        match handle.join() {
            Ok(result) => result,
            Err(_) => Err(Error::internal("HTTP server thread panicked")),
        }
    }
}

impl Drop for RunningServer {
    fn drop(&mut self) {
        self.shutdown();
        if let Some(handle) = self.join.take() {
            let _ = handle.join();
        }
    }
}

struct ConnectionGuard {
    state: Arc<ServerState>,
}

impl Drop for ConnectionGuard {
    fn drop(&mut self) {
        self.state.active_connections.fetch_sub(1, Ordering::SeqCst);
    }
}

#[derive(Debug, Default)]
struct Metrics {
    requests_total: AtomicU64,
    status_2xx: AtomicU64,
    status_4xx: AtomicU64,
    status_5xx: AtomicU64,
    bytes_in: AtomicU64,
    bytes_out: AtomicU64,
    rejected_connections: AtomicU64,
    spawn_failures: AtomicU64,
}

#[derive(Debug)]
struct ServerState {
    config: ServerConfig,
    engine: Arc<Engine>,
    store: Arc<Mutex<Store>>,
    jobs: JobQueue,
    shutdown: AtomicBool,
    active_connections: AtomicUsize,
    metrics: Metrics,
    started: Instant,
}

impl ServerState {
    fn new(config: ServerConfig, engine: Engine, store: Store) -> Result<Self> {
        let jobs = JobQueue::new(config.queue_capacity, config.job_workers)?;
        Ok(Self {
            config,
            engine: Arc::new(engine),
            store: Arc::new(Mutex::new(store)),
            jobs,
            shutdown: AtomicBool::new(false),
            active_connections: AtomicUsize::new(0),
            metrics: Metrics::default(),
            started: Instant::now(),
        })
    }
}

fn acquire_connection(state: &ServerState) -> bool {
    let current = state.active_connections.fetch_add(1, Ordering::SeqCst);
    if current >= state.config.max_connections {
        state.active_connections.fetch_sub(1, Ordering::SeqCst);
        false
    } else {
        true
    }
}

fn reject_busy(mut stream: TcpStream, state: &ServerState) {
    // Drain the request before writing the rejection. Closing a socket with
    // unread bytes can turn the response into a reset, which a client sees as
    // a truncated response instead of a clean 503.
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_read_timeout(Some(Duration::from_millis(100)));
    let _ = stream.set_write_timeout(Some(state.config.request_timeout));
    let drain_cap = state.config.max_body_bytes.min(MAX_REJECT_DRAIN_BYTES);
    let _ = http::read_request(&mut stream, drain_cap);
    let response = error_response(HttpError::new(
        503,
        "connection_limit",
        "the server connection limit is reached; retry later",
    ));
    let _ = http::write_response(&mut stream, &response);
}

fn wait_for_connections(state: &ServerState) {
    let timeout = state.config.request_timeout.saturating_mul(2);
    let deadline = Instant::now() + timeout;
    while state.active_connections.load(Ordering::SeqCst) > 0 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
}

fn handle_connection(mut stream: TcpStream, peer: SocketAddr, state: &Arc<ServerState>) {
    // The listener is non-blocking so the accept loop can observe shutdown.
    // An accepted stream must be switched back to blocking mode before the
    // per-connection timeout is installed; otherwise a partial request would
    // fail immediately with WouldBlock instead of waiting for the timeout.
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_read_timeout(Some(state.config.request_timeout));
    let _ = stream.set_write_timeout(Some(state.config.request_timeout));
    let _ = stream.set_nodelay(true);
    state.metrics.requests_total.fetch_add(1, Ordering::Relaxed);

    let response = match http::read_request(&mut stream, state.config.max_body_bytes) {
        Ok(request) => {
            state
                .metrics
                .bytes_in
                .fetch_add(request.body.len() as u64, Ordering::Relaxed);
            route(state, &request, peer)
        }
        Err(error) => error_response(error),
    };
    record_response(state, &response);
    let _ = http::write_response(&mut stream, &response);
}

fn record_response(state: &ServerState, response: &Response) {
    state
        .metrics
        .bytes_out
        .fetch_add(response.body.len() as u64, Ordering::Relaxed);
    match response.status {
        200..=299 => {
            state.metrics.status_2xx.fetch_add(1, Ordering::Relaxed);
        }
        400..=499 => {
            state.metrics.status_4xx.fetch_add(1, Ordering::Relaxed);
        }
        500..=599 => {
            state.metrics.status_5xx.fetch_add(1, Ordering::Relaxed);
        }
        _ => {}
    }
}

fn route(state: &Arc<ServerState>, request: &Request, peer: SocketAddr) -> Response {
    if !authorized(state, request) {
        let mut response = error_response(HttpError::new(
            401,
            "unauthorized",
            "a valid bearer token is required",
        ));
        response
            .headers
            .push(("WWW-Authenticate".to_owned(), "Bearer".to_owned()));
        return response;
    }

    let Some(allowed) = allowed_methods(&request.path) else {
        return error_response(HttpError::new(404, "not_found", "unknown route"));
    };
    if !allowed
        .split(", ")
        .any(|method| method == request.method.as_str())
    {
        let mut response = error_response(HttpError::new(
            405,
            "method_not_allowed",
            format!(
                "{} is not allowed on {}; use {}",
                request.method.as_str(),
                request.path,
                allowed
            ),
        ));
        response
            .headers
            .push(("Allow".to_owned(), allowed.to_owned()));
        return response;
    }

    if request.query.is_some() {
        return error_response(HttpError::bad_request("query parameters are not supported"));
    }

    match dispatch(state, request, peer) {
        Ok(response) => response,
        Err(error) => error_response(error),
    }
}

fn allowed_methods(path: &str) -> Option<&'static str> {
    match path {
        "/health" | "/ready" | "/v1/models" | "/v1/stats" => Some("GET"),
        "/v1/embeddings" | "/v1/documents" | "/v1/search" | "/v1/shutdown" => Some("POST"),
        path if path.starts_with("/v1/documents/") => Some("DELETE"),
        path if path.starts_with("/v1/jobs/") => Some("GET, DELETE"),
        _ => None,
    }
}

fn dispatch(state: &Arc<ServerState>, request: &Request, peer: SocketAddr) -> HttpResult<Response> {
    match request.path.as_str() {
        "/health" => health(state),
        "/ready" => ready(state),
        "/v1/models" => models(state),
        "/v1/stats" => stats(state),
        "/v1/embeddings" => embeddings(state, request),
        "/v1/documents" => create_document(state, request),
        "/v1/search" => search(state, request),
        "/v1/shutdown" => {
            require_json(request)?;
            shutdown(state, peer)
        }
        path if path.starts_with("/v1/documents/") => delete_document(state, request),
        path if path.starts_with("/v1/jobs/") => jobs_route(state, request),
        _ => Err(HttpError::new(404, "not_found", "unknown route")),
    }
}

fn authorized(state: &ServerState, request: &Request) -> bool {
    let Some(expected) = &state.config.access_token else {
        return true;
    };
    if let Some(value) = request.header("authorization") {
        let Some((scheme, token)) = value.split_once(' ') else {
            return false;
        };
        if !scheme.eq_ignore_ascii_case("bearer") {
            return false;
        }
        return constant_time_eq(token.as_bytes(), expected.as_bytes());
    }
    if let Some(value) = request.header("x-api-key") {
        return constant_time_eq(value.as_bytes(), expected.as_bytes());
    }
    false
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut difference = 0u8;
    for (left, right) in left.iter().zip(right) {
        difference |= left ^ right;
    }
    difference == 0
}

fn health(state: &ServerState) -> HttpResult<Response> {
    let mut object = Object::new();
    object.insert("status".to_owned(), Value::String("ok".to_owned()));
    object.insert("version".to_owned(), Value::String(VERSION.to_owned()));
    object.insert("target".to_owned(), Value::String(TARGET_TRIPLE.to_owned()));
    object.insert(
        "uptime_seconds".to_owned(),
        Value::Number(Number::Unsigned(state.started.elapsed().as_secs())),
    );
    json_response(200, &Value::Object(object))
}

fn ready(state: &ServerState) -> HttpResult<Response> {
    let (documents, chunks, poisoned) = {
        let store = state
            .store
            .lock()
            .map_err(|_| HttpError::internal("store lock is poisoned"))?;
        (
            store.document_count(),
            store.chunk_count(),
            store.is_poisoned(),
        )
    };
    let mut object = Object::new();
    object.insert("ready".to_owned(), Value::Bool(!poisoned));
    object.insert("storage".to_owned(), Value::Bool(!poisoned));
    object.insert(
        "documents".to_owned(),
        Value::Number(Number::Unsigned(documents as u64)),
    );
    object.insert(
        "chunks".to_owned(),
        Value::Number(Number::Unsigned(chunks as u64)),
    );

    let mut declared = Object::new();
    declared.insert("text".to_owned(), Value::Bool(true));
    declared.insert(
        "image".to_owned(),
        Value::Bool(state.engine.image_available()),
    );
    declared.insert(
        "audio".to_owned(),
        Value::Bool(state.engine.audio_available()),
    );
    object.insert("models".to_owned(), Value::Object(declared));

    let mut loaded = Object::new();
    loaded.insert("text".to_owned(), Value::Bool(true));
    loaded.insert(
        "image".to_owned(),
        Value::Bool(state.engine.image_loaded().map_err(map_error)?),
    );
    loaded.insert(
        "audio".to_owned(),
        Value::Bool(state.engine.audio_loaded().map_err(map_error)?),
    );
    object.insert("loaded".to_owned(), Value::Object(loaded));
    object.insert(
        "text_input".to_owned(),
        Value::Bool(state.engine.text_input_available()),
    );
    json_response(if poisoned { 503 } else { 200 }, &Value::Object(object))
}

fn models(state: &ServerState) -> HttpResult<Response> {
    let mut object = Object::new();
    object.insert(
        "model".to_owned(),
        Value::String(state.engine.model_id().to_owned()),
    );
    object.insert(
        "revision".to_owned(),
        Value::String(state.engine.revision().to_owned()),
    );
    object.insert(
        "version".to_owned(),
        Value::String(state.engine.version().to_owned()),
    );
    object.insert("program".to_owned(), Value::String(PROGRAM_NAME.to_owned()));
    object.insert("target".to_owned(), Value::String(TARGET_TRIPLE.to_owned()));
    object.insert(
        "embedding_dimension".to_owned(),
        Value::Number(Number::Unsigned(state.engine.embedding_dimension() as u64)),
    );
    object.insert(
        "matryoshka_dimensions".to_owned(),
        Value::Array(
            state
                .engine
                .matryoshka_dimensions()
                .iter()
                .map(|dimension| Value::Number(Number::Unsigned(*dimension as u64)))
                .collect(),
        ),
    );
    let mut modalities = Object::new();
    for modality in state.engine.available_modalities() {
        modalities.insert(modality.to_owned(), Value::Bool(true));
    }
    object.insert("modalities".to_owned(), Value::Object(modalities));
    object.insert(
        "text_input".to_owned(),
        Value::Bool(state.engine.text_input_available()),
    );
    object.insert("token_input".to_owned(), Value::Bool(true));
    let mut limits = Object::new();
    limits.insert(
        "max_body_bytes".to_owned(),
        Value::Number(Number::Unsigned(state.config.max_body_bytes as u64)),
    );
    limits.insert(
        "max_connections".to_owned(),
        Value::Number(Number::Unsigned(state.config.max_connections as u64)),
    );
    limits.insert(
        "queue_capacity".to_owned(),
        Value::Number(Number::Unsigned(state.config.queue_capacity as u64)),
    );
    limits.insert(
        "job_workers".to_owned(),
        Value::Number(Number::Unsigned(state.config.job_workers as u64)),
    );
    object.insert("limits".to_owned(), Value::Object(limits));
    json_response(200, &Value::Object(object))
}

fn stats(state: &ServerState) -> HttpResult<Response> {
    let (documents, chunks, recovery) = {
        let store = state
            .store
            .lock()
            .map_err(|_| HttpError::internal("store lock is poisoned"))?;
        (
            store.document_count(),
            store.chunk_count(),
            store.recovery().clone(),
        )
    };
    let jobs = state.jobs.stats().map_err(map_error)?;
    let mut object = Object::new();
    object.insert(
        "uptime_seconds".to_owned(),
        Value::Number(Number::Unsigned(state.started.elapsed().as_secs())),
    );
    object.insert(
        "requests_total".to_owned(),
        Value::Number(Number::Unsigned(
            state.metrics.requests_total.load(Ordering::Relaxed),
        )),
    );
    object.insert(
        "status_2xx".to_owned(),
        Value::Number(Number::Unsigned(
            state.metrics.status_2xx.load(Ordering::Relaxed),
        )),
    );
    object.insert(
        "status_4xx".to_owned(),
        Value::Number(Number::Unsigned(
            state.metrics.status_4xx.load(Ordering::Relaxed),
        )),
    );
    object.insert(
        "status_5xx".to_owned(),
        Value::Number(Number::Unsigned(
            state.metrics.status_5xx.load(Ordering::Relaxed),
        )),
    );
    object.insert(
        "bytes_in".to_owned(),
        Value::Number(Number::Unsigned(
            state.metrics.bytes_in.load(Ordering::Relaxed),
        )),
    );
    object.insert(
        "bytes_out".to_owned(),
        Value::Number(Number::Unsigned(
            state.metrics.bytes_out.load(Ordering::Relaxed),
        )),
    );
    object.insert(
        "active_connections".to_owned(),
        Value::Number(Number::Unsigned(
            state.active_connections.load(Ordering::SeqCst) as u64,
        )),
    );
    object.insert(
        "rejected_connections".to_owned(),
        Value::Number(Number::Unsigned(
            state.metrics.rejected_connections.load(Ordering::Relaxed),
        )),
    );
    object.insert(
        "spawn_failures".to_owned(),
        Value::Number(Number::Unsigned(
            state.metrics.spawn_failures.load(Ordering::Relaxed),
        )),
    );
    let mut job_object = Object::new();
    job_object.insert(
        "queued".to_owned(),
        Value::Number(Number::Unsigned(jobs.queued as u64)),
    );
    job_object.insert(
        "running".to_owned(),
        Value::Number(Number::Unsigned(jobs.running as u64)),
    );
    job_object.insert(
        "succeeded".to_owned(),
        Value::Number(Number::Unsigned(jobs.succeeded as u64)),
    );
    job_object.insert(
        "failed".to_owned(),
        Value::Number(Number::Unsigned(jobs.failed as u64)),
    );
    job_object.insert(
        "cancelled".to_owned(),
        Value::Number(Number::Unsigned(jobs.cancelled as u64)),
    );
    object.insert("jobs".to_owned(), Value::Object(job_object));
    object.insert(
        "documents".to_owned(),
        Value::Number(Number::Unsigned(documents as u64)),
    );
    object.insert(
        "chunks".to_owned(),
        Value::Number(Number::Unsigned(chunks as u64)),
    );
    let mut recovery_object = Object::new();
    recovery_object.insert(
        "replayed_records".to_owned(),
        Value::Number(Number::Unsigned(recovery.replayed_records)),
    );
    recovery_object.insert(
        "snapshot_records".to_owned(),
        Value::Number(Number::Unsigned(recovery.snapshot_records)),
    );
    recovery_object.insert(
        "truncated_bytes".to_owned(),
        Value::Number(Number::Unsigned(recovery.truncated_bytes)),
    );
    object.insert("recovery".to_owned(), Value::Object(recovery_object));
    json_response(200, &Value::Object(object))
}

fn embeddings(state: &ServerState, request: &Request) -> HttpResult<Response> {
    require_json(request)?;
    let body = parse_json_body(request)?;
    let object = expect_object(&body, "request")?;
    reject_unknown(object, &["input", "dimension"], "request")?;
    let input_value = object
        .get("input")
        .ok_or_else(|| HttpError::bad_request("missing input object"))?;
    let input = parse_embedding_input(input_value)?;
    let dimension = parse_dimension(object, state)?;
    let embedding = embed_input(state, &input, dimension).map_err(map_error)?;
    let mut response = Object::new();
    response.insert(
        "model".to_owned(),
        Value::String(state.engine.model_id().to_owned()),
    );
    response.insert(
        "revision".to_owned(),
        Value::String(state.engine.revision().to_owned()),
    );
    response.insert(
        "dimension".to_owned(),
        Value::Number(Number::Unsigned(dimension as u64)),
    );
    response.insert(
        "modality".to_owned(),
        Value::String(input.modality().to_owned()),
    );
    response.insert(
        "embedding".to_owned(),
        Value::Array(
            embedding
                .iter()
                .map(|value| Value::Number(Number::Float(f64::from(*value))))
                .collect(),
        ),
    );
    json_response(200, &Value::Object(response))
}

fn create_document(state: &ServerState, request: &Request) -> HttpResult<Response> {
    require_json(request)?;
    let body = parse_json_body(request)?;
    let object = expect_object(&body, "request")?;
    reject_unknown(
        object,
        &["id", "input", "title", "source", "metadata", "dimension"],
        "request",
    )?;
    let id_value = required_string(object, "id")?;
    let id = DocumentId::new(id_value).map_err(map_error)?;
    let input_value = object
        .get("input")
        .ok_or_else(|| HttpError::bad_request("missing input object"))?;
    let input = parse_document_input(input_value)?;
    let modality = input.modality();
    let content = match input {
        DocumentInput::Text(text) => IngestContent::Text(text),
        DocumentInput::Tokens { ids, text } => IngestContent::Tokens { ids, text },
        DocumentInput::Image { bytes, format } => IngestContent::Image { bytes, format },
        DocumentInput::Audio(bytes) => IngestContent::Audio(bytes),
    };
    let mut document = Document::new(id.clone(), modality);
    if let Some(title) = optional_string(object, "title")? {
        document = document.with_title(title).map_err(map_error)?;
    }
    if let Some(source) = optional_string(object, "source")? {
        document = document.with_source(source).map_err(map_error)?;
    }
    if let Some(metadata_value) = object.get("metadata") {
        document = document.with_metadata(parse_metadata(metadata_value)?);
    }
    let dimension = parse_dimension(object, state)?;
    let task = IngestTask {
        document,
        content,
        dimension,
        vision_budget: state.config.vision_budget,
        store: Arc::clone(&state.store),
        engine: Arc::clone(&state.engine),
    };
    let job_id = match state.jobs.submit(Box::new(task)) {
        Ok(id) => id,
        Err(Error::LimitExceeded(message)) => {
            return Err(HttpError::new(429, "queue_full", message));
        }
        Err(error) => return Err(map_error(error)),
    };
    let snapshot = state
        .jobs
        .status(job_id)
        .map_err(map_error)?
        .ok_or_else(|| HttpError::internal("submitted job is missing from the registry"))?;
    let mut response = Object::new();
    response.insert("job".to_owned(), job_json(&snapshot));
    json_response(202, &Value::Object(response))
}

fn search(state: &ServerState, request: &Request) -> HttpResult<Response> {
    require_json(request)?;
    let body = parse_json_body(request)?;
    let object = expect_object(&body, "request")?;
    reject_unknown(
        object,
        &["query", "mode", "limit", "offset", "filters", "dimension"],
        "request",
    )?;
    let mode = match optional_string(object, "mode")? {
        Some(value) => SearchMode::from_name(value).ok_or_else(|| {
            HttpError::bad_request(format!(
                "unknown search mode {value}; use hybrid, lexical, or vector"
            ))
        })?,
        None => SearchMode::Hybrid,
    };
    let limit = match optional_u64(object, "limit")? {
        Some(value) => {
            let value = usize::try_from(value)
                .map_err(|_| HttpError::bad_request("limit does not fit usize"))?;
            if value == 0 || value > MAX_SEARCH_LIMIT {
                return Err(HttpError::bad_request(format!(
                    "limit must be in 1..={MAX_SEARCH_LIMIT}"
                )));
            }
            value
        }
        None => 10,
    };
    let offset = match optional_u64(object, "offset")? {
        Some(value) => {
            let value = usize::try_from(value)
                .map_err(|_| HttpError::bad_request("offset does not fit usize"))?;
            if value > MAX_SEARCH_OFFSET {
                return Err(HttpError::bad_request(format!(
                    "offset must be at most {MAX_SEARCH_OFFSET}"
                )));
            }
            value
        }
        None => 0,
    };
    let filters = match object.get("filters") {
        Some(value) => parse_filters(value)?,
        None => Filters::new(),
    };
    let query_value = object
        .get("query")
        .ok_or_else(|| HttpError::bad_request("missing query object"))?;
    let query = parse_search_query(query_value, mode)?;
    let dimension = parse_dimension(object, state)?;

    let embedding = if mode == SearchMode::Lexical {
        None
    } else {
        Some(match &query {
            SearchQuery::Text { text, prompt } => state
                .engine
                .embed_text(text, *prompt, dimension)
                .map_err(map_error)?,
            SearchQuery::Tokens { ids, .. } => state
                .engine
                .embed_tokens(ids, dimension)
                .map_err(map_error)?,
            SearchQuery::Image { bytes, format } => state
                .engine
                .embed_image(bytes, *format, dimension, state.config.vision_budget)
                .map_err(map_error)?,
            SearchQuery::Audio { bytes } => state
                .engine
                .embed_audio(bytes, dimension)
                .map_err(map_error)?,
        })
    };

    let query_text = query.text();
    if mode != SearchMode::Vector && query_text.is_none() {
        return Err(HttpError::bad_request(
            "lexical and hybrid search require query text",
        ));
    }

    let store = ReadOnlyStore::open(&state.config.data_dir).map_err(map_error)?;
    let index = Index::build_read_only(&store).map_err(map_error)?;
    let mut search_request = SearchRequest::new(mode, limit);
    search_request.offset = offset;
    search_request.filters = filters;
    search_request.text = if mode == SearchMode::Vector {
        None
    } else {
        query_text
    };
    search_request.embedding = embedding.as_deref();
    let hits = index.search(&search_request).map_err(map_error)?;

    let mut response = Object::new();
    response.insert("mode".to_owned(), Value::String(mode.as_str().to_owned()));
    response.insert(
        "limit".to_owned(),
        Value::Number(Number::Unsigned(limit as u64)),
    );
    response.insert(
        "offset".to_owned(),
        Value::Number(Number::Unsigned(offset as u64)),
    );
    response.insert(
        "hits".to_owned(),
        Value::Array(hits.iter().map(hit_json).collect()),
    );
    json_response(200, &Value::Object(response))
}

fn delete_document(state: &ServerState, request: &Request) -> HttpResult<Response> {
    let segment = request
        .path
        .strip_prefix("/v1/documents/")
        .ok_or_else(|| HttpError::new(404, "not_found", "unknown route"))?;
    if segment.is_empty() || segment.contains('/') {
        return Err(HttpError::new(404, "not_found", "unknown document route"));
    }
    let decoded = http::decode_path_segment(segment)?;
    let id = DocumentId::new(decoded.clone()).map_err(map_error)?;
    let deleted = {
        let mut store = state
            .store
            .lock()
            .map_err(|_| HttpError::internal("store lock is poisoned"))?;
        store.delete_document(&id).map_err(map_error)?
    };
    let mut object = Object::new();
    object.insert("id".to_owned(), Value::String(decoded));
    object.insert("deleted".to_owned(), Value::Bool(deleted));
    json_response(200, &Value::Object(object))
}

fn jobs_route(state: &ServerState, request: &Request) -> HttpResult<Response> {
    let segment = request
        .path
        .strip_prefix("/v1/jobs/")
        .ok_or_else(|| HttpError::new(404, "not_found", "unknown route"))?;
    if segment.is_empty() || segment.contains('/') {
        return Err(HttpError::new(404, "not_found", "unknown job route"));
    }
    let decoded = http::decode_path_segment(segment)?;
    let value = decoded
        .parse::<u64>()
        .map_err(|_| HttpError::bad_request("job id must be a positive integer"))?;
    let id = JobId::new(value).map_err(map_error)?;
    match request.method {
        Method::Get => {
            let snapshot = state
                .jobs
                .status(id)
                .map_err(map_error)?
                .ok_or_else(|| HttpError::new(404, "job_not_found", "unknown job"))?;
            let mut object = Object::new();
            object.insert("job".to_owned(), job_json(&snapshot));
            json_response(200, &Value::Object(object))
        }
        Method::Delete => {
            let outcome = state.jobs.cancel(id).map_err(map_error)?;
            let (status, text) = match outcome {
                CancelOutcome::Requested => (202, "cancellation_requested"),
                CancelOutcome::AlreadyFinished => (200, "already_finished"),
                CancelOutcome::NotFound => {
                    return Err(HttpError::new(404, "job_not_found", "unknown job"));
                }
            };
            let mut object = Object::new();
            object.insert("id".to_owned(), Value::Number(Number::Unsigned(id.get())));
            object.insert("status".to_owned(), Value::String(text.to_owned()));
            json_response(status, &Value::Object(object))
        }
        Method::Post => Err(HttpError::internal(
            "job route reached an unreachable method",
        )),
    }
}

fn shutdown(state: &ServerState, peer: SocketAddr) -> HttpResult<Response> {
    if !peer.ip().is_loopback() {
        return Err(HttpError::new(
            403,
            "forbidden",
            "shutdown is only available from a loopback connection",
        ));
    }
    state.shutdown.store(true, Ordering::SeqCst);
    let mut object = Object::new();
    object.insert(
        "status".to_owned(),
        Value::String("shutting_down".to_owned()),
    );
    json_response(202, &Value::Object(object))
}

#[derive(Debug)]
enum EmbeddingInput {
    Text { text: String, prompt: Prompt },
    Tokens(Vec<u32>),
    Image { bytes: Vec<u8>, format: ImageFormat },
    Audio(Vec<u8>),
}

impl EmbeddingInput {
    fn modality(&self) -> &'static str {
        match self {
            Self::Text { .. } | Self::Tokens(_) => "text",
            Self::Image { .. } => "image",
            Self::Audio(_) => "audio",
        }
    }
}

#[derive(Debug)]
enum SearchQuery {
    Text { text: String, prompt: Prompt },
    Tokens { ids: Vec<u32>, text: Option<String> },
    Image { bytes: Vec<u8>, format: ImageFormat },
    Audio { bytes: Vec<u8> },
}

impl SearchQuery {
    fn text(&self) -> Option<&str> {
        match self {
            Self::Text { text, .. } => Some(text),
            Self::Tokens { text, .. } => text.as_deref(),
            Self::Image { .. } | Self::Audio { .. } => None,
        }
    }
}

#[derive(Debug)]
enum DocumentInput {
    Text(String),
    Tokens { ids: Vec<u32>, text: String },
    Image { bytes: Vec<u8>, format: ImageFormat },
    Audio(Vec<u8>),
}

impl DocumentInput {
    fn modality(&self) -> Modality {
        match self {
            Self::Text(_) | Self::Tokens { .. } => Modality::Text,
            Self::Image { .. } => Modality::Image,
            Self::Audio(_) => Modality::Audio,
        }
    }
}

struct IngestTask {
    document: Document,
    content: IngestContent,
    dimension: usize,
    vision_budget: usize,
    store: Arc<Mutex<Store>>,
    engine: Arc<Engine>,
}

enum IngestContent {
    Text(String),
    Tokens { ids: Vec<u32>, text: String },
    Image { bytes: Vec<u8>, format: ImageFormat },
    Audio(Vec<u8>),
}

impl JobTask for IngestTask {
    fn kind(&self) -> &'static str {
        "ingest"
    }

    fn run(&mut self, cancel: &CancelFlag) -> Result<JobOutcome> {
        if cancel.is_cancelled() {
            return Err(Error::usage("job cancelled"));
        }
        let (embedding, chunk_text) = match &self.content {
            IngestContent::Text(text) => {
                let embedding = self
                    .engine
                    .embed_text(text, Prompt::Document, self.dimension)?;
                (embedding, text.clone())
            }
            IngestContent::Tokens { ids, text } => {
                let embedding = self.engine.embed_tokens(ids, self.dimension)?;
                (embedding, text.clone())
            }
            IngestContent::Image { bytes, format } => {
                let embedding =
                    self.engine
                        .embed_image(bytes, *format, self.dimension, self.vision_budget)?;
                (embedding, String::new())
            }
            IngestContent::Audio(bytes) => {
                let embedding = self.engine.embed_audio(bytes, self.dimension)?;
                (embedding, String::new())
            }
        };
        if cancel.is_cancelled() {
            return Err(Error::usage("job cancelled"));
        }
        let mut store = self
            .store
            .lock()
            .map_err(|_| Error::internal("store lock is poisoned"))?;
        if cancel.is_cancelled() {
            return Err(Error::usage("job cancelled"));
        }
        let chunk_ids = store.put_document(
            self.document.clone(),
            vec![ChunkInput::new(chunk_text, embedding)],
        )?;
        Ok(JobOutcome::new(1, chunk_ids.len()))
    }
}

fn embed_input(state: &ServerState, input: &EmbeddingInput, dimension: usize) -> Result<Vec<f32>> {
    match input {
        EmbeddingInput::Text { text, prompt } => state.engine.embed_text(text, *prompt, dimension),
        EmbeddingInput::Tokens(ids) => state.engine.embed_tokens(ids, dimension),
        EmbeddingInput::Image { bytes, format } => {
            state
                .engine
                .embed_image(bytes, *format, dimension, state.config.vision_budget)
        }
        EmbeddingInput::Audio(bytes) => state.engine.embed_audio(bytes, dimension),
    }
}

fn parse_embedding_input(value: &Value) -> HttpResult<EmbeddingInput> {
    let object = expect_object(value, "input")?;
    let kind = required_string(object, "type")?;
    match kind {
        "text" => {
            reject_unknown(object, &["type", "text", "prompt"], "input")?;
            let text = required_string(object, "text")?;
            if text.len() > MAX_TEXT_INPUT_BYTES {
                return Err(HttpError::too_large(format!(
                    "text input is {} bytes, above the {MAX_TEXT_INPUT_BYTES}-byte cap",
                    text.len()
                )));
            }
            let prompt = parse_prompt(object)?;
            Ok(EmbeddingInput::Text {
                text: text.to_owned(),
                prompt,
            })
        }
        "tokens" => {
            reject_unknown(object, &["type", "ids"], "input")?;
            Ok(EmbeddingInput::Tokens(parse_token_ids(object)?))
        }
        "image" => {
            reject_unknown(object, &["type", "data", "encoding"], "input")?;
            let bytes = parse_media_data(object, MAX_MEDIA_BYTES)?;
            let format = Engine::detect_image_format(&bytes).map_err(map_error)?;
            Ok(EmbeddingInput::Image { bytes, format })
        }
        "audio" => {
            reject_unknown(object, &["type", "data", "encoding"], "input")?;
            let bytes = parse_media_data(object, MAX_MEDIA_BYTES)?;
            Ok(EmbeddingInput::Audio(bytes))
        }
        other => Err(HttpError::bad_request(format!(
            "unknown input type {other}; use text, tokens, image, or audio"
        ))),
    }
}

fn parse_document_input(value: &Value) -> HttpResult<DocumentInput> {
    let object = expect_object(value, "input")?;
    let kind = required_string(object, "type")?;
    match kind {
        "text" => {
            reject_unknown(object, &["type", "text"], "input")?;
            let text = required_string(object, "text")?;
            if text.len() > MAX_TEXT_INPUT_BYTES {
                return Err(HttpError::too_large(format!(
                    "text input is {} bytes, above the {MAX_TEXT_INPUT_BYTES}-byte cap",
                    text.len()
                )));
            }
            Ok(DocumentInput::Text(text.to_owned()))
        }
        "tokens" => {
            reject_unknown(object, &["type", "ids", "text"], "input")?;
            let ids = parse_token_ids(object)?;
            let text = optional_string(object, "text")?.unwrap_or("").to_owned();
            if text.len() > MAX_TEXT_INPUT_BYTES {
                return Err(HttpError::too_large(format!(
                    "chunk text is {} bytes, above the {MAX_TEXT_INPUT_BYTES}-byte cap",
                    text.len()
                )));
            }
            Ok(DocumentInput::Tokens { ids, text })
        }
        "image" => {
            reject_unknown(object, &["type", "data", "encoding"], "input")?;
            let bytes = parse_media_data(object, MAX_MEDIA_BYTES)?;
            let format = Engine::detect_image_format(&bytes).map_err(map_error)?;
            Ok(DocumentInput::Image { bytes, format })
        }
        "audio" => {
            reject_unknown(object, &["type", "data", "encoding"], "input")?;
            let bytes = parse_media_data(object, MAX_MEDIA_BYTES)?;
            Ok(DocumentInput::Audio(bytes))
        }
        other => Err(HttpError::bad_request(format!(
            "unknown input type {other}; use text, tokens, image, or audio"
        ))),
    }
}

fn parse_search_query(value: &Value, mode: SearchMode) -> HttpResult<SearchQuery> {
    let object = expect_object(value, "query")?;
    let kind = required_string(object, "type")?;
    match kind {
        "text" => {
            reject_unknown(object, &["type", "text", "prompt"], "query")?;
            let text = required_string(object, "text")?;
            if text.len() > MAX_TEXT_INPUT_BYTES {
                return Err(HttpError::too_large(format!(
                    "query text is {} bytes, above the {MAX_TEXT_INPUT_BYTES}-byte cap",
                    text.len()
                )));
            }
            Ok(SearchQuery::Text {
                text: text.to_owned(),
                prompt: parse_prompt(object)?,
            })
        }
        "tokens" => {
            reject_unknown(object, &["type", "ids", "text"], "query")?;
            let ids = parse_token_ids(object)?;
            let text = optional_string(object, "text")?.map(str::to_owned);
            if mode != SearchMode::Vector && text.is_none() {
                return Err(HttpError::bad_request(
                    "lexical and hybrid token queries require text for BM25",
                ));
            }
            Ok(SearchQuery::Tokens { ids, text })
        }
        "image" => {
            reject_unknown(object, &["type", "data", "encoding"], "query")?;
            if mode != SearchMode::Vector {
                return Err(HttpError::bad_request("image queries require vector mode"));
            }
            let bytes = parse_media_data(object, MAX_MEDIA_BYTES)?;
            let format = Engine::detect_image_format(&bytes).map_err(map_error)?;
            Ok(SearchQuery::Image { bytes, format })
        }
        "audio" => {
            reject_unknown(object, &["type", "data", "encoding"], "query")?;
            if mode != SearchMode::Vector {
                return Err(HttpError::bad_request("audio queries require vector mode"));
            }
            let bytes = parse_media_data(object, MAX_MEDIA_BYTES)?;
            Ok(SearchQuery::Audio { bytes })
        }
        other => Err(HttpError::bad_request(format!(
            "unknown query type {other}; use text, tokens, image, or audio"
        ))),
    }
}

fn parse_media_data(object: &Object, max_bytes: usize) -> HttpResult<Vec<u8>> {
    if let Some(encoding) = optional_string(object, "encoding")?
        && !encoding.eq_ignore_ascii_case("base64")
    {
        return Err(HttpError::bad_request(format!(
            "unsupported media encoding {encoding}; use base64"
        )));
    }
    let data = required_string(object, "data")?;
    base64::decode(data, max_bytes).map_err(map_error)
}

fn parse_prompt(object: &Object) -> HttpResult<Prompt> {
    match optional_string(object, "prompt")? {
        Some(value) => Prompt::from_name(value).ok_or_else(|| {
            HttpError::bad_request(format!(
                "unknown prompt {value}; use SearchQuery, Document, Classification, Clustering, CodeRetrieval, FactChecking, SentenceSimilarity, or QuestionAnswering"
            ))
        }),
        None => Ok(Prompt::SearchQuery),
    }
}

fn parse_token_ids(object: &Object) -> HttpResult<Vec<u32>> {
    let value = object
        .get("ids")
        .ok_or_else(|| HttpError::bad_request("missing ids array"))?;
    let array = value
        .as_array()
        .ok_or_else(|| HttpError::bad_request("ids must be an array"))?;
    if array.is_empty() {
        return Err(HttpError::bad_request("ids must not be empty"));
    }
    if array.len() > MAX_TOKEN_IDS {
        return Err(HttpError::too_large(format!(
            "ids has {} entries, above the {MAX_TOKEN_IDS}-entry cap",
            array.len()
        )));
    }
    let mut ids = Vec::with_capacity(array.len());
    for value in array {
        let number = value
            .as_number()
            .ok_or_else(|| HttpError::bad_request("each token id must be a number"))?;
        let value = number
            .as_u64()
            .ok_or_else(|| HttpError::bad_request("token id must be a non-negative integer"))?;
        let id = u32::try_from(value)
            .map_err(|_| HttpError::bad_request("token id does not fit u32"))?;
        ids.push(id);
    }
    Ok(ids)
}

fn parse_metadata(value: &Value) -> HttpResult<Metadata> {
    let object = expect_object(value, "metadata")?;
    if object.len() > crate::storage::model::MAX_METADATA_ENTRIES {
        return Err(HttpError::too_large(format!(
            "metadata has {} entries, above the {}-entry cap",
            object.len(),
            crate::storage::model::MAX_METADATA_ENTRIES
        )));
    }
    let mut metadata = Metadata::new();
    for (key, value) in object {
        let value = value
            .as_str()
            .ok_or_else(|| HttpError::bad_request("metadata values must be strings"))?;
        metadata.insert(key, value).map_err(map_error)?;
    }
    Ok(metadata)
}

fn parse_filters(value: &Value) -> HttpResult<Filters> {
    let object = expect_object(value, "filters")?;
    reject_unknown(object, &["modality", "metadata"], "filters")?;
    let mut filters = Filters::new();
    if let Some(modality) = optional_string(object, "modality")? {
        let modality = Modality::from_name(modality).ok_or_else(|| {
            HttpError::bad_request(format!(
                "unknown modality {modality}; use text, image, or audio"
            ))
        })?;
        filters = filters.with_modality(modality);
    }
    if let Some(metadata_value) = object.get("metadata") {
        let metadata = expect_object(metadata_value, "filters.metadata")?;
        for (key, value) in metadata {
            let value = value
                .as_str()
                .ok_or_else(|| HttpError::bad_request("filter values must be strings"))?;
            filters.add_metadata(key, value).map_err(map_error)?;
        }
    }
    Ok(filters)
}

fn parse_dimension(object: &Object, state: &ServerState) -> HttpResult<usize> {
    let dimension = match optional_u64(object, "dimension")? {
        Some(value) => usize::try_from(value)
            .map_err(|_| HttpError::bad_request("dimension does not fit usize"))?,
        None => state.config.dimension,
    };
    state
        .engine
        .validate_dimension(dimension)
        .map_err(map_error)?;
    Ok(dimension)
}

fn parse_json_body(request: &Request) -> HttpResult<Value> {
    let text = std::str::from_utf8(&request.body)
        .map_err(|_| HttpError::bad_request("request body is not valid UTF-8"))?;
    json::parse_with_limits(text, &api_json_limits()).map_err(map_json_error)
}

fn api_json_limits() -> Limits {
    Limits {
        max_input_bytes: MAX_BODY_BYTES,
        max_depth: 32,
        // A base64 media string is larger than the decoded media cap; the
        // body cap and the base64 decoder enforce the actual media limit.
        max_string_bytes: MAX_BODY_BYTES,
        max_container_entries: 16_384,
        max_total_values: 100_000,
    }
}

fn require_json(request: &Request) -> HttpResult<()> {
    let content_type = request.header("content-type").ok_or_else(|| {
        HttpError::new(
            415,
            "unsupported_media_type",
            "Content-Type must be application/json",
        )
    })?;
    let mut parts = content_type.split(';');
    let essence = parts.next().unwrap_or("").trim();
    if !essence.eq_ignore_ascii_case("application/json") {
        return Err(HttpError::new(
            415,
            "unsupported_media_type",
            "Content-Type must be application/json",
        ));
    }
    for parameter in parts {
        let parameter = parameter.trim();
        if parameter.is_empty() {
            continue;
        }
        let Some((name, value)) = parameter.split_once('=') else {
            return Err(HttpError::new(
                415,
                "unsupported_media_type",
                "malformed Content-Type parameter",
            ));
        };
        if name.trim().eq_ignore_ascii_case("charset") && value.trim().eq_ignore_ascii_case("utf-8")
        {
            continue;
        }
        return Err(HttpError::new(
            415,
            "unsupported_media_type",
            "only the utf-8 charset is supported",
        ));
    }
    Ok(())
}

fn json_response(status: u16, value: &Value) -> HttpResult<Response> {
    Response::json(status, value).map_err(|error| HttpError::internal(error.to_string()))
}

fn error_response(error: HttpError) -> Response {
    let mut error_object = Object::new();
    error_object.insert("code".to_owned(), Value::String(error.code.to_owned()));
    error_object.insert("message".to_owned(), Value::String(error.message));
    let mut object = Object::new();
    object.insert("error".to_owned(), Value::Object(error_object));
    match Response::json(error.status, &Value::Object(object)) {
        Ok(response) => response,
        Err(_) => Response::empty(500),
    }
}

fn map_json_error(error: Error) -> HttpError {
    match error {
        Error::LimitExceeded(message) => HttpError::too_large(message),
        Error::Corrupt(message) => HttpError::bad_request(message),
        Error::Usage(message) => HttpError::bad_request(message),
        Error::Unsupported(message) => HttpError::unsupported(message),
        Error::Internal(message) => HttpError::internal(message),
        Error::Io(error) => HttpError::internal(error.to_string()),
    }
}

fn map_error(error: Error) -> HttpError {
    match error {
        Error::Usage(message) => HttpError::bad_request(message),
        Error::Corrupt(message) => HttpError::new(400, "invalid_input", message),
        Error::LimitExceeded(message) => HttpError::too_large(message),
        Error::Unsupported(message) => HttpError::unsupported(message),
        Error::Internal(message) => HttpError::internal(message),
        Error::Io(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
            HttpError::new(503, "unavailable", error.to_string())
        }
        Error::Io(error) => HttpError::internal(error.to_string()),
    }
}

fn expect_object<'a>(value: &'a Value, context: &str) -> HttpResult<&'a Object> {
    value
        .as_object()
        .ok_or_else(|| HttpError::bad_request(format!("{context} must be a JSON object")))
}

fn reject_unknown(object: &Object, allowed: &[&str], context: &str) -> HttpResult<()> {
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(HttpError::bad_request(format!(
                "unknown field {key} in {context}"
            )));
        }
    }
    Ok(())
}

fn required_string<'a>(object: &'a Object, key: &str) -> HttpResult<&'a str> {
    object
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| HttpError::bad_request(format!("missing or invalid string field {key}")))
}

fn optional_string<'a>(object: &'a Object, key: &str) -> HttpResult<Option<&'a str>> {
    match object.get(key) {
        Some(value) => value
            .as_str()
            .map(Some)
            .ok_or_else(|| HttpError::bad_request(format!("field {key} must be a string"))),
        None => Ok(None),
    }
}

fn optional_u64(object: &Object, key: &str) -> HttpResult<Option<u64>> {
    match object.get(key) {
        Some(value) => value
            .as_number()
            .and_then(Number::as_u64)
            .map(Some)
            .ok_or_else(|| {
                HttpError::bad_request(format!("field {key} must be a non-negative integer"))
            }),
        None => Ok(None),
    }
}

fn job_json(snapshot: &JobSnapshot) -> Value {
    let mut object = Object::new();
    object.insert(
        "id".to_owned(),
        Value::Number(Number::Unsigned(snapshot.id.get())),
    );
    object.insert("kind".to_owned(), Value::String(snapshot.kind.clone()));
    object.insert(
        "status".to_owned(),
        Value::String(snapshot.status.as_str().to_owned()),
    );
    object.insert(
        "created_ms".to_owned(),
        Value::Number(Number::Unsigned(snapshot.created_ms)),
    );
    object.insert(
        "started_ms".to_owned(),
        snapshot
            .started_ms
            .map_or(Value::Null, |value| Value::Number(Number::Unsigned(value))),
    );
    object.insert(
        "finished_ms".to_owned(),
        snapshot
            .finished_ms
            .map_or(Value::Null, |value| Value::Number(Number::Unsigned(value))),
    );
    object.insert(
        "documents".to_owned(),
        Value::Number(Number::Unsigned(snapshot.documents as u64)),
    );
    object.insert(
        "chunks".to_owned(),
        Value::Number(Number::Unsigned(snapshot.chunks as u64)),
    );
    object.insert(
        "message".to_owned(),
        snapshot
            .message
            .as_ref()
            .map_or(Value::Null, |message| Value::String(message.clone())),
    );
    Value::Object(object)
}

fn hit_json(hit: &crate::index::SearchHit<'_>) -> Value {
    let mut object = Object::new();
    object.insert(
        "document_id".to_owned(),
        Value::String(hit.document_id.as_str().to_owned()),
    );
    object.insert(
        "chunk_id".to_owned(),
        Value::Number(Number::Unsigned(hit.chunk_id.get())),
    );
    object.insert(
        "ordinal".to_owned(),
        Value::Number(Number::Unsigned(u64::from(hit.ordinal))),
    );
    object.insert("score".to_owned(), Value::Number(Number::Float(hit.score)));
    object.insert(
        "lexical_rank".to_owned(),
        hit.lexical_rank.map_or(Value::Null, |rank| {
            Value::Number(Number::Unsigned(rank as u64))
        }),
    );
    object.insert(
        "vector_rank".to_owned(),
        hit.vector_rank.map_or(Value::Null, |rank| {
            Value::Number(Number::Unsigned(rank as u64))
        }),
    );
    object.insert("text".to_owned(), Value::String(hit.text.to_owned()));
    object.insert(
        "title".to_owned(),
        hit.title
            .map_or(Value::Null, |title| Value::String(title.to_owned())),
    );
    object.insert(
        "source".to_owned(),
        hit.source
            .map_or(Value::Null, |source| Value::String(source.to_owned())),
    );
    object.insert(
        "modality".to_owned(),
        Value::String(hit.modality.as_str().to_owned()),
    );
    let mut metadata = Object::new();
    for (key, value) in hit.metadata.iter() {
        metadata.insert(key.to_owned(), Value::String(value.to_owned()));
    }
    object.insert("metadata".to_owned(), Value::Object(metadata));
    Value::Object(object)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_a_non_loopback_bind_without_a_token() {
        let config = ServerConfig {
            bind: "0.0.0.0:8080".parse().expect("address"),
            ..ServerConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn requires_a_long_token_for_a_lan_bind() {
        let config = ServerConfig {
            bind: "0.0.0.0:8080".parse().expect("address"),
            access_token: Some("short".to_owned()),
            ..ServerConfig::default()
        };
        assert!(config.validate().is_err());
        let config = ServerConfig {
            bind: "0.0.0.0:8080".parse().expect("address"),
            access_token: Some("0123456789abcdef".to_owned()),
            ..ServerConfig::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn constant_time_comparison_matches_only_identical_tokens() {
        assert!(constant_time_eq(b"token", b"token"));
        assert!(!constant_time_eq(b"token", b"tokeN"));
        assert!(!constant_time_eq(b"token", b"token "));
    }

    #[test]
    fn route_table_lists_allowed_methods() {
        assert_eq!(allowed_methods("/health"), Some("GET"));
        assert_eq!(allowed_methods("/v1/documents/doc-1"), Some("DELETE"));
        assert_eq!(allowed_methods("/v1/jobs/1"), Some("GET, DELETE"));
        assert_eq!(allowed_methods("/nope"), None);
    }

    #[test]
    fn token_parser_rejects_non_integer_ids() {
        let value = json::parse(r#"{"ids":[1,2]}"#).expect("json");
        let object = value.as_object().expect("object");
        assert_eq!(parse_token_ids(object).expect("ids"), vec![1, 2]);
        let value = json::parse(r#"{"ids":[1,"two"]}"#).expect("json");
        assert!(parse_token_ids(value.as_object().expect("object")).is_err());
    }
}
