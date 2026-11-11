//! Document store with writer locking, deterministic replay, and recovery.
//!
//! A store owns one data directory. Opening it acquires an exclusive lock,
//! scans the journal, validates every transaction, and replays the committed
//! prefix into memory. A torn tail is truncated only after the committed
//! records have been applied; corruption in the middle is reported instead of
//! being discarded. Writes append one atomic transaction record and synchronize
//! it before the in-memory state changes.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions, TryLockError};
use std::io;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::storage::journal::{self, JOURNAL_FILE_NAME, Journal};
use crate::storage::model::{
    Chunk, ChunkId, ChunkInput, Document, DocumentId, MAX_CHUNKS_PER_DOCUMENT,
    MAX_EMBEDDING_DIMENSION, Operation, Transaction,
};
use crate::storage::snapshot::{self, SNAPSHOT_FILE_NAME, SnapshotHeader};

/// Writer lock file name inside a data directory.
pub const LOCK_FILE_NAME: &str = "writer.lock";
/// Shared/exclusive lock file serializing compaction against read-only views.
pub const COMPACTION_LOCK_FILE_NAME: &str = "compaction.lock";
/// Temporary journal name used during compaction.
pub const JOURNAL_TMP_FILE_NAME: &str = "journal.tmp";
/// Maximum number of documents held in one store.
pub const MAX_DOCUMENTS: usize = 1_000_000;
/// Maximum number of chunks held in one store.
pub const MAX_CHUNKS: usize = 10_000_000;

/// Recovery information produced by opening a store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recovery {
    /// Number of committed journal transactions replayed after the snapshot.
    pub replayed_records: u64,
    /// Number of snapshot transaction records applied.
    pub snapshot_records: u64,
    /// Number of trailing journal bytes discarded as an uncommitted tail.
    pub truncated_bytes: u64,
}

/// Read-only verification report for a data directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verification {
    /// Number of journal transactions applied after the snapshot.
    pub records: u64,
    /// Number of snapshot transaction records applied.
    pub snapshot_records: u64,
    /// Number of documents reconstructed from the committed state.
    pub documents: usize,
    /// Number of chunks reconstructed from the committed state.
    pub chunks: usize,
    /// Number of trailing journal bytes that are not part of a committed record.
    pub truncated_bytes: u64,
}

/// Summary of a successful compaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Compaction {
    /// Number of transaction records written to the snapshot.
    pub snapshot_records: u64,
    /// Snapshot size in bytes.
    pub snapshot_bytes: u64,
    /// Number of documents materialized.
    pub documents: usize,
    /// Number of chunks materialized.
    pub chunks: usize,
}

/// Persistent document store.
#[derive(Debug)]
pub struct Store {
    dir: PathBuf,
    _lock: File,
    journal: Journal,
    state: StoreState,
    next_sequence: u64,
    recovery: Recovery,
    poisoned: bool,
}

#[derive(Debug)]
struct StoreState {
    documents: BTreeMap<DocumentId, Document>,
    chunks: BTreeMap<ChunkId, Chunk>,
    chunks_by_document: BTreeMap<DocumentId, Vec<ChunkId>>,
    next_chunk_id: u64,
    embedding_dimension: Option<usize>,
}

impl Default for StoreState {
    fn default() -> Self {
        Self {
            documents: BTreeMap::new(),
            chunks: BTreeMap::new(),
            chunks_by_document: BTreeMap::new(),
            next_chunk_id: 1,
            embedding_dimension: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ApplyMode {
    Live,
    Replay,
    Snapshot,
}

struct LoadedState {
    state: StoreState,
    next_sequence: u64,
    journal_valid_end: u64,
    journal_records: u64,
    snapshot_records: u64,
    truncated_bytes: u64,
}

/// Read-only view of a store, safe to open while a writer holds the lock.
///
/// The view replays the snapshot and the committed journal prefix without
/// acquiring writer.lock and without modifying either file. A concurrent
/// append can produce a torn journal tail; that tail is reported in the
/// recovery information and is not applied.
#[derive(Debug)]
pub struct ReadOnlyStore {
    dir: PathBuf,
    state: StoreState,
    recovery: Recovery,
}

impl ReadOnlyStore {
    /// Opens a read-only view of an existing data directory.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let metadata = fs::metadata(&dir)?;
        if !metadata.is_dir() {
            return Err(Error::corrupt(format!(
                "{} is not a directory",
                dir.display()
            )));
        }
        let compaction_lock_path = dir.join(COMPACTION_LOCK_FILE_NAME);
        let compaction_lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&compaction_lock_path)?;
        compaction_lock.lock_shared()?;
        let loaded = load_state(&dir)?;
        drop(compaction_lock);
        let recovery = Recovery {
            replayed_records: loaded.journal_records,
            snapshot_records: loaded.snapshot_records,
            truncated_bytes: loaded.truncated_bytes,
        };
        Ok(Self {
            dir,
            state: loaded.state,
            recovery,
        })
    }

    /// Data directory.
    pub fn data_dir(&self) -> &Path {
        &self.dir
    }

    /// Recovery information from this read-only open.
    pub fn recovery(&self) -> &Recovery {
        &self.recovery
    }

    /// Number of documents currently stored.
    pub fn document_count(&self) -> usize {
        self.state.documents.len()
    }

    /// Number of chunks currently stored.
    pub fn chunk_count(&self) -> usize {
        self.state.chunks.len()
    }

    /// Next chunk identifier that would be assigned by a writer.
    pub fn next_chunk_id(&self) -> u64 {
        self.state.next_chunk_id
    }

    /// Embedding dimension shared by every stored chunk, if any chunk exists.
    pub fn embedding_dimension(&self) -> Option<usize> {
        self.state.embedding_dimension
    }

    /// Looks up a document by id.
    pub fn document(&self, id: &DocumentId) -> Option<&Document> {
        self.state.documents.get(id)
    }

    /// Looks up a chunk by id.
    pub fn chunk(&self, id: ChunkId) -> Option<&Chunk> {
        self.state.chunks.get(&id)
    }

    /// Iterates documents in deterministic id order.
    pub fn documents(&self) -> impl Iterator<Item = &Document> {
        self.state.documents.values()
    }

    /// Iterates chunks in deterministic id order.
    pub fn chunks(&self) -> impl Iterator<Item = &Chunk> {
        self.state.chunks.values()
    }

    /// Chunk ids belonging to a document, in ordinal order.
    pub fn document_chunks(&self, id: &DocumentId) -> &[ChunkId] {
        self.state
            .chunks_by_document
            .get(id)
            .map_or(&[], Vec::as_slice)
    }
}

impl Store {
    /// Opens or creates a store in the given directory.
    ///
    /// The directory must be dedicated to this store. Opening acquires an
    /// exclusive writer lock; a second writer receives an I/O error with
    /// ErrorKind::WouldBlock. Recovery applies the committed journal prefix and
    /// truncates only an uncommitted tail.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        sync_directory(&dir)?;

        let lock_path = dir.join(LOCK_FILE_NAME);
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)?;
        sync_directory(&dir)?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => {
                return Err(Error::Io(io::Error::new(
                    io::ErrorKind::WouldBlock,
                    format!(
                        "data directory {} is locked by another writer",
                        dir.display()
                    ),
                )));
            }
            Err(TryLockError::Error(error)) => return Err(Error::Io(error)),
        }

        let compaction_lock_path = dir.join(COMPACTION_LOCK_FILE_NAME);
        let compaction_lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&compaction_lock_path)?;
        drop(compaction_lock);
        sync_directory(&dir)?;

        let loaded = load_state(&dir)?;
        let journal_path = dir.join(JOURNAL_FILE_NAME);
        let journal = Journal::open(
            &journal_path,
            loaded.journal_valid_end,
            loaded.truncated_bytes > 0,
        )?;
        let recovery = Recovery {
            replayed_records: loaded.journal_records,
            snapshot_records: loaded.snapshot_records,
            truncated_bytes: loaded.truncated_bytes,
        };

        Ok(Self {
            dir,
            _lock: lock,
            journal,
            state: loaded.state,
            next_sequence: loaded.next_sequence,
            recovery,
            poisoned: false,
        })
    }

    /// Opens a read-only view without acquiring the writer lock.
    ///
    /// The view is safe to use while a writer holds writer.lock. It replays the
    /// snapshot and the committed journal prefix; a concurrent torn append is
    /// reported but not applied.
    pub fn open_read_only(dir: impl AsRef<Path>) -> Result<ReadOnlyStore> {
        ReadOnlyStore::open(dir)
    }

    /// Verifies a data directory without acquiring the writer lock or modifying
    /// the snapshot or journal.
    ///
    /// The committed state is decoded and applied to a temporary state to
    /// validate referential integrity and checksums. An uncommitted journal tail
    /// is reported in truncated_bytes but is not removed.
    pub fn verify(dir: impl AsRef<Path>) -> Result<Verification> {
        let dir = dir.as_ref();
        let metadata = fs::metadata(dir)?;
        if !metadata.is_dir() {
            return Err(Error::corrupt(format!(
                "{} is not a directory",
                dir.display()
            )));
        }
        let compaction_lock_path = dir.join(COMPACTION_LOCK_FILE_NAME);
        let compaction_lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&compaction_lock_path)?;
        compaction_lock.lock_shared()?;
        let loaded = load_state(dir)?;
        drop(compaction_lock);
        Ok(Verification {
            records: loaded.journal_records,
            snapshot_records: loaded.snapshot_records,
            documents: loaded.state.documents.len(),
            chunks: loaded.state.chunks.len(),
            truncated_bytes: loaded.truncated_bytes,
        })
    }

    /// Materializes the current state into a snapshot and replaces the journal
    /// with an empty file.
    ///
    /// The snapshot is written to snapshot.tmp, synchronized, and renamed before
    /// the journal is replaced. A crash before the snapshot rename leaves the
    /// old snapshot and journal intact. A crash after the snapshot rename but
    /// before the journal rename leaves a valid snapshot and the old journal;
    /// the next open skips journal records already covered by the snapshot base
    /// sequence.
    pub fn compact(&mut self) -> Result<Compaction> {
        if self.poisoned {
            return Err(Error::internal(
                "store is poisoned after a failed post-commit apply; reopen it",
            ));
        }

        let compaction_lock_path = self.dir.join(COMPACTION_LOCK_FILE_NAME);
        let compaction_lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&compaction_lock_path)?;
        compaction_lock.lock()?;

        let base_sequence = self
            .next_sequence
            .checked_sub(1)
            .ok_or_else(|| Error::internal("store sequence is zero"))?;
        let document_count = self.state.documents.len();
        let chunk_count = self.state.chunks.len();
        let header = SnapshotHeader {
            base_sequence,
            next_chunk_id: self.state.next_chunk_id,
            embedding_dimension: self.state.embedding_dimension,
            document_count: document_count as u64,
            chunk_count: chunk_count as u64,
            record_count: document_count as u64,
        };
        let header_bytes = snapshot::encode_header(&header)?;
        let mut bytes = Vec::with_capacity(header_bytes.len());
        bytes.extend_from_slice(&header_bytes);

        let mut snapshot_sequence = 1u64;
        for document in self.state.documents.values() {
            let chunk_ids = self.state.document_chunks(document.id());
            let mut operations = Vec::with_capacity(1 + chunk_ids.len());
            operations.push(Operation::PutDocument(document.clone()));
            for chunk_id in chunk_ids {
                let chunk = self.state.chunks.get(chunk_id).ok_or_else(|| {
                    Error::internal(format!(
                        "document {} references missing chunk {chunk_id}",
                        document.id()
                    ))
                })?;
                operations.push(Operation::PutChunk(chunk.clone()));
            }
            let transaction = Transaction::new(operations)?;
            let payload = transaction.encode()?;
            bytes.extend_from_slice(&journal::encode_record(snapshot_sequence, &payload));
            snapshot_sequence = snapshot_sequence
                .checked_add(1)
                .ok_or_else(|| Error::internal("snapshot sequence overflow"))?;
        }
        if bytes.len() as u64 > snapshot::MAX_SNAPSHOT_BYTES {
            return Err(Error::limit_exceeded(format!(
                "snapshot is {} bytes, above the {}-byte cap",
                bytes.len(),
                snapshot::MAX_SNAPSHOT_BYTES
            )));
        }

        snapshot::write_atomic(&self.dir, &bytes)?;

        let journal_path = self.dir.join(JOURNAL_FILE_NAME);
        let journal_tmp = self.dir.join(JOURNAL_TMP_FILE_NAME);
        let journal_result = (|| -> Result<()> {
            let file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&journal_tmp)?;
            file.sync_all()?;
            drop(file);
            fs::rename(&journal_tmp, &journal_path)?;
            sync_directory(&self.dir)?;
            Ok(())
        })();
        if let Err(error) = journal_result {
            let _ = fs::remove_file(&journal_tmp);
            return Err(error);
        }

        let journal = match Journal::open(&journal_path, 0, false) {
            Ok(journal) => journal,
            Err(error) => {
                self.poisoned = true;
                return Err(error);
            }
        };
        self.journal = journal;
        self.next_sequence = base_sequence
            .checked_add(1)
            .ok_or_else(|| Error::internal("journal sequence overflow"))?;
        self.recovery = Recovery {
            replayed_records: 0,
            snapshot_records: header.record_count,
            truncated_bytes: 0,
        };
        Ok(Compaction {
            snapshot_records: header.record_count,
            snapshot_bytes: bytes.len() as u64,
            documents: document_count,
            chunks: chunk_count,
        })
    }

    /// Data directory.
    pub fn data_dir(&self) -> &Path {
        &self.dir
    }

    /// Recovery information from opening the store.
    pub fn recovery(&self) -> &Recovery {
        &self.recovery
    }

    /// Number of documents currently stored.
    pub fn document_count(&self) -> usize {
        self.state.documents.len()
    }

    /// Number of chunks currently stored.
    pub fn chunk_count(&self) -> usize {
        self.state.chunks.len()
    }

    /// Next chunk identifier that will be assigned.
    pub fn next_chunk_id(&self) -> u64 {
        self.state.next_chunk_id
    }

    /// Embedding dimension shared by every stored chunk, if any chunk exists.
    pub fn embedding_dimension(&self) -> Option<usize> {
        self.state.embedding_dimension
    }

    /// Indicates whether a failed post-commit apply poisoned the store.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Looks up a document by id.
    pub fn document(&self, id: &DocumentId) -> Option<&Document> {
        self.state.documents.get(id)
    }

    /// Looks up a chunk by id.
    pub fn chunk(&self, id: ChunkId) -> Option<&Chunk> {
        self.state.chunks.get(&id)
    }

    /// Iterates documents in deterministic id order.
    pub fn documents(&self) -> impl Iterator<Item = &Document> {
        self.state.documents.values()
    }

    /// Iterates chunks in deterministic id order.
    pub fn chunks(&self) -> impl Iterator<Item = &Chunk> {
        self.state.chunks.values()
    }

    /// Chunk ids belonging to a document, in ordinal order.
    pub fn document_chunks(&self, id: &DocumentId) -> &[ChunkId] {
        self.state
            .chunks_by_document
            .get(id)
            .map_or(&[], Vec::as_slice)
    }

    /// Inserts a document and its chunks in one atomic transaction.
    ///
    /// The store assigns chunk ids and ordinals. Every embedding must match the
    /// store embedding dimension and is normalized before it is written. The
    /// transaction is synchronized before the in-memory state changes; a
    /// failure leaves the previous state intact.
    pub fn put_document(
        &mut self,
        document: Document,
        chunks: Vec<ChunkInput>,
    ) -> Result<Vec<ChunkId>> {
        if self.poisoned {
            return Err(Error::internal(
                "store is poisoned after a failed post-commit apply; reopen it",
            ));
        }
        if chunks.is_empty() {
            return Err(Error::usage(format!(
                "document {} has no chunks",
                document.id()
            )));
        }
        if chunks.len() > MAX_CHUNKS_PER_DOCUMENT {
            return Err(Error::limit_exceeded(format!(
                "document has {} chunks, above the {MAX_CHUNKS_PER_DOCUMENT}-chunk cap",
                chunks.len()
            )));
        }
        if self.state.documents.contains_key(document.id()) {
            return Err(Error::usage(format!(
                "document {} already exists",
                document.id()
            )));
        }
        if self.state.documents.len() >= MAX_DOCUMENTS {
            return Err(Error::limit_exceeded(format!(
                "store already holds {MAX_DOCUMENTS} documents"
            )));
        }
        let total_chunks = self
            .state
            .chunks
            .len()
            .checked_add(chunks.len())
            .ok_or_else(|| Error::limit_exceeded("chunk count overflow"))?;
        if total_chunks > MAX_CHUNKS {
            return Err(Error::limit_exceeded(format!(
                "store would hold {total_chunks} chunks, above the {MAX_CHUNKS}-chunk cap"
            )));
        }

        let document_id = document.id().clone();
        let mut operations = Vec::with_capacity(chunks.len() + 1);
        operations.push(Operation::PutDocument(document));
        let mut chunk_ids = Vec::with_capacity(chunks.len());
        let mut next_chunk_id = self.state.next_chunk_id;

        for (ordinal, input) in chunks.into_iter().enumerate() {
            let id = ChunkId::new(next_chunk_id)?;
            let chunk = Chunk::new(
                id,
                document_id.clone(),
                ordinal as u32,
                input.text,
                input.embedding,
            )?;
            if let Some(dimension) = self.state.embedding_dimension
                && chunk.dimension() != dimension
            {
                return Err(Error::usage(format!(
                    "chunk embedding dimension {} does not match the store dimension {dimension}",
                    chunk.dimension()
                )));
            }
            operations.push(Operation::PutChunk(chunk));
            chunk_ids.push(id);
            next_chunk_id = next_chunk_id
                .checked_add(1)
                .ok_or_else(|| Error::limit_exceeded("chunk id overflow"))?;
        }

        let transaction = Transaction::new(operations)?;
        let payload = transaction.encode()?;
        self.journal.append(self.next_sequence, &payload)?;
        if let Err(error) = self.state.apply(&transaction, ApplyMode::Live) {
            self.poisoned = true;
            return Err(Error::internal(format!(
                "transaction committed but failed to apply: {error}"
            )));
        }
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or_else(|| Error::internal("journal sequence overflow"))?;
        Ok(chunk_ids)
    }

    /// Deletes a document and all of its chunks in one atomic transaction.
    ///
    /// Returns false when the document does not exist; deletion is idempotent
    /// at the API level.
    pub fn delete_document(&mut self, id: &DocumentId) -> Result<bool> {
        if self.poisoned {
            return Err(Error::internal(
                "store is poisoned after a failed post-commit apply; reopen it",
            ));
        }
        if !self.state.documents.contains_key(id) {
            return Ok(false);
        }
        let transaction = Transaction::new(vec![Operation::DeleteDocument(id.clone())])?;
        let payload = transaction.encode()?;
        self.journal.append(self.next_sequence, &payload)?;
        if let Err(error) = self.state.apply(&transaction, ApplyMode::Live) {
            self.poisoned = true;
            return Err(Error::internal(format!(
                "transaction committed but failed to apply: {error}"
            )));
        }
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or_else(|| Error::internal("journal sequence overflow"))?;
        Ok(true)
    }

    /// Current journal append offset.
    pub fn journal_offset(&self) -> u64 {
        self.journal.offset()
    }
}

fn load_state(dir: &Path) -> Result<LoadedState> {
    let snapshot_path = dir.join(SNAPSHOT_FILE_NAME);
    let snapshot_bytes = snapshot::read_file(&snapshot_path)?;
    let mut state = StoreState::default();
    let mut base_sequence = 0u64;
    let mut snapshot_records = 0u64;

    if let Some(bytes) = snapshot_bytes {
        let snapshot = snapshot::parse(&bytes)?;
        let header = snapshot.header;
        state.initialize_for_snapshot(header.next_chunk_id, header.embedding_dimension)?;
        for record in &snapshot.records {
            let transaction = Transaction::decode(record.payload)?;
            state.apply(&transaction, ApplyMode::Snapshot)?;
        }
        state.finish_snapshot(&header)?;
        base_sequence = header.base_sequence;
        snapshot_records = header.record_count;
    }

    let journal_path = dir.join(JOURNAL_FILE_NAME);
    let journal_bytes = journal::read_file(&journal_path)?;
    let scan = scan_journal(&journal_bytes, base_sequence)?;
    let mut journal_records = 0u64;
    let mut next_sequence = base_sequence
        .checked_add(1)
        .ok_or_else(|| Error::internal("journal sequence overflow"))?;

    for record in &scan.records {
        if record.sequence <= base_sequence {
            continue;
        }
        let transaction = Transaction::decode(record.payload)?;
        state.apply(&transaction, ApplyMode::Replay)?;
        journal_records += 1;
        next_sequence = record
            .sequence
            .checked_add(1)
            .ok_or_else(|| Error::internal("journal sequence overflow"))?;
    }

    Ok(LoadedState {
        state,
        next_sequence,
        journal_valid_end: scan.valid_end,
        journal_records,
        snapshot_records,
        truncated_bytes: scan.truncated_bytes,
    })
}

fn scan_journal(bytes: &[u8], base_sequence: u64) -> Result<journal::Scan<'_>> {
    let expected_start = base_sequence
        .checked_add(1)
        .ok_or_else(|| Error::internal("journal sequence overflow"))?;
    match journal::first_sequence(bytes)? {
        None => journal::scan_with_start(bytes, expected_start),
        Some(sequence) if sequence == expected_start => {
            journal::scan_with_start(bytes, expected_start)
        }
        Some(1) if base_sequence > 0 => journal::scan_with_start(bytes, 1),
        Some(sequence) => Err(Error::corrupt(format!(
            "journal starts at sequence {sequence}, expected {expected_start} or 1 after snapshot base {base_sequence}"
        ))),
    }
}

impl StoreState {
    fn document_chunks(&self, id: &DocumentId) -> &[ChunkId] {
        self.chunks_by_document.get(id).map_or(&[], Vec::as_slice)
    }

    fn initialize_for_snapshot(
        &mut self,
        next_chunk_id: u64,
        embedding_dimension: Option<usize>,
    ) -> Result<()> {
        if !self.documents.is_empty() || !self.chunks.is_empty() {
            return Err(Error::internal(
                "snapshot state must start from an empty store",
            ));
        }
        if next_chunk_id == 0 {
            return Err(Error::corrupt("snapshot next chunk id is zero"));
        }
        if let Some(dimension) = embedding_dimension
            && (dimension == 0 || dimension > MAX_EMBEDDING_DIMENSION)
        {
            return Err(Error::corrupt(format!(
                "snapshot embedding dimension {dimension} is outside 1..={MAX_EMBEDDING_DIMENSION}"
            )));
        }
        self.next_chunk_id = next_chunk_id;
        self.embedding_dimension = embedding_dimension;
        Ok(())
    }

    fn finish_snapshot(&self, header: &SnapshotHeader) -> Result<()> {
        if self.documents.len() as u64 != header.document_count {
            return Err(Error::corrupt(format!(
                "snapshot declares {} documents, restored {}",
                header.document_count,
                self.documents.len()
            )));
        }
        if self.chunks.len() as u64 != header.chunk_count {
            return Err(Error::corrupt(format!(
                "snapshot declares {} chunks, restored {}",
                header.chunk_count,
                self.chunks.len()
            )));
        }
        if !self.chunks.is_empty() {
            let Some(dimension) = self.embedding_dimension else {
                return Err(Error::corrupt(
                    "snapshot contains chunks without an embedding dimension",
                ));
            };
            for chunk in self.chunks.values() {
                if chunk.dimension() != dimension {
                    return Err(Error::corrupt(format!(
                        "snapshot chunk {} has dimension {}, expected {dimension}",
                        chunk.id(),
                        chunk.dimension()
                    )));
                }
            }
            if let Some(chunk_id) = self.chunks.keys().next_back()
                && chunk_id.get() >= header.next_chunk_id
            {
                return Err(Error::corrupt(format!(
                    "snapshot chunk id {} is not below the next id {}",
                    chunk_id, header.next_chunk_id
                )));
            }
        }
        Ok(())
    }

    fn apply(&mut self, transaction: &Transaction, mode: ApplyMode) -> Result<()> {
        for operation in transaction.operations() {
            match operation {
                Operation::PutDocument(document) => {
                    if self.documents.contains_key(document.id()) {
                        return Err(inconsistency(
                            mode,
                            format!("document {} is inserted twice", document.id()),
                        ));
                    }
                    if self.documents.len() >= MAX_DOCUMENTS {
                        return Err(Error::limit_exceeded(format!(
                            "store holds {MAX_DOCUMENTS} documents"
                        )));
                    }
                    self.documents
                        .insert(document.id().clone(), document.clone());
                }
                Operation::PutChunk(chunk) => {
                    if mode == ApplyMode::Snapshot {
                        if chunk.id().get() >= self.next_chunk_id {
                            return Err(inconsistency(
                                mode,
                                format!(
                                    "chunk id {} is not below the snapshot next id {}",
                                    chunk.id(),
                                    self.next_chunk_id
                                ),
                            ));
                        }
                    } else if chunk.id().get() != self.next_chunk_id {
                        return Err(inconsistency(
                            mode,
                            format!(
                                "chunk id {} does not match the next id {}",
                                chunk.id(),
                                self.next_chunk_id
                            ),
                        ));
                    }
                    if self.chunks.contains_key(&chunk.id()) {
                        return Err(inconsistency(
                            mode,
                            format!("chunk {} is inserted twice", chunk.id()),
                        ));
                    }
                    if !self.documents.contains_key(chunk.document_id()) {
                        return Err(inconsistency(
                            mode,
                            format!(
                                "chunk {} references missing document {}",
                                chunk.id(),
                                chunk.document_id()
                            ),
                        ));
                    }
                    if self.chunks.len() >= MAX_CHUNKS {
                        return Err(Error::limit_exceeded(format!(
                            "store holds {MAX_CHUNKS} chunks"
                        )));
                    }
                    if let Some(ids) = self.chunks_by_document.get(chunk.document_id()) {
                        if ids.len() >= MAX_CHUNKS_PER_DOCUMENT {
                            return Err(Error::limit_exceeded(format!(
                                "document {} holds {MAX_CHUNKS_PER_DOCUMENT} chunks",
                                chunk.document_id()
                            )));
                        }
                        if ids.iter().any(|id| {
                            self.chunks
                                .get(id)
                                .is_some_and(|existing| existing.ordinal() == chunk.ordinal())
                        }) {
                            return Err(inconsistency(
                                mode,
                                format!(
                                    "document {} has duplicate chunk ordinal {}",
                                    chunk.document_id(),
                                    chunk.ordinal()
                                ),
                            ));
                        }
                    }
                    if let Some(dimension) = self.embedding_dimension {
                        if chunk.dimension() != dimension {
                            return Err(inconsistency(
                                mode,
                                format!(
                                    "chunk embedding dimension {} differs from store dimension {dimension}",
                                    chunk.dimension()
                                ),
                            ));
                        }
                    } else if mode == ApplyMode::Snapshot {
                        return Err(Error::corrupt(
                            "snapshot contains a chunk without an embedding dimension",
                        ));
                    } else {
                        self.embedding_dimension = Some(chunk.dimension());
                    }
                    self.chunks_by_document
                        .entry(chunk.document_id().clone())
                        .or_default()
                        .push(chunk.id());
                    self.chunks.insert(chunk.id(), chunk.clone());
                    if mode != ApplyMode::Snapshot {
                        self.next_chunk_id = self
                            .next_chunk_id
                            .checked_add(1)
                            .ok_or_else(|| Error::internal("chunk id overflow"))?;
                    }
                }
                Operation::DeleteDocument(id) => {
                    if mode == ApplyMode::Snapshot {
                        return Err(Error::corrupt(
                            "snapshot contains a delete-document operation",
                        ));
                    }
                    if self.documents.remove(id).is_none() {
                        return Err(inconsistency(
                            mode,
                            format!("deleted document {id} does not exist"),
                        ));
                    }
                    let ids = self.chunks_by_document.remove(id).unwrap_or_default();
                    for chunk_id in ids {
                        if self.chunks.remove(&chunk_id).is_none() {
                            return Err(inconsistency(
                                mode,
                                format!("document {id} references missing chunk {chunk_id}"),
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

fn inconsistency(mode: ApplyMode, message: String) -> Error {
    match mode {
        ApplyMode::Live => Error::internal(format!("invalid live transaction: {message}")),
        ApplyMode::Replay => Error::corrupt(format!("invalid journal transaction: {message}")),
        ApplyMode::Snapshot => Error::corrupt(format!("invalid snapshot transaction: {message}")),
    }
}

fn sync_directory(path: &Path) -> Result<()> {
    let directory = File::open(path)?;
    directory.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::journal::{self, FOOTER_LEN, HEADER_LEN, Journal};
    use crate::storage::model::{MAX_TRANSACTION_BYTES, Metadata, Modality};
    use crate::storage::snapshot::{SNAPSHOT_FILE_NAME, SNAPSHOT_TMP_FILE_NAME};
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir(name: &str) -> PathBuf {
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "seldon-store-{}-{}-{unique}",
            std::process::id(),
            name
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create temp dir");
        path
    }

    fn document(id: &str) -> Document {
        let mut metadata = Metadata::new();
        metadata.insert("lang", "en").expect("metadata");
        Document::new(DocumentId::new(id).expect("document id"), Modality::Text)
            .with_title("A title")
            .expect("title")
            .with_source("fixtures/source.txt")
            .expect("source")
            .with_metadata(metadata)
    }

    fn chunk(text: &str, embedding: Vec<f32>) -> ChunkInput {
        ChunkInput::new(text.to_owned(), embedding)
    }

    #[test]
    fn documents_and_chunks_survive_reopen() {
        let dir = temp_dir("reopen");
        let mut store = Store::open(&dir).expect("open store");
        assert_eq!(store.document_count(), 0);
        assert_eq!(store.chunk_count(), 0);
        assert_eq!(store.embedding_dimension(), None);

        let chunk_ids = store
            .put_document(
                document("doc-1"),
                vec![
                    chunk("hello world", vec![3.0, 4.0]),
                    chunk("second chunk", vec![0.0, 5.0]),
                ],
            )
            .expect("put document");
        assert_eq!(chunk_ids.len(), 2);
        assert_eq!(chunk_ids[0].get(), 1);
        assert_eq!(chunk_ids[1].get(), 2);
        assert_eq!(store.document_count(), 1);
        assert_eq!(store.chunk_count(), 2);
        assert_eq!(store.embedding_dimension(), Some(2));
        assert_eq!(store.recovery().replayed_records, 0);
        assert_eq!(store.recovery().truncated_bytes, 0);
        drop(store);

        let store = Store::open(&dir).expect("reopen store");
        assert_eq!(store.document_count(), 1);
        assert_eq!(store.chunk_count(), 2);
        assert_eq!(store.embedding_dimension(), Some(2));
        assert_eq!(store.recovery().replayed_records, 1);
        let document_id = DocumentId::new("doc-1").expect("id");
        let stored = store.document(&document_id).expect("document");
        assert_eq!(stored.title(), Some("A title"));
        assert_eq!(stored.source(), Some("fixtures/source.txt"));
        assert_eq!(stored.metadata().get("lang"), Some("en"));
        let chunk_ids = store.document_chunks(&document_id);
        assert_eq!(chunk_ids.len(), 2);
        let first = store.chunk(chunk_ids[0]).expect("chunk");
        assert_eq!(first.text(), "hello world");
        assert_eq!(first.ordinal(), 0);
        assert_eq!(first.document_id(), &document_id);
        let norm = first
            .embedding()
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn deletion_cascades_chunks_and_survives_reopen() {
        let dir = temp_dir("delete");
        let mut store = Store::open(&dir).expect("open store");
        let document_id = DocumentId::new("doc-1").expect("id");
        store
            .put_document(
                document("doc-1"),
                vec![
                    chunk("first", vec![1.0, 0.0]),
                    chunk("second", vec![0.0, 1.0]),
                ],
            )
            .expect("put document");
        assert!(store.delete_document(&document_id).expect("delete"));
        assert_eq!(store.document_count(), 0);
        assert_eq!(store.chunk_count(), 0);
        assert!(store.document_chunks(&document_id).is_empty());
        assert_eq!(store.next_chunk_id(), 3);
        drop(store);

        let store = Store::open(&dir).expect("reopen");
        assert_eq!(store.document_count(), 0);
        assert_eq!(store.chunk_count(), 0);
        assert_eq!(store.next_chunk_id(), 3);
        assert_eq!(store.recovery().replayed_records, 2);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn duplicate_documents_are_rejected_without_writing() {
        let dir = temp_dir("duplicate");
        let mut store = Store::open(&dir).expect("open store");
        store
            .put_document(document("doc-1"), vec![chunk("first", vec![1.0, 0.0])])
            .expect("first insert");
        let offset = store.journal_offset();
        let error = store
            .put_document(document("doc-1"), vec![chunk("second", vec![0.0, 1.0])])
            .expect_err("duplicate must fail");
        assert!(matches!(error, Error::Usage(_)), "{error}");
        assert_eq!(store.document_count(), 1);
        assert_eq!(store.chunk_count(), 1);
        assert_eq!(store.journal_offset(), offset);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn embedding_dimensions_must_match_across_the_store() {
        let dir = temp_dir("dimension");
        let mut store = Store::open(&dir).expect("open store");
        store
            .put_document(document("doc-1"), vec![chunk("first", vec![1.0, 0.0])])
            .expect("first insert");
        let offset = store.journal_offset();
        let error = store
            .put_document(
                document("doc-2"),
                vec![chunk("second", vec![1.0, 0.0, 0.0])],
            )
            .expect_err("dimension mismatch must fail");
        assert!(matches!(error, Error::Usage(_)), "{error}");
        assert_eq!(store.document_count(), 1);
        assert_eq!(store.journal_offset(), offset);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn audio_documents_persist_and_empty_chunk_lists_are_rejected() {
        let dir = temp_dir("audio");
        let mut store = Store::open(&dir).expect("open store");
        let audio = Document::new(DocumentId::new("audio").expect("id"), Modality::Audio);
        let chunk_ids = store
            .put_document(audio, vec![chunk("ignored", vec![1.0, 0.0])])
            .expect("audio document");
        assert_eq!(chunk_ids.len(), 1);
        assert_eq!(store.document_count(), 1);

        let error = store
            .put_document(document("empty"), Vec::new())
            .expect_err("empty document");
        assert!(matches!(error, Error::Usage(_)), "{error}");
        assert_eq!(store.document_count(), 1);
        assert!(store.journal_offset() > 0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn image_documents_persist_and_replay() {
        let dir = temp_dir("image");
        let mut store = Store::open(&dir).expect("open store");
        let image = Document::new(DocumentId::new("image-1").expect("id"), Modality::Image)
            .with_title("Quadrants")
            .expect("title")
            .with_source("fixtures/media/quadrants_16x16.png")
            .expect("source");
        store
            .put_document(image, vec![chunk("", vec![1.0, 0.0])])
            .expect("image insert");
        drop(store);

        let store = Store::open(&dir).expect("reopen store");
        let document = store
            .document(&DocumentId::new("image-1").expect("id"))
            .expect("image document");
        assert_eq!(document.modality(), Modality::Image);
        assert_eq!(document.title(), Some("Quadrants"));
        assert_eq!(
            document.source(),
            Some("fixtures/media/quadrants_16x16.png")
        );
        let chunk_id = store.document_chunks(document.id())[0];
        let chunk = store.chunk(chunk_id).expect("image chunk");
        assert_eq!(chunk.text(), "");
        assert_eq!(chunk.embedding(), &[1.0, 0.0]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn writer_lock_rejects_a_second_store() {
        let dir = temp_dir("lock");
        let first = Store::open(&dir).expect("first store");
        let error = Store::open(&dir).expect_err("second store must be locked");
        match error {
            Error::Io(error) => assert_eq!(error.kind(), io::ErrorKind::WouldBlock),
            other => panic!("expected WouldBlock, got {other:?}"),
        }
        drop(first);
        let second = Store::open(&dir).expect("lock is released");
        drop(second);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn crash_injection_at_every_byte_recovers_the_committed_prefix() {
        let source = temp_dir("crash-source");
        {
            let mut store = Store::open(&source).expect("open source");
            store
                .put_document(document("doc-1"), vec![chunk("hello", vec![1.0, 0.0])])
                .expect("put document");
        }
        let bytes = fs::read(source.join(JOURNAL_FILE_NAME)).expect("read journal");
        assert!(!bytes.is_empty());

        for cut in 0..=bytes.len() {
            let dir = temp_dir("crash-target");
            fs::write(dir.join(JOURNAL_FILE_NAME), &bytes[..cut]).expect("write prefix");
            let store = Store::open(&dir).expect("recover prefix");
            if cut == bytes.len() {
                assert_eq!(store.document_count(), 1, "cut {cut}");
                assert_eq!(store.chunk_count(), 1, "cut {cut}");
                assert_eq!(store.recovery().truncated_bytes, 0, "cut {cut}");
            } else {
                assert_eq!(store.document_count(), 0, "cut {cut}");
                assert_eq!(store.chunk_count(), 0, "cut {cut}");
                assert_eq!(store.recovery().truncated_bytes, cut as u64, "cut {cut}");
            }
            drop(store);
            let _ = fs::remove_dir_all(&dir);
        }
        let _ = fs::remove_dir_all(&source);
    }

    #[test]
    fn crash_injection_keeps_complete_transactions() {
        let source = temp_dir("two-crash-source");
        {
            let mut store = Store::open(&source).expect("open source");
            store
                .put_document(document("doc-1"), vec![chunk("first", vec![1.0, 0.0])])
                .expect("first document");
            store
                .put_document(document("doc-2"), vec![chunk("second", vec![0.0, 1.0])])
                .expect("second document");
        }
        let bytes = fs::read(source.join(JOURNAL_FILE_NAME)).expect("read journal");
        let scan = journal::scan(&bytes).expect("scan");
        let first_record_end = HEADER_LEN + scan.records[0].payload.len() + FOOTER_LEN;

        for cut in 0..=bytes.len() {
            let dir = temp_dir("two-crash-target");
            fs::write(dir.join(JOURNAL_FILE_NAME), &bytes[..cut]).expect("write prefix");
            let store = Store::open(&dir).expect("recover prefix");
            let expected_first = cut >= first_record_end;
            let expected_second = cut == bytes.len();
            assert_eq!(
                store.document_count(),
                usize::from(expected_first) + usize::from(expected_second),
                "cut {cut}"
            );
            assert_eq!(
                store.chunk_count(),
                usize::from(expected_first) + usize::from(expected_second),
                "cut {cut}"
            );
            drop(store);
            let _ = fs::remove_dir_all(&dir);
        }
        let _ = fs::remove_dir_all(&source);
    }

    #[test]
    fn corruption_is_reported_instead_of_truncated() {
        let source = temp_dir("corrupt-source");
        {
            let mut store = Store::open(&source).expect("open source");
            store
                .put_document(document("doc-1"), vec![chunk("hello", vec![1.0, 0.0])])
                .expect("put document");
        }
        let mut bytes = fs::read(source.join(JOURNAL_FILE_NAME)).expect("read journal");
        let payload_start = HEADER_LEN;
        bytes[payload_start] ^= 0xFF;
        let dir = temp_dir("corrupt-target");
        fs::write(dir.join(JOURNAL_FILE_NAME), &bytes).expect("write corrupt journal");
        let error = Store::open(&dir).expect_err("corruption must fail");
        assert!(matches!(error, Error::Corrupt(_)), "{error}");
        let _ = fs::remove_dir_all(&source);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn invalid_transaction_payloads_are_rejected_on_open() {
        let dir = temp_dir("invalid-transaction");
        {
            let mut journal =
                Journal::open(&dir.join(JOURNAL_FILE_NAME), 0, false).expect("open journal");
            journal.append(1, &[1, 0, 1, 0, 0, 0, 9]).expect("append");
        }
        let error = Store::open(&dir).expect_err("invalid transaction must fail");
        assert!(matches!(error, Error::Corrupt(_)), "{error}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn journal_sequence_continues_after_reopen() {
        let dir = temp_dir("sequence");
        {
            let mut store = Store::open(&dir).expect("open");
            store
                .put_document(document("doc-1"), vec![chunk("first", vec![1.0, 0.0])])
                .expect("first");
        }
        {
            let mut store = Store::open(&dir).expect("reopen");
            store
                .put_document(document("doc-2"), vec![chunk("second", vec![0.0, 1.0])])
                .expect("second");
        }
        let bytes = fs::read(dir.join(JOURNAL_FILE_NAME)).expect("read");
        let scan = journal::scan(&bytes).expect("scan");
        assert_eq!(scan.records.len(), 2);
        assert_eq!(scan.records[0].sequence, 1);
        assert_eq!(scan.records[1].sequence, 2);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn oversized_document_payload_is_rejected_before_writing() {
        let dir = temp_dir("oversized-document");
        let mut store = Store::open(&dir).expect("open");
        let text = "x".repeat(MAX_TRANSACTION_BYTES);
        let error = store
            .put_document(document("doc-1"), vec![chunk(&text, vec![1.0, 0.0])])
            .expect_err("payload cap must be enforced");
        assert!(matches!(error, Error::LimitExceeded(_)), "{error}");
        assert_eq!(store.document_count(), 0);
        assert_eq!(store.journal_offset(), 0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn delete_missing_document_is_a_no_op() {
        let dir = temp_dir("delete-missing");
        let mut store = Store::open(&dir).expect("open");
        let id = DocumentId::new("missing").expect("id");
        assert!(!store.delete_document(&id).expect("delete"));
        assert_eq!(store.journal_offset(), 0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_reports_the_committed_prefix_without_modifying_the_journal() {
        let dir = temp_dir("verify");
        {
            let mut store = Store::open(&dir).expect("open");
            store
                .put_document(document("doc-1"), vec![chunk("first", vec![1.0, 0.0])])
                .expect("first");
            store
                .put_document(document("doc-2"), vec![chunk("second", vec![0.0, 1.0])])
                .expect("second");
        }
        let path = dir.join(JOURNAL_FILE_NAME);
        let bytes = fs::read(&path).expect("read journal");
        let scan = journal::scan(&bytes).expect("scan");
        let second_start = HEADER_LEN + scan.records[0].payload.len() + FOOTER_LEN;
        let torn = &bytes[..second_start + 7];
        fs::write(&path, torn).expect("write torn journal");

        let verification = Store::verify(&dir).expect("verify");
        assert_eq!(verification.records, 1);
        assert_eq!(verification.documents, 1);
        assert_eq!(verification.chunks, 1);
        assert_eq!(verification.truncated_bytes, 7);
        let after = fs::read(&path).expect("journal remains");
        assert_eq!(after, torn);

        let error = Store::verify(dir.join("missing")).expect_err("missing directory");
        assert!(matches!(error, Error::Io(_)), "{error}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn compaction_preserves_ids_next_chunk_id_and_sequences() {
        let dir = temp_dir("compact-ids");
        let document_one = DocumentId::new("doc-1").expect("id");
        {
            let mut store = Store::open(&dir).expect("open");
            store
                .put_document(
                    document("doc-1"),
                    vec![
                        chunk("first", vec![1.0, 0.0]),
                        chunk("second", vec![0.0, 1.0]),
                    ],
                )
                .expect("doc 1");
            store
                .put_document(document("doc-2"), vec![chunk("third", vec![1.0, 1.0])])
                .expect("doc 2");
            store
                .delete_document(&DocumentId::new("doc-2").expect("id"))
                .expect("delete doc 2");
            assert_eq!(store.next_chunk_id(), 4);

            let compaction = store.compact().expect("compact");
            assert_eq!(compaction.snapshot_records, 1);
            assert_eq!(compaction.documents, 1);
            assert_eq!(compaction.chunks, 2);
            assert!(compaction.snapshot_bytes > 0);
            assert_eq!(store.recovery().snapshot_records, 1);
            assert_eq!(store.recovery().replayed_records, 0);
            assert_eq!(store.journal_offset(), 0);
        }

        {
            let store = Store::open(&dir).expect("reopen");
            assert_eq!(store.document_count(), 1);
            assert_eq!(store.chunk_count(), 2);
            assert_eq!(store.next_chunk_id(), 4);
            assert_eq!(store.embedding_dimension(), Some(2));
            assert_eq!(store.recovery().snapshot_records, 1);
            assert_eq!(store.recovery().replayed_records, 0);
            assert_eq!(
                store.document_chunks(&document_one),
                &[ChunkId::new(1).expect("id"), ChunkId::new(2).expect("id")]
            );
        }

        {
            let mut store = Store::open(&dir).expect("reopen for append");
            let ids = store
                .put_document(document("doc-3"), vec![chunk("fourth", vec![0.0, 2.0])])
                .expect("doc 3");
            assert_eq!(ids[0].get(), 4);
        }

        let store = Store::open(&dir).expect("reopen with journal");
        assert_eq!(store.document_count(), 2);
        assert_eq!(store.chunk_count(), 3);
        assert_eq!(store.next_chunk_id(), 5);
        assert_eq!(store.recovery().snapshot_records, 1);
        assert_eq!(store.recovery().replayed_records, 1);
        let bytes = fs::read(dir.join(JOURNAL_FILE_NAME)).expect("read journal");
        let scan = journal::scan_with_start(&bytes, 4).expect("scan journal");
        assert_eq!(scan.records.len(), 1);
        assert_eq!(scan.records[0].sequence, 4);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn compaction_of_an_empty_store_round_trips() {
        let dir = temp_dir("compact-empty");
        {
            let mut store = Store::open(&dir).expect("open");
            let compaction = store.compact().expect("compact empty store");
            assert_eq!(compaction.snapshot_records, 0);
            assert_eq!(compaction.documents, 0);
            assert_eq!(compaction.chunks, 0);
        }
        {
            let store = Store::open(&dir).expect("reopen empty snapshot");
            assert_eq!(store.document_count(), 0);
            assert_eq!(store.chunk_count(), 0);
            assert_eq!(store.next_chunk_id(), 1);
            assert_eq!(store.embedding_dimension(), None);
            assert_eq!(store.recovery().snapshot_records, 0);
        }
        {
            let mut store = Store::open(&dir).expect("open for append");
            let ids = store
                .put_document(document("doc-1"), vec![chunk("first", vec![1.0, 0.0])])
                .expect("append after empty snapshot");
            assert_eq!(ids[0].get(), 1);
        }
        let store = Store::open(&dir).expect("reopen after append");
        assert_eq!(store.document_count(), 1);
        assert_eq!(store.recovery().snapshot_records, 0);
        assert_eq!(store.recovery().replayed_records, 1);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn compaction_preserves_the_dimension_after_deleting_every_chunk() {
        let dir = temp_dir("compact-dimension");
        {
            let mut store = Store::open(&dir).expect("open");
            store
                .put_document(document("doc-1"), vec![chunk("first", vec![1.0, 0.0])])
                .expect("doc");
            store
                .delete_document(&DocumentId::new("doc-1").expect("id"))
                .expect("delete");
            assert_eq!(store.embedding_dimension(), Some(2));
            assert_eq!(store.next_chunk_id(), 2);
            store.compact().expect("compact");
        }
        {
            let store = Store::open(&dir).expect("reopen");
            assert_eq!(store.document_count(), 0);
            assert_eq!(store.chunk_count(), 0);
            assert_eq!(store.embedding_dimension(), Some(2));
            assert_eq!(store.next_chunk_id(), 2);
        }
        {
            let mut store = Store::open(&dir).expect("open");
            let ids = store
                .put_document(document("doc-2"), vec![chunk("second", vec![0.0, 1.0])])
                .expect("same dimension");
            assert_eq!(ids[0].get(), 2);
            let error = store
                .put_document(document("doc-3"), vec![chunk("third", vec![1.0, 0.0, 0.0])])
                .expect_err("dimension must still be enforced");
            assert!(matches!(error, Error::Usage(_)), "{error}");
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_truncated_snapshot_temp_file_is_ignored_at_every_byte() {
        let source = temp_dir("snapshot-temp-source");
        let journal_bytes;
        let snapshot_bytes;
        {
            let mut store = Store::open(&source).expect("open source");
            store
                .put_document(document("doc-1"), vec![chunk("first", vec![1.0, 0.0])])
                .expect("put");
            journal_bytes = fs::read(source.join(JOURNAL_FILE_NAME)).expect("read journal");
            store.compact().expect("compact source");
            snapshot_bytes = fs::read(source.join(SNAPSHOT_FILE_NAME)).expect("read snapshot");
        }

        for cut in 0..=snapshot_bytes.len() {
            let dir = temp_dir("snapshot-temp-target");
            fs::write(dir.join(JOURNAL_FILE_NAME), &journal_bytes).expect("write journal");
            fs::write(dir.join(SNAPSHOT_TMP_FILE_NAME), &snapshot_bytes[..cut])
                .expect("write snapshot temp");
            let store = Store::open(&dir).expect("temp snapshot is ignored");
            assert_eq!(store.document_count(), 1, "cut {cut}");
            assert_eq!(store.chunk_count(), 1, "cut {cut}");
            assert_eq!(store.recovery().snapshot_records, 0, "cut {cut}");
            assert_eq!(store.recovery().replayed_records, 1, "cut {cut}");
            drop(store);
            let _ = fs::remove_dir_all(&dir);
        }
        let _ = fs::remove_dir_all(&source);
    }

    #[test]
    fn a_truncated_journal_temp_file_is_ignored_at_every_byte() {
        let source = temp_dir("journal-temp-source");
        let snapshot_bytes;
        {
            let mut store = Store::open(&source).expect("open source");
            store
                .put_document(document("doc-1"), vec![chunk("first", vec![1.0, 0.0])])
                .expect("put");
            store.compact().expect("compact source");
            snapshot_bytes = fs::read(source.join(SNAPSHOT_FILE_NAME)).expect("read snapshot");
        }
        let record = journal::encode_record(1, b"ignored");
        for cut in 0..=record.len() {
            let dir = temp_dir("journal-temp-target");
            fs::write(dir.join(SNAPSHOT_FILE_NAME), &snapshot_bytes).expect("write snapshot");
            fs::write(dir.join(JOURNAL_TMP_FILE_NAME), &record[..cut]).expect("write journal temp");
            let store = Store::open(&dir).expect("temp journal is ignored");
            assert_eq!(store.document_count(), 1, "cut {cut}");
            assert_eq!(store.recovery().snapshot_records, 1, "cut {cut}");
            assert_eq!(store.recovery().replayed_records, 0, "cut {cut}");
            drop(store);
            let _ = fs::remove_dir_all(&dir);
        }
        let _ = fs::remove_dir_all(&source);
    }

    #[test]
    fn a_truncated_snapshot_file_is_rejected() {
        let source = temp_dir("snapshot-truncated-source");
        let snapshot_bytes;
        {
            let mut store = Store::open(&source).expect("open source");
            store
                .put_document(document("doc-1"), vec![chunk("first", vec![1.0, 0.0])])
                .expect("put");
            store.compact().expect("compact source");
            snapshot_bytes = fs::read(source.join(SNAPSHOT_FILE_NAME)).expect("read snapshot");
        }

        for cut in 0..snapshot_bytes.len() {
            let dir = temp_dir("snapshot-truncated-target");
            fs::write(dir.join(SNAPSHOT_FILE_NAME), &snapshot_bytes[..cut])
                .expect("write truncated snapshot");
            let error = Store::open(&dir).expect_err("truncated snapshot must fail");
            assert!(matches!(error, Error::Corrupt(_)), "cut {cut}: {error}");
            let _ = fs::remove_dir_all(&dir);
        }

        let dir = temp_dir("snapshot-full-target");
        fs::write(dir.join(SNAPSHOT_FILE_NAME), &snapshot_bytes).expect("write snapshot");
        let store = Store::open(&dir).expect("complete snapshot opens");
        assert_eq!(store.document_count(), 1);
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&source);
    }

    #[test]
    fn a_corrupted_snapshot_header_is_rejected() {
        let dir = temp_dir("snapshot-corrupt");
        {
            let mut store = Store::open(&dir).expect("open");
            store
                .put_document(document("doc-1"), vec![chunk("first", vec![1.0, 0.0])])
                .expect("put");
            store.compact().expect("compact");
        }
        let path = dir.join(SNAPSHOT_FILE_NAME);
        let mut bytes = fs::read(&path).expect("read snapshot");
        bytes[8] ^= 0xFF;
        fs::write(&path, bytes).expect("write corrupt snapshot");
        let error = Store::open(&dir).expect_err("corrupt snapshot must fail");
        assert!(matches!(error, Error::Corrupt(_)), "{error}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_old_journal_after_snapshot_is_skipped_and_new_records_apply() {
        let dir = temp_dir("snapshot-old-journal");
        let old_journal;
        {
            let mut store = Store::open(&dir).expect("open");
            store
                .put_document(document("doc-1"), vec![chunk("first", vec![1.0, 0.0])])
                .expect("doc 1");
            store
                .put_document(document("doc-2"), vec![chunk("second", vec![0.0, 1.0])])
                .expect("doc 2");
            old_journal = fs::read(dir.join(JOURNAL_FILE_NAME)).expect("read journal");
            store.compact().expect("compact");
        }

        fs::write(dir.join(JOURNAL_FILE_NAME), &old_journal).expect("restore old journal");
        {
            let mut store = Store::open(&dir).expect("open with snapshot and old journal");
            assert_eq!(store.document_count(), 2);
            assert_eq!(store.recovery().snapshot_records, 2);
            assert_eq!(store.recovery().replayed_records, 0);
            let ids = store
                .put_document(document("doc-3"), vec![chunk("third", vec![1.0, 1.0])])
                .expect("doc 3");
            assert_eq!(ids[0].get(), 3);
        }

        let store = Store::open(&dir).expect("reopen");
        assert_eq!(store.document_count(), 3);
        assert_eq!(store.chunk_count(), 3);
        assert_eq!(store.next_chunk_id(), 4);
        assert_eq!(store.recovery().snapshot_records, 2);
        assert_eq!(store.recovery().replayed_records, 1);
        let bytes = fs::read(dir.join(JOURNAL_FILE_NAME)).expect("read journal");
        let scan = journal::scan(&bytes).expect("scan journal");
        assert_eq!(scan.records.len(), 3);
        assert_eq!(scan.records[2].sequence, 3);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_only_views_see_writer_state_and_ignore_torn_tails() {
        let dir = temp_dir("read-only");
        let mut writer = Store::open(&dir).expect("writer");
        writer
            .put_document(document("doc-1"), vec![chunk("first", vec![1.0, 0.0])])
            .expect("doc 1");

        {
            let reader = ReadOnlyStore::open(&dir).expect("read-only while writer open");
            assert_eq!(reader.document_count(), 1);
            assert_eq!(reader.chunk_count(), 1);
            assert_eq!(reader.recovery().replayed_records, 1);
            assert_eq!(reader.recovery().truncated_bytes, 0);
        }

        writer
            .put_document(document("doc-2"), vec![chunk("second", vec![0.0, 1.0])])
            .expect("doc 2");
        {
            let reader = ReadOnlyStore::open(&dir).expect("read-only after second write");
            assert_eq!(reader.document_count(), 2);
            assert_eq!(reader.chunk_count(), 2);
            assert_eq!(reader.recovery().replayed_records, 2);
        }
        drop(writer);

        let path = dir.join(JOURNAL_FILE_NAME);
        let committed = fs::read(&path).expect("read journal");
        let mut torn = committed.clone();
        torn.extend_from_slice(b"torn!!!");
        fs::write(&path, &torn).expect("append torn bytes");

        {
            let reader = ReadOnlyStore::open(&dir).expect("read-only with torn tail");
            assert_eq!(reader.document_count(), 2);
            assert_eq!(reader.recovery().truncated_bytes, 7);
            let unchanged = fs::read(&path).expect("read journal");
            assert_eq!(unchanged, torn);
        }

        {
            let store = Store::open(&dir).expect("writer recovers torn tail");
            assert_eq!(store.document_count(), 2);
            assert_eq!(store.recovery().truncated_bytes, 7);
        }
        let recovered = fs::read(&path).expect("read recovered journal");
        assert_eq!(recovered, committed);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_reports_snapshot_and_journal_records() {
        let dir = temp_dir("verify-snapshot");
        {
            let mut store = Store::open(&dir).expect("open");
            store
                .put_document(document("doc-1"), vec![chunk("first", vec![1.0, 0.0])])
                .expect("doc 1");
            store.compact().expect("compact");
            store
                .put_document(document("doc-2"), vec![chunk("second", vec![0.0, 1.0])])
                .expect("doc 2");
        }
        let verification = Store::verify(&dir).expect("verify");
        assert_eq!(verification.snapshot_records, 1);
        assert_eq!(verification.records, 1);
        assert_eq!(verification.documents, 2);
        assert_eq!(verification.chunks, 2);
        assert_eq!(verification.truncated_bytes, 0);
        let _ = fs::remove_dir_all(&dir);
    }
}
