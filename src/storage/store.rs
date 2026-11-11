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
    Chunk, ChunkId, ChunkInput, Document, DocumentId, MAX_CHUNKS_PER_DOCUMENT, Modality, Operation,
    Transaction,
};

/// Writer lock file name inside a data directory.
pub const LOCK_FILE_NAME: &str = "writer.lock";
/// Maximum number of documents held in one store.
pub const MAX_DOCUMENTS: usize = 1_000_000;
/// Maximum number of chunks held in one store.
pub const MAX_CHUNKS: usize = 10_000_000;

/// Recovery information produced by opening a store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recovery {
    /// Number of committed transactions replayed.
    pub replayed_records: u64,
    /// Number of trailing bytes discarded as an uncommitted tail.
    pub truncated_bytes: u64,
}

/// Read-only verification report for a data directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verification {
    /// Number of committed transactions found in the journal.
    pub records: u64,
    /// Number of documents reconstructed from the committed prefix.
    pub documents: usize,
    /// Number of chunks reconstructed from the committed prefix.
    pub chunks: usize,
    /// Number of trailing bytes that are not part of a committed record.
    pub truncated_bytes: u64,
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

        let journal_path = dir.join(JOURNAL_FILE_NAME);
        let bytes = journal::read_file(&journal_path)?;
        let scan = journal::scan(&bytes)?;
        let mut state = StoreState::default();
        let mut next_sequence = 1u64;

        for record in &scan.records {
            let transaction = Transaction::decode(record.payload)?;
            state.apply(&transaction, true)?;
            next_sequence = record
                .sequence
                .checked_add(1)
                .ok_or_else(|| Error::internal("journal sequence overflow"))?;
        }

        let journal = Journal::open(&journal_path, scan.valid_end, scan.truncated_bytes > 0)?;
        let recovery = Recovery {
            replayed_records: scan.records.len() as u64,
            truncated_bytes: scan.truncated_bytes,
        };

        Ok(Self {
            dir,
            _lock: lock,
            journal,
            state,
            next_sequence,
            recovery,
            poisoned: false,
        })
    }

    /// Verifies a data directory without acquiring the writer lock or modifying
    /// the journal.
    ///
    /// The committed prefix is decoded and applied to a temporary state to
    /// validate referential integrity and checksums. An uncommitted tail is
    /// reported in truncated_bytes but is not removed.
    pub fn verify(dir: impl AsRef<Path>) -> Result<Verification> {
        let dir = dir.as_ref();
        let metadata = fs::metadata(dir)?;
        if !metadata.is_dir() {
            return Err(Error::corrupt(format!(
                "{} is not a directory",
                dir.display()
            )));
        }
        let bytes = journal::read_file(&dir.join(JOURNAL_FILE_NAME))?;
        let scan = journal::scan(&bytes)?;
        let mut state = StoreState::default();
        for record in &scan.records {
            let transaction = Transaction::decode(record.payload)?;
            state.apply(&transaction, true)?;
        }
        Ok(Verification {
            records: scan.records.len() as u64,
            documents: state.documents.len(),
            chunks: state.chunks.len(),
            truncated_bytes: scan.truncated_bytes,
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
        if document.modality() != Modality::Text {
            return Err(Error::unsupported(format!(
                "modality {} is not implemented in milestone M3",
                document.modality().as_str()
            )));
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
        if let Err(error) = self.state.apply(&transaction, false) {
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
        if let Err(error) = self.state.apply(&transaction, false) {
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

impl StoreState {
    fn apply(&mut self, transaction: &Transaction, replay: bool) -> Result<()> {
        for operation in transaction.operations() {
            match operation {
                Operation::PutDocument(document) => {
                    if document.modality() != Modality::Text {
                        return Err(Error::unsupported(format!(
                            "modality {} is not implemented in milestone M3",
                            document.modality().as_str()
                        )));
                    }
                    if self.documents.contains_key(document.id()) {
                        return Err(inconsistency(
                            replay,
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
                    if chunk.id().get() != self.next_chunk_id {
                        return Err(inconsistency(
                            replay,
                            format!(
                                "chunk id {} does not match the next id {}",
                                chunk.id(),
                                self.next_chunk_id
                            ),
                        ));
                    }
                    if self.chunks.contains_key(&chunk.id()) {
                        return Err(inconsistency(
                            replay,
                            format!("chunk {} is inserted twice", chunk.id()),
                        ));
                    }
                    if !self.documents.contains_key(chunk.document_id()) {
                        return Err(inconsistency(
                            replay,
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
                                replay,
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
                                replay,
                                format!(
                                    "chunk embedding dimension {} differs from store dimension {dimension}",
                                    chunk.dimension()
                                ),
                            ));
                        }
                    } else {
                        self.embedding_dimension = Some(chunk.dimension());
                    }
                    self.chunks_by_document
                        .entry(chunk.document_id().clone())
                        .or_default()
                        .push(chunk.id());
                    self.chunks.insert(chunk.id(), chunk.clone());
                    self.next_chunk_id = self
                        .next_chunk_id
                        .checked_add(1)
                        .ok_or_else(|| Error::internal("chunk id overflow"))?;
                }
                Operation::DeleteDocument(id) => {
                    if self.documents.remove(id).is_none() {
                        return Err(inconsistency(
                            replay,
                            format!("deleted document {id} does not exist"),
                        ));
                    }
                    let ids = self.chunks_by_document.remove(id).unwrap_or_default();
                    for chunk_id in ids {
                        if self.chunks.remove(&chunk_id).is_none() {
                            return Err(inconsistency(
                                replay,
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

fn inconsistency(replay: bool, message: String) -> Error {
    if replay {
        Error::corrupt(format!("invalid journal transaction: {message}"))
    } else {
        Error::internal(format!("invalid live transaction: {message}"))
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
    use crate::storage::model::{MAX_TRANSACTION_BYTES, Metadata};
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
    fn non_text_modalities_and_empty_chunk_lists_are_rejected() {
        let dir = temp_dir("rejected");
        let mut store = Store::open(&dir).expect("open store");
        let image = Document::new(DocumentId::new("image").expect("id"), Modality::Image);
        let error = store
            .put_document(image, vec![chunk("ignored", vec![1.0, 0.0])])
            .expect_err("image is not implemented");
        assert!(matches!(error, Error::Unsupported(_)), "{error}");

        let error = store
            .put_document(document("empty"), Vec::new())
            .expect_err("empty document");
        assert!(matches!(error, Error::Usage(_)), "{error}");
        assert_eq!(store.document_count(), 0);
        assert_eq!(store.journal_offset(), 0);
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
}
