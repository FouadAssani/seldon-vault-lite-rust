//! Document, chunk, and transaction model for persistent storage.
//!
//! The model is deliberately small for milestone M3: documents, chunks, their
//! modality, metadata, and normalized embeddings. Identifiers are newtypes so
//! a document id cannot be confused with a chunk id or a free-form string.
//! Every constructor validates lengths and numeric values; decoders map the
//! same failures to corruption errors because the bytes came from disk.

use std::collections::BTreeMap;
use std::fmt;

use crate::error::{Error, Result};
use crate::storage::codec::{Decoder, Encoder};

/// Storage schema version written into every transaction.
pub const SCHEMA_VERSION: u16 = 1;

/// Maximum size of a document or chunk identifier in bytes.
pub const MAX_ID_BYTES: usize = 256;
/// Maximum size of a document title in bytes.
pub const MAX_TITLE_BYTES: usize = 4096;
/// Maximum size of a document source in bytes.
pub const MAX_SOURCE_BYTES: usize = 4096;
/// Maximum number of metadata entries per document.
pub const MAX_METADATA_ENTRIES: usize = 64;
/// Maximum size of a metadata key in bytes.
pub const MAX_METADATA_KEY_BYTES: usize = 128;
/// Maximum size of a metadata value in bytes.
pub const MAX_METADATA_VALUE_BYTES: usize = 4096;
/// Maximum size of a chunk text in bytes.
pub const MAX_TEXT_BYTES: usize = 1024 * 1024;
/// Maximum number of chunks in one document.
pub const MAX_CHUNKS_PER_DOCUMENT: usize = 4096;
/// Maximum embedding dimension accepted by storage.
pub const MAX_EMBEDDING_DIMENSION: usize = 8192;
/// Maximum number of operations in one atomic transaction.
pub const MAX_TRANSACTION_OPERATIONS: usize = 8192;
/// Maximum encoded size of one transaction payload.
pub const MAX_TRANSACTION_BYTES: usize = 16 * 1024 * 1024;

/// Relative tolerance accepted when replaying a stored normalized embedding.
///
/// Live ingestion normalizes with f64 accumulation then stores f32 values, so
/// the recomputed f32 norm can differ from 1.0 by a few ULPs. A hostile
/// journal cannot smuggle a non-normalized vector into cosine search.
const NORMALIZED_TOLERANCE: f64 = 1e-3;

/// Stable document identifier supplied by the user.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DocumentId(String);

impl DocumentId {
    /// Validates and creates a document identifier.
    ///
    /// Identifiers are non-empty UTF-8 strings of at most MAX_ID_BYTES.
    /// Control characters are rejected so an identifier cannot corrupt logs
    /// or a future HTTP header.
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        if value.is_empty() {
            return Err(Error::usage("document id must not be empty"));
        }
        if value.len() > MAX_ID_BYTES {
            return Err(Error::limit_exceeded(format!(
                "document id is {} bytes, above the {MAX_ID_BYTES}-byte cap",
                value.len()
            )));
        }
        if value.chars().any(char::is_control) {
            return Err(Error::usage(
                "document id must not contain control characters",
            ));
        }
        Ok(Self(value))
    }

    /// Borrows the identifier as a string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for DocumentId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Stable chunk identifier assigned by the store.
///
/// Chunk ids are positive and monotonically assigned; zero is reserved as an
/// invalid value so a zero-filled torn record cannot masquerade as a chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChunkId(u64);

impl ChunkId {
    /// Creates a positive chunk identifier.
    pub fn new(value: u64) -> Result<Self> {
        if value == 0 {
            return Err(Error::usage("chunk id must be positive"));
        }
        Ok(Self(value))
    }

    /// Returns the raw identifier.
    pub fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for ChunkId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// Content modality stored with a document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modality {
    /// Text or code content.
    Text,
    /// Image content, reserved for milestone M4.
    Image,
    /// Audio content, reserved for milestone M5.
    Audio,
}

impl Modality {
    /// Short name used by the CLI and filters.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Image => "image",
            Self::Audio => "audio",
        }
    }

    /// Parses a modality name.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "text" => Some(Self::Text),
            "image" => Some(Self::Image),
            "audio" => Some(Self::Audio),
            _ => None,
        }
    }

    /// Encoding tag written to the journal.
    fn code(self) -> u8 {
        match self {
            Self::Text => 0,
            Self::Image => 1,
            Self::Audio => 2,
        }
    }

    /// Decodes a modality tag.
    fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Text),
            1 => Some(Self::Image),
            2 => Some(Self::Audio),
            _ => None,
        }
    }
}

/// Deterministic document metadata with explicit size caps.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Metadata(BTreeMap<String, String>);

impl Metadata {
    /// Creates an empty metadata map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts or replaces one metadata entry after validation.
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<String>) -> Result<()> {
        let key = key.into();
        let value = value.into();
        if key.is_empty() {
            return Err(Error::usage("metadata key must not be empty"));
        }
        if key.len() > MAX_METADATA_KEY_BYTES {
            return Err(Error::limit_exceeded(format!(
                "metadata key is {} bytes, above the {MAX_METADATA_KEY_BYTES}-byte cap",
                key.len()
            )));
        }
        if key.chars().any(char::is_control) {
            return Err(Error::usage(
                "metadata key must not contain control characters",
            ));
        }
        if value.len() > MAX_METADATA_VALUE_BYTES {
            return Err(Error::limit_exceeded(format!(
                "metadata value is {} bytes, above the {MAX_METADATA_VALUE_BYTES}-byte cap",
                value.len()
            )));
        }
        if value.chars().any(char::is_control) {
            return Err(Error::usage(
                "metadata value must not contain control characters",
            ));
        }
        if !self.0.contains_key(&key) && self.0.len() >= MAX_METADATA_ENTRIES {
            return Err(Error::limit_exceeded(format!(
                "metadata has {MAX_METADATA_ENTRIES} entries, the documented cap"
            )));
        }
        self.0.insert(key, value);
        Ok(())
    }

    /// Looks up one metadata value.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).map(String::as_str)
    }

    /// Iterates entries in deterministic key order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.0
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_str()))
    }

    /// Number of metadata entries.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns true when there is no metadata.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// A document stored in the vault.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Document {
    id: DocumentId,
    title: Option<String>,
    source: Option<String>,
    modality: Modality,
    metadata: Metadata,
}

impl Document {
    /// Creates a document with no title, source, or metadata.
    pub fn new(id: DocumentId, modality: Modality) -> Self {
        Self {
            id,
            title: None,
            source: None,
            modality,
            metadata: Metadata::new(),
        }
    }

    /// Adds a validated title.
    pub fn with_title(mut self, title: impl Into<String>) -> Result<Self> {
        let title = title.into();
        validate_text_field(&title, MAX_TITLE_BYTES, "document title")?;
        self.title = Some(title);
        Ok(self)
    }

    /// Adds a validated source reference.
    pub fn with_source(mut self, source: impl Into<String>) -> Result<Self> {
        let source = source.into();
        validate_text_field(&source, MAX_SOURCE_BYTES, "document source")?;
        self.source = Some(source);
        Ok(self)
    }

    /// Attaches already validated metadata.
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Document identifier.
    pub fn id(&self) -> &DocumentId {
        &self.id
    }

    /// Optional title.
    pub fn title(&self) -> Option<&str> {
        self.title.as_deref()
    }

    /// Optional source reference.
    pub fn source(&self) -> Option<&str> {
        self.source.as_deref()
    }

    /// Content modality.
    pub fn modality(&self) -> Modality {
        self.modality
    }

    /// Document metadata.
    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }
}

/// A searchable chunk with its normalized embedding.
#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    id: ChunkId,
    document_id: DocumentId,
    ordinal: u32,
    text: String,
    embedding: Vec<f32>,
}

impl Chunk {
    /// Creates a chunk and normalizes its embedding to unit L2 norm.
    ///
    /// Embeddings must be finite, non-empty, and have a positive norm. The
    /// normalization guarantees that exact vector search computes a cosine
    /// similarity, not an arbitrary dot product.
    pub fn new(
        id: ChunkId,
        document_id: DocumentId,
        ordinal: u32,
        text: String,
        embedding: Vec<f32>,
    ) -> Result<Self> {
        if text.len() > MAX_TEXT_BYTES {
            return Err(Error::limit_exceeded(format!(
                "chunk text is {} bytes, above the {MAX_TEXT_BYTES}-byte cap",
                text.len()
            )));
        }
        if ordinal as usize >= MAX_CHUNKS_PER_DOCUMENT {
            return Err(Error::limit_exceeded(format!(
                "chunk ordinal {ordinal} reaches the {MAX_CHUNKS_PER_DOCUMENT}-chunk cap"
            )));
        }
        let mut embedding = embedding;
        normalize_embedding(&mut embedding)?;
        Ok(Self {
            id,
            document_id,
            ordinal,
            text,
            embedding,
        })
    }

    /// Rebuilds a chunk read from a journal or snapshot.
    ///
    /// Unlike Chunk::new, this constructor does not renormalize: replay must
    /// reproduce the exact stored bytes. It verifies that the stored vector
    /// is finite and already normalized within a small tolerance.
    pub fn from_stored(
        id: ChunkId,
        document_id: DocumentId,
        ordinal: u32,
        text: String,
        embedding: Vec<f32>,
    ) -> Result<Self> {
        if text.len() > MAX_TEXT_BYTES {
            return Err(Error::corrupt(format!(
                "stored chunk text is {} bytes, above the {MAX_TEXT_BYTES}-byte cap",
                text.len()
            )));
        }
        if ordinal as usize >= MAX_CHUNKS_PER_DOCUMENT {
            return Err(Error::corrupt(format!(
                "stored chunk ordinal {ordinal} reaches the {MAX_CHUNKS_PER_DOCUMENT}-chunk cap"
            )));
        }
        validate_stored_embedding(&embedding)?;
        Ok(Self {
            id,
            document_id,
            ordinal,
            text,
            embedding,
        })
    }

    /// Chunk identifier.
    pub fn id(&self) -> ChunkId {
        self.id
    }

    /// Owning document identifier.
    pub fn document_id(&self) -> &DocumentId {
        &self.document_id
    }

    /// Zero-based position inside the document.
    pub fn ordinal(&self) -> u32 {
        self.ordinal
    }

    /// Chunk text.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Normalized embedding.
    pub fn embedding(&self) -> &[f32] {
        &self.embedding
    }

    /// Embedding dimension.
    pub fn dimension(&self) -> usize {
        self.embedding.len()
    }
}

/// A chunk supplied by an ingestion caller before the store assigns its id.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkInput {
    /// Text stored with the chunk.
    pub text: String,
    /// Raw embedding; the store normalizes it.
    pub embedding: Vec<f32>,
}

impl ChunkInput {
    /// Creates a chunk input.
    pub fn new(text: impl Into<String>, embedding: Vec<f32>) -> Self {
        Self {
            text: text.into(),
            embedding,
        }
    }
}

/// One atomic storage operation.
#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    /// Inserts a new document.
    PutDocument(Document),
    /// Inserts a new chunk belonging to an existing document.
    PutChunk(Chunk),
    /// Deletes a document and all of its chunks.
    DeleteDocument(DocumentId),
}

/// One atomic journal transaction.
#[derive(Debug, Clone, PartialEq)]
pub struct Transaction {
    schema_version: u16,
    operations: Vec<Operation>,
}

impl Transaction {
    /// Builds a transaction with a bounded, non-empty operation list.
    pub fn new(operations: Vec<Operation>) -> Result<Self> {
        if operations.is_empty() {
            return Err(Error::usage(
                "transaction must contain at least one operation",
            ));
        }
        if operations.len() > MAX_TRANSACTION_OPERATIONS {
            return Err(Error::limit_exceeded(format!(
                "transaction has {} operations, above the {MAX_TRANSACTION_OPERATIONS}-operation cap",
                operations.len()
            )));
        }
        Ok(Self {
            schema_version: SCHEMA_VERSION,
            operations,
        })
    }

    /// Schema version recorded in the payload.
    pub fn schema_version(&self) -> u16 {
        self.schema_version
    }

    /// Operations in application order.
    pub fn operations(&self) -> &[Operation] {
        &self.operations
    }

    /// Encodes the transaction as a bounded journal payload.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut encoder = Encoder::new(MAX_TRANSACTION_BYTES);
        encoder.write_u16(self.schema_version)?;
        let count = u32::try_from(self.operations.len())
            .map_err(|_| Error::internal("transaction operation count does not fit u32"))?;
        encoder.write_u32(count)?;
        for operation in &self.operations {
            encode_operation(&mut encoder, operation)?;
        }
        Ok(encoder.finish())
    }

    /// Decodes a journal payload with explicit caps and corruption mapping.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut decoder = Decoder::new(bytes);
        let schema_version = decoder.read_u16()?;
        if schema_version != SCHEMA_VERSION {
            return Err(Error::unsupported(format!(
                "storage schema version {schema_version} is not supported (expected {SCHEMA_VERSION})"
            )));
        }
        let count = decoder.read_u32()? as usize;
        if count == 0 {
            return Err(Error::corrupt("transaction contains no operations"));
        }
        if count > MAX_TRANSACTION_OPERATIONS {
            return Err(Error::limit_exceeded(format!(
                "transaction has {count} operations, above the {MAX_TRANSACTION_OPERATIONS}-operation cap"
            )));
        }
        let mut operations = Vec::with_capacity(count);
        for _ in 0..count {
            operations.push(decode_operation(&mut decoder)?);
        }
        decoder.finish()?;
        Ok(Self {
            schema_version,
            operations,
        })
    }
}

fn validate_text_field(value: &str, max_bytes: usize, name: &str) -> Result<()> {
    if value.is_empty() {
        return Err(Error::usage(format!("{name} must not be empty")));
    }
    if value.len() > max_bytes {
        return Err(Error::limit_exceeded(format!(
            "{name} is {} bytes, above the {max_bytes}-byte cap",
            value.len()
        )));
    }
    if value.chars().any(char::is_control) {
        return Err(Error::usage(format!(
            "{name} must not contain control characters"
        )));
    }
    Ok(())
}

fn normalize_embedding(values: &mut [f32]) -> Result<()> {
    if values.is_empty() {
        return Err(Error::usage("embedding must have at least one dimension"));
    }
    if values.len() > MAX_EMBEDDING_DIMENSION {
        return Err(Error::limit_exceeded(format!(
            "embedding has {} dimensions, above the {MAX_EMBEDDING_DIMENSION}-dimension cap",
            values.len()
        )));
    }

    let mut sum = 0.0f64;
    for value in values.iter() {
        if !value.is_finite() {
            return Err(Error::usage("embedding contains a non-finite value"));
        }
        sum += f64::from(*value) * f64::from(*value);
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(Error::usage(
            "embedding norm is zero or non-finite; cosine similarity is undefined",
        ));
    }

    let norm = sum.sqrt();
    for value in values.iter_mut() {
        let normalized = (f64::from(*value) / norm) as f32;
        if !normalized.is_finite() {
            return Err(Error::usage(
                "embedding normalization produced a non-finite value",
            ));
        }
        *value = normalized;
    }
    Ok(())
}

fn validate_stored_embedding(values: &[f32]) -> Result<()> {
    if values.is_empty() {
        return Err(Error::corrupt("stored embedding has no dimensions"));
    }
    if values.len() > MAX_EMBEDDING_DIMENSION {
        return Err(Error::limit_exceeded(format!(
            "stored embedding has {} dimensions, above the {MAX_EMBEDDING_DIMENSION}-dimension cap",
            values.len()
        )));
    }

    let mut sum = 0.0f64;
    for value in values {
        if !value.is_finite() {
            return Err(Error::corrupt(
                "stored embedding contains a non-finite value",
            ));
        }
        sum += f64::from(*value) * f64::from(*value);
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(Error::corrupt(
            "stored embedding norm is zero or non-finite",
        ));
    }
    let norm = sum.sqrt();
    if (norm - 1.0).abs() > NORMALIZED_TOLERANCE {
        return Err(Error::corrupt(format!(
            "stored embedding has norm {norm}, expected 1.0 within {NORMALIZED_TOLERANCE}"
        )));
    }
    Ok(())
}

fn encode_operation(encoder: &mut Encoder, operation: &Operation) -> Result<()> {
    match operation {
        Operation::PutDocument(document) => {
            encoder.write_u8(1)?;
            encoder.write_string(document.id.as_str())?;
            encoder.write_u8(document.modality.code())?;
            match document.title() {
                Some(title) => {
                    encoder.write_u8(1)?;
                    encoder.write_string(title)?;
                }
                None => encoder.write_u8(0)?,
            }
            match document.source() {
                Some(source) => {
                    encoder.write_u8(1)?;
                    encoder.write_string(source)?;
                }
                None => encoder.write_u8(0)?,
            }
            let count = u32::try_from(document.metadata.len())
                .map_err(|_| Error::internal("metadata entry count does not fit u32"))?;
            encoder.write_u32(count)?;
            for (key, value) in document.metadata.iter() {
                encoder.write_string(key)?;
                encoder.write_string(value)?;
            }
        }
        Operation::PutChunk(chunk) => {
            encoder.write_u8(2)?;
            encoder.write_u64(chunk.id.get())?;
            encoder.write_string(chunk.document_id.as_str())?;
            encoder.write_u32(chunk.ordinal)?;
            encoder.write_string(chunk.text())?;
            let dimension = u32::try_from(chunk.dimension())
                .map_err(|_| Error::internal("embedding dimension does not fit u32"))?;
            encoder.write_u32(dimension)?;
            for value in chunk.embedding() {
                encoder.write_f32(*value)?;
            }
        }
        Operation::DeleteDocument(id) => {
            encoder.write_u8(3)?;
            encoder.write_string(id.as_str())?;
        }
    }
    Ok(())
}

fn decode_operation(decoder: &mut Decoder<'_>) -> Result<Operation> {
    let tag = decoder.read_u8()?;
    match tag {
        1 => {
            let id = DocumentId::new(decoder.read_string(MAX_ID_BYTES)?).map_err(|error| {
                Error::corrupt(format!("invalid document id in storage: {error}"))
            })?;
            let modality_code = decoder.read_u8()?;
            let modality = Modality::from_code(modality_code).ok_or_else(|| {
                Error::corrupt(format!("unknown modality tag {modality_code} in storage"))
            })?;
            let mut document = Document::new(id, modality);

            match decoder.read_u8()? {
                0 => {}
                1 => {
                    let title = decoder.read_string(MAX_TITLE_BYTES)?;
                    document = document.with_title(title).map_err(|error| {
                        Error::corrupt(format!("invalid document title in storage: {error}"))
                    })?;
                }
                flag => {
                    return Err(Error::corrupt(format!(
                        "invalid title presence flag {flag} in storage"
                    )));
                }
            }
            match decoder.read_u8()? {
                0 => {}
                1 => {
                    let source = decoder.read_string(MAX_SOURCE_BYTES)?;
                    document = document.with_source(source).map_err(|error| {
                        Error::corrupt(format!("invalid document source in storage: {error}"))
                    })?;
                }
                flag => {
                    return Err(Error::corrupt(format!(
                        "invalid source presence flag {flag} in storage"
                    )));
                }
            }

            let metadata_count = decoder.read_u32()? as usize;
            if metadata_count > MAX_METADATA_ENTRIES {
                return Err(Error::limit_exceeded(format!(
                    "stored document has {metadata_count} metadata entries, above the {MAX_METADATA_ENTRIES}-entry cap"
                )));
            }
            let mut metadata = Metadata::new();
            for _ in 0..metadata_count {
                let key = decoder.read_string(MAX_METADATA_KEY_BYTES)?;
                let value = decoder.read_string(MAX_METADATA_VALUE_BYTES)?;
                metadata.insert(key, value).map_err(|error| {
                    Error::corrupt(format!("invalid metadata in storage: {error}"))
                })?;
            }
            Ok(Operation::PutDocument(document.with_metadata(metadata)))
        }
        2 => {
            let raw_id = decoder.read_u64()?;
            let id = ChunkId::new(raw_id)
                .map_err(|error| Error::corrupt(format!("invalid chunk id in storage: {error}")))?;
            let document_id =
                DocumentId::new(decoder.read_string(MAX_ID_BYTES)?).map_err(|error| {
                    Error::corrupt(format!("invalid document id in chunk: {error}"))
                })?;
            let ordinal = decoder.read_u32()?;
            let text = decoder.read_string(MAX_TEXT_BYTES)?;
            let dimension = decoder.read_u32()? as usize;
            if dimension == 0 {
                return Err(Error::corrupt("stored embedding has zero dimensions"));
            }
            if dimension > MAX_EMBEDDING_DIMENSION {
                return Err(Error::limit_exceeded(format!(
                    "stored embedding has {dimension} dimensions, above the {MAX_EMBEDDING_DIMENSION}-dimension cap"
                )));
            }
            let mut embedding = Vec::with_capacity(dimension);
            for _ in 0..dimension {
                embedding.push(decoder.read_f32()?);
            }
            let chunk = Chunk::from_stored(id, document_id, ordinal, text, embedding)?;
            Ok(Operation::PutChunk(chunk))
        }
        3 => {
            let id = DocumentId::new(decoder.read_string(MAX_ID_BYTES)?).map_err(|error| {
                Error::corrupt(format!("invalid deleted document id in storage: {error}"))
            })?;
            Ok(Operation::DeleteDocument(id))
        }
        other => Err(Error::corrupt(format!(
            "unknown storage operation tag {other}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn document() -> Document {
        let mut metadata = Metadata::new();
        metadata.insert("lang", "en").expect("metadata");
        Document::new(DocumentId::new("doc-1").expect("id"), Modality::Text)
            .with_title("A title")
            .expect("title")
            .with_source("fixtures/doc.txt")
            .expect("source")
            .with_metadata(metadata)
    }

    fn chunk(document_id: DocumentId) -> Chunk {
        Chunk::new(
            ChunkId::new(7).expect("chunk id"),
            document_id,
            0,
            "hello world".to_owned(),
            vec![3.0, 4.0],
        )
        .expect("chunk")
    }

    #[test]
    fn document_ids_reject_invalid_values() {
        for value in ["", "a\u{0}b"] {
            let error = DocumentId::new(value).expect_err("invalid id must fail");
            assert!(matches!(error, Error::Usage(_)), "{error}");
        }
        let long = "a".repeat(MAX_ID_BYTES + 1);
        let error = DocumentId::new(long).expect_err("long id must fail");
        assert!(matches!(error, Error::LimitExceeded(_)));
        assert_eq!(DocumentId::new("doc-1").expect("valid").as_str(), "doc-1");
    }

    #[test]
    fn chunk_ids_must_be_positive() {
        assert!(ChunkId::new(0).is_err());
        assert_eq!(ChunkId::new(1).expect("valid").get(), 1);
    }

    #[test]
    fn metadata_enforces_caps_and_control_characters() {
        let mut metadata = Metadata::new();
        metadata.insert("lang", "en").expect("valid");
        assert_eq!(metadata.get("lang"), Some("en"));
        assert_eq!(metadata.iter().collect::<Vec<_>>(), vec![("lang", "en")]);

        let error = metadata.insert("", "value").expect_err("empty key");
        assert!(matches!(error, Error::Usage(_)));
        let error = metadata
            .insert("bad\nkey", "value")
            .expect_err("control key");
        assert!(matches!(error, Error::Usage(_)));
        let error = metadata
            .insert("key", "x".repeat(MAX_METADATA_VALUE_BYTES + 1))
            .expect_err("long value");
        assert!(matches!(error, Error::LimitExceeded(_)));

        let mut full = Metadata::new();
        for index in 0..MAX_METADATA_ENTRIES {
            full.insert(format!("key{index}"), "value")
                .expect("entry within cap");
        }
        let error = full.insert("overflow", "value").expect_err("entry cap");
        assert!(matches!(error, Error::LimitExceeded(_)));
    }

    #[test]
    fn document_fields_are_validated() {
        let id = DocumentId::new("doc").expect("id");
        let document = Document::new(id.clone(), Modality::Text);
        assert_eq!(document.id(), &id);
        assert_eq!(document.modality(), Modality::Text);
        assert!(document.title().is_none());

        let error = document.clone().with_title("").expect_err("empty title");
        assert!(matches!(error, Error::Usage(_)));
        let error = document
            .clone()
            .with_title("bad\ttitle")
            .expect_err("control");
        assert!(matches!(error, Error::Usage(_)));
        let error = document
            .clone()
            .with_source("x".repeat(MAX_SOURCE_BYTES + 1))
            .expect_err("long source");
        assert!(matches!(error, Error::LimitExceeded(_)));
    }

    #[test]
    fn chunk_normalizes_its_embedding() {
        let chunk = chunk(DocumentId::new("doc").expect("id"));
        assert_eq!(chunk.dimension(), 2);
        let norm = chunk
            .embedding()
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((chunk.embedding()[0] - 0.6).abs() < 1e-6);
        assert!((chunk.embedding()[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn chunk_rejects_invalid_embeddings() {
        let document_id = DocumentId::new("doc").expect("id");
        let id = ChunkId::new(1).expect("id");
        for embedding in [vec![], vec![0.0, 0.0], vec![f32::NAN], vec![f32::INFINITY]] {
            let error = Chunk::new(id, document_id.clone(), 0, "text".to_owned(), embedding)
                .expect_err("invalid embedding must fail");
            assert!(matches!(error, Error::Usage(_)), "{error}");
        }
        let error = Chunk::new(
            id,
            document_id,
            0,
            "text".to_owned(),
            vec![0.0; MAX_EMBEDDING_DIMENSION + 1],
        )
        .expect_err("dimension cap");
        assert!(matches!(error, Error::LimitExceeded(_)));
    }

    #[test]
    fn stored_embeddings_must_be_normalized() {
        let document_id = DocumentId::new("doc").expect("id");
        let id = ChunkId::new(1).expect("id");
        let error = Chunk::from_stored(
            id,
            document_id.clone(),
            0,
            "text".to_owned(),
            vec![1.0, 1.0],
        )
        .expect_err("unnormalized stored vector");
        assert!(matches!(error, Error::Corrupt(_)));

        let chunk = Chunk::from_stored(id, document_id, 0, "text".to_owned(), vec![0.6, 0.8])
            .expect("normalized stored vector");
        assert_eq!(chunk.dimension(), 2);
    }

    #[test]
    fn transaction_round_trips_every_operation() {
        let document = document();
        let chunk = chunk(document.id().clone());
        let transaction = Transaction::new(vec![
            Operation::PutDocument(document.clone()),
            Operation::PutChunk(chunk.clone()),
            Operation::DeleteDocument(DocumentId::new("doc-2").expect("id")),
        ])
        .expect("transaction");
        let bytes = transaction.encode().expect("encode");
        let decoded = Transaction::decode(&bytes).expect("decode");
        assert_eq!(decoded.schema_version(), SCHEMA_VERSION);
        assert_eq!(decoded.operations(), transaction.operations());
    }

    #[test]
    fn transaction_rejects_empty_and_oversized_operation_lists() {
        let error = Transaction::new(Vec::new()).expect_err("empty transaction");
        assert!(matches!(error, Error::Usage(_)));

        let document = document();
        let operations = (0..=MAX_TRANSACTION_OPERATIONS)
            .map(|_| Operation::PutDocument(document.clone()))
            .collect();
        let error = Transaction::new(operations).expect_err("operation cap");
        assert!(matches!(error, Error::LimitExceeded(_)));
    }

    #[test]
    fn transaction_decode_rejects_unknown_schema_and_trailing_bytes() {
        let document = document();
        let transaction =
            Transaction::new(vec![Operation::PutDocument(document)]).expect("transaction");
        let mut bytes = transaction.encode().expect("encode");
        bytes[0] = 2;
        let error = Transaction::decode(&bytes).expect_err("unknown schema");
        assert!(matches!(error, Error::Unsupported(_)));

        let mut bytes = transaction.encode().expect("encode");
        bytes.push(0);
        let error = Transaction::decode(&bytes).expect_err("trailing byte");
        assert!(matches!(error, Error::Corrupt(_)));
    }

    #[test]
    fn transaction_decode_rejects_unknown_tags_and_short_payloads() {
        let error = Transaction::decode(&[1, 0, 1, 0, 0, 0, 9]).expect_err("unknown tag");
        assert!(matches!(error, Error::Corrupt(_)));

        let error = Transaction::decode(&[1, 0]).expect_err("truncated");
        assert!(matches!(error, Error::Corrupt(_)));
    }
}
