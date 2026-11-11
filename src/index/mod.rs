//! Deterministic hybrid search over the persistent store.
//!
//! Milestone M3 combines a BM25 inverted index with exact cosine vector search
//! and Reciprocal Rank Fusion. The index is rebuilt from the journal-backed
//! store on open; no approximate index is introduced before measurements show
//! that exact search is insufficient. Filters are applied before ranking, and
//! pagination slices a fully deterministic ranking.

pub mod hybrid;
pub mod lexical;
pub mod tokenize;
pub mod vector;

use crate::error::{Error, Result};
use crate::storage::model::{
    Chunk, ChunkId, Document, DocumentId, MAX_EMBEDDING_DIMENSION, Metadata, Modality,
};
use crate::storage::{ReadOnlyStore, Store};

/// Maximum number of hits returned by one search.
pub const MAX_SEARCH_LIMIT: usize = 1000;
/// Maximum pagination offset accepted by one search.
pub const MAX_SEARCH_OFFSET: usize = 1_000_000;

/// Search modality and metadata filters.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Filters {
    modality: Option<Modality>,
    metadata: Vec<(String, String)>,
}

impl Filters {
    /// Creates an empty filter set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Restricts results to one modality.
    pub fn with_modality(mut self, modality: Modality) -> Self {
        self.modality = Some(modality);
        self
    }

    /// Adds an exact metadata equality filter after validating its key and value.
    pub fn add_metadata(&mut self, key: &str, value: &str) -> Result<()> {
        let mut check = Metadata::new();
        check.insert(key, value)?;
        self.metadata.push((key.to_owned(), value.to_owned()));
        Ok(())
    }

    /// Selected modality filter, if any.
    pub fn modality(&self) -> Option<Modality> {
        self.modality
    }

    /// Metadata equality filters in insertion order.
    pub fn metadata(&self) -> &[(String, String)] {
        &self.metadata
    }

    /// Returns true when a document satisfies every filter.
    pub fn matches(&self, document: &Document) -> bool {
        if let Some(modality) = self.modality
            && document.modality() != modality
        {
            return false;
        }
        self.metadata
            .iter()
            .all(|(key, value)| document.metadata().get(key) == Some(value.as_str()))
    }

    /// Returns true when no filter is active.
    pub fn is_empty(&self) -> bool {
        self.modality.is_none() && self.metadata.is_empty()
    }
}

/// Ranking strategy for one search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    /// BM25 and exact vector search fused with RRF.
    Hybrid,
    /// BM25 lexical search only.
    Lexical,
    /// Exact cosine vector search only.
    Vector,
}

impl SearchMode {
    /// Short name used by the CLI.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hybrid => "hybrid",
            Self::Lexical => "lexical",
            Self::Vector => "vector",
        }
    }

    /// Parses a mode name.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "hybrid" => Some(Self::Hybrid),
            "lexical" => Some(Self::Lexical),
            "vector" => Some(Self::Vector),
            _ => None,
        }
    }
}

/// One search request.
#[derive(Debug, Clone)]
pub struct SearchRequest<'a> {
    /// Ranking strategy.
    pub mode: SearchMode,
    /// Query text for lexical or hybrid search.
    pub text: Option<&'a str>,
    /// Query embedding for vector or hybrid search.
    pub embedding: Option<&'a [f32]>,
    /// Modality and metadata filters.
    pub filters: Filters,
    /// Maximum number of hits to return, from 1 to MAX_SEARCH_LIMIT.
    pub limit: usize,
    /// Number of ranked hits to skip.
    pub offset: usize,
}

impl<'a> SearchRequest<'a> {
    /// Creates a request with an empty filter set and no pagination offset.
    pub fn new(mode: SearchMode, limit: usize) -> Self {
        Self {
            mode,
            text: None,
            embedding: None,
            filters: Filters::new(),
            limit,
            offset: 0,
        }
    }

    /// Validates the mode inputs, pagination bounds, and query sizes.
    pub fn validate(&self) -> Result<()> {
        if self.limit == 0 || self.limit > MAX_SEARCH_LIMIT {
            return Err(Error::usage(format!(
                "search limit must be between 1 and {MAX_SEARCH_LIMIT}"
            )));
        }
        if self.offset > MAX_SEARCH_OFFSET {
            return Err(Error::usage(format!(
                "search offset must not exceed {MAX_SEARCH_OFFSET}"
            )));
        }

        match self.mode {
            SearchMode::Hybrid => {
                if self.text.is_none() || self.embedding.is_none() {
                    return Err(Error::usage(
                        "hybrid search requires both query text and a query embedding",
                    ));
                }
            }
            SearchMode::Lexical => {
                if self.text.is_none() {
                    return Err(Error::usage("lexical search requires query text"));
                }
            }
            SearchMode::Vector => {
                if self.embedding.is_none() {
                    return Err(Error::usage("vector search requires a query embedding"));
                }
            }
        }

        if let Some(text) = self.text
            && text.len() > tokenize::MAX_TOKENIZE_BYTES
        {
            return Err(Error::limit_exceeded(format!(
                "query text is {} bytes, above the {}-byte cap",
                text.len(),
                tokenize::MAX_TOKENIZE_BYTES
            )));
        }
        if let Some(embedding) = self.embedding {
            if embedding.is_empty() {
                return Err(Error::usage(
                    "query embedding must have at least one dimension",
                ));
            }
            if embedding.len() > MAX_EMBEDDING_DIMENSION {
                return Err(Error::limit_exceeded(format!(
                    "query embedding has {} dimensions, above the {MAX_EMBEDDING_DIMENSION}-dimension cap",
                    embedding.len()
                )));
            }
            if embedding.iter().any(|value| !value.is_finite()) {
                return Err(Error::usage("query embedding contains a non-finite value"));
            }
        }
        Ok(())
    }
}

/// One ranked search hit.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit<'a> {
    /// Chunk identifier.
    pub chunk_id: ChunkId,
    /// Owning document identifier.
    pub document_id: &'a DocumentId,
    /// Zero-based chunk ordinal inside the document.
    pub ordinal: u32,
    /// BM25 score, cosine score, or RRF score depending on the mode.
    pub score: f64,
    /// One-based lexical rank, when the hit appeared in the lexical list.
    pub lexical_rank: Option<usize>,
    /// One-based vector rank, when the hit appeared in the vector list.
    pub vector_rank: Option<usize>,
    /// Chunk text.
    pub text: &'a str,
    /// Document title, if any.
    pub title: Option<&'a str>,
    /// Document source, if any.
    pub source: Option<&'a str>,
    /// Document modality.
    pub modality: Modality,
    /// Document metadata.
    pub metadata: &'a Metadata,
}

#[derive(Debug)]
pub(crate) struct IndexedChunk<'a> {
    pub(crate) chunk: &'a Chunk,
    pub(crate) document: &'a Document,
    pub(crate) tokens: Vec<String>,
}

/// In-memory hybrid index rebuilt from a store.
#[derive(Debug)]
pub struct Index<'a> {
    chunks: Vec<IndexedChunk<'a>>,
    lexical: lexical::LexicalIndex,
    dimension: Option<usize>,
}

impl<'a> Index<'a> {
    /// Builds the lexical and vector views from the store's committed chunks.
    ///
    /// Tokenization is capped per chunk; a document that exceeds the cap is an
    /// explicit limit error rather than a silently truncated index entry.
    pub fn build(store: &'a Store) -> Result<Self> {
        Self::build_from(
            store.chunks(),
            |id| store.document(id),
            store.embedding_dimension(),
        )
    }

    /// Builds the index from a lock-free read-only store view.
    ///
    /// This is safe while a writer holds the store lock. A concurrent torn
    /// append is excluded from the read-only view by its recovery logic.
    pub fn build_read_only(store: &'a ReadOnlyStore) -> Result<Self> {
        Self::build_from(
            store.chunks(),
            |id| store.document(id),
            store.embedding_dimension(),
        )
    }

    fn build_from<I, F>(chunks: I, document: F, dimension: Option<usize>) -> Result<Self>
    where
        I: Iterator<Item = &'a Chunk>,
        F: Fn(&DocumentId) -> Option<&'a Document>,
    {
        let mut indexed = Vec::new();
        for chunk in chunks {
            let document = document(chunk.document_id()).ok_or_else(|| {
                Error::internal(format!(
                    "chunk {} references missing document {}",
                    chunk.id(),
                    chunk.document_id()
                ))
            })?;
            if let Some(dimension) = dimension
                && chunk.dimension() != dimension
            {
                return Err(Error::internal(format!(
                    "chunk {} has dimension {} but the store dimension is {dimension}",
                    chunk.id(),
                    chunk.dimension()
                )));
            }
            let tokens = tokenize::tokenize(chunk.text(), tokenize::MAX_INDEX_TOKENS_PER_CHUNK)?;
            indexed.push(IndexedChunk {
                chunk,
                document,
                tokens,
            });
        }

        let lexical = lexical::LexicalIndex::build(&indexed);
        Ok(Self {
            chunks: indexed,
            lexical,
            dimension,
        })
    }

    /// Number of indexed chunks.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Embedding dimension shared by the indexed chunks, if any.
    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    /// Executes a validated search request.
    ///
    /// The returned hits borrow the store and remain valid as long as the store
    /// is not mutated. Filters are applied before ranking, and pagination is a
    /// slice of the full deterministic ranking.
    pub fn search(&self, request: &SearchRequest<'_>) -> Result<Vec<SearchHit<'a>>> {
        request.validate()?;

        match request.mode {
            SearchMode::Lexical => {
                let text = request
                    .text
                    .ok_or_else(|| Error::usage("lexical search requires query text"))?;
                let query = tokenize::tokenize(text, tokenize::MAX_QUERY_TOKENS)?;
                let ranked = self.lexical.rank(&query, &self.chunks, &request.filters);
                Ok(ranked
                    .iter()
                    .enumerate()
                    .skip(request.offset)
                    .take(request.limit)
                    .map(|(position, (index, score))| {
                        self.hit(*index, *score, Some(position + 1), None)
                    })
                    .collect())
            }
            SearchMode::Vector => {
                let embedding = request
                    .embedding
                    .ok_or_else(|| Error::usage("vector search requires a query embedding"))?;
                let ranked =
                    vector::rank(embedding, &self.chunks, &request.filters, self.dimension)?;
                Ok(ranked
                    .iter()
                    .enumerate()
                    .skip(request.offset)
                    .take(request.limit)
                    .map(|(position, (index, score))| {
                        self.hit(*index, *score, None, Some(position + 1))
                    })
                    .collect())
            }
            SearchMode::Hybrid => {
                let text = request
                    .text
                    .ok_or_else(|| Error::usage("hybrid search requires query text"))?;
                let embedding = request
                    .embedding
                    .ok_or_else(|| Error::usage("hybrid search requires a query embedding"))?;
                let query = tokenize::tokenize(text, tokenize::MAX_QUERY_TOKENS)?;
                let lexical_ranked = self.lexical.rank(&query, &self.chunks, &request.filters);
                let vector_ranked =
                    vector::rank(embedding, &self.chunks, &request.filters, self.dimension)?;
                let fused = hybrid::fuse(&lexical_ranked, &vector_ranked, &self.chunks);
                Ok(fused
                    .iter()
                    .skip(request.offset)
                    .take(request.limit)
                    .map(|candidate| {
                        self.hit(
                            candidate.index,
                            candidate.score,
                            candidate.lexical_rank,
                            candidate.vector_rank,
                        )
                    })
                    .collect())
            }
        }
    }

    fn hit(
        &self,
        index: usize,
        score: f64,
        lexical_rank: Option<usize>,
        vector_rank: Option<usize>,
    ) -> SearchHit<'a> {
        let indexed = &self.chunks[index];
        SearchHit {
            chunk_id: indexed.chunk.id(),
            document_id: indexed.document.id(),
            ordinal: indexed.chunk.ordinal(),
            score,
            lexical_rank,
            vector_rank,
            text: indexed.chunk.text(),
            title: indexed.document.title(),
            source: indexed.document.source(),
            modality: indexed.document.modality(),
            metadata: indexed.document.metadata(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::model::ChunkInput;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir(name: &str) -> PathBuf {
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "seldon-index-{}-{}-{unique}",
            std::process::id(),
            name
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create temp dir");
        path
    }

    fn document(id: &str, language: &str) -> Document {
        let mut metadata = Metadata::new();
        metadata.insert("lang", language).expect("metadata");
        Document::new(DocumentId::new(id).expect("id"), Modality::Text).with_metadata(metadata)
    }

    fn build_store(dir: &std::path::Path) -> Store {
        let mut store = Store::open(dir).expect("open store");
        store
            .put_document(
                document("doc-1", "en"),
                vec![ChunkInput::new("rust language systems", vec![1.0, 0.0])],
            )
            .expect("doc 1");
        store
            .put_document(
                document("doc-2", "fr"),
                vec![ChunkInput::new("python language scripting", vec![0.0, 1.0])],
            )
            .expect("doc 2");
        store
            .put_document(
                document("doc-3", "en"),
                vec![ChunkInput::new("rust rust rust", vec![1.0, 1.0])],
            )
            .expect("doc 3");
        store
    }

    fn lexical_request<'a>(text: &'a str) -> SearchRequest<'a> {
        let mut request = SearchRequest::new(SearchMode::Lexical, 10);
        request.text = Some(text);
        request
    }

    fn vector_request<'a>(embedding: &'a [f32]) -> SearchRequest<'a> {
        let mut request = SearchRequest::new(SearchMode::Vector, 10);
        request.embedding = Some(embedding);
        request
    }

    #[test]
    fn lexical_search_orders_by_bm25() {
        let dir = temp_dir("lexical");
        let store = build_store(&dir);
        let index = Index::build(&store).expect("index");
        let hits = index
            .search(&lexical_request("rust"))
            .expect("lexical search");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].chunk_id.get(), 3);
        assert_eq!(hits[1].chunk_id.get(), 1);
        assert_eq!(hits[0].lexical_rank, Some(1));
        assert_eq!(hits[1].lexical_rank, Some(2));
        assert_eq!(hits[0].vector_rank, None);
        assert!(hits[0].score > hits[1].score);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn vector_search_orders_by_cosine() {
        let dir = temp_dir("vector");
        let store = build_store(&dir);
        let index = Index::build(&store).expect("index");
        let hits = index
            .search(&vector_request(&[1.0, 0.0]))
            .expect("vector search");
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].chunk_id.get(), 1);
        assert_eq!(hits[1].chunk_id.get(), 3);
        assert_eq!(hits[2].chunk_id.get(), 2);
        assert_eq!(hits[0].vector_rank, Some(1));
        assert!((hits[0].score - 1.0).abs() < 1e-9);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn image_documents_are_vector_searchable_and_filterable() {
        let dir = temp_dir("image");
        let mut store = Store::open(&dir).expect("open store");
        store
            .put_document(
                Document::new(DocumentId::new("img-1").expect("id"), Modality::Image)
                    .with_source("fixtures/media/quadrants_16x16.png")
                    .expect("source"),
                vec![ChunkInput::new("", vec![0.0, 1.0])],
            )
            .expect("image document");
        store
            .put_document(
                document("doc-1", "en"),
                vec![ChunkInput::new("rust language", vec![1.0, 0.0])],
            )
            .expect("text document");

        let index = Index::build(&store).expect("index");
        let hits = index
            .search(&vector_request(&[0.0, 1.0]))
            .expect("vector search");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].document_id.as_str(), "img-1");
        assert_eq!(hits[0].modality, Modality::Image);
        assert_eq!(hits[0].text, "");
        assert_eq!(hits[0].source, Some("fixtures/media/quadrants_16x16.png"));

        let lexical = index
            .search(&lexical_request("rust"))
            .expect("lexical search");
        assert_eq!(lexical.len(), 1);
        assert_eq!(lexical[0].document_id.as_str(), "doc-1");

        let mut request = vector_request(&[0.0, 1.0]);
        request.filters = Filters::new().with_modality(Modality::Image);
        let filtered = index.search(&request).expect("filtered vector search");
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].document_id.as_str(), "img-1");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn hybrid_search_fuses_lexical_and_vector_ranks() {
        let dir = temp_dir("hybrid");
        let store = build_store(&dir);
        let index = Index::build(&store).expect("index");
        let mut request = SearchRequest::new(SearchMode::Hybrid, 10);
        request.text = Some("language");
        request.embedding = Some(&[1.0, 0.0]);
        let hits = index.search(&request).expect("hybrid search");
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].chunk_id.get(), 1);
        assert_eq!(hits[1].chunk_id.get(), 2);
        assert_eq!(hits[2].chunk_id.get(), 3);
        assert_eq!(hits[0].lexical_rank, Some(1));
        assert_eq!(hits[0].vector_rank, Some(1));
        assert_eq!(hits[1].lexical_rank, Some(2));
        assert_eq!(hits[1].vector_rank, Some(3));
        assert_eq!(hits[2].lexical_rank, None);
        assert_eq!(hits[2].vector_rank, Some(2));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn filters_apply_before_ranking() {
        let dir = temp_dir("filters");
        let store = build_store(&dir);
        let index = Index::build(&store).expect("index");
        let mut filters = Filters::new();
        filters.add_metadata("lang", "fr").expect("filter");
        let mut request = lexical_request("language");
        request.filters = filters.clone();
        let hits = index.search(&request).expect("filtered lexical");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].document_id.as_str(), "doc-2");
        assert_eq!(hits[0].lexical_rank, Some(1));

        let embedding = [1.0, 0.0];
        let mut request = vector_request(&embedding);
        request.filters = filters;
        let hits = index.search(&request).expect("filtered vector");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].document_id.as_str(), "doc-2");
        assert_eq!(hits[0].vector_rank, Some(1));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn pagination_slices_the_deterministic_ranking() {
        let dir = temp_dir("pagination");
        let store = build_store(&dir);
        let index = Index::build(&store).expect("index");
        let mut request = lexical_request("language");
        request.limit = 1;
        request.offset = 1;
        let hits = index.search(&request).expect("paginated");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].chunk_id.get(), 2);
        assert_eq!(hits[0].lexical_rank, Some(2));

        request.offset = 10;
        assert!(index.search(&request).expect("empty page").is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_store_yields_no_hits() {
        let dir = temp_dir("empty");
        let store = Store::open(&dir).expect("store");
        let index = Index::build(&store).expect("index");
        assert_eq!(index.chunk_count(), 0);
        assert_eq!(index.dimension(), None);
        assert!(
            index
                .search(&lexical_request("rust"))
                .expect("lexical")
                .is_empty()
        );
        assert!(
            index
                .search(&vector_request(&[1.0, 0.0]))
                .expect("vector")
                .is_empty()
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn dimension_mismatch_is_rejected() {
        let dir = temp_dir("dimension");
        let store = build_store(&dir);
        let index = Index::build(&store).expect("index");
        let error = index
            .search(&vector_request(&[1.0, 0.0, 0.0]))
            .expect_err("dimension mismatch");
        assert!(matches!(error, Error::Usage(_)), "{error}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn request_validation_rejects_missing_inputs_and_bad_bounds() {
        let dir = temp_dir("validation");
        let store = build_store(&dir);
        let index = Index::build(&store).expect("index");

        let error = index
            .search(&SearchRequest::new(SearchMode::Hybrid, 10))
            .expect_err("hybrid without inputs");
        assert!(matches!(error, Error::Usage(_)), "{error}");

        let embedding = [1.0, 0.0];
        let mut request = SearchRequest::new(SearchMode::Lexical, 10);
        request.embedding = Some(&embedding);
        let error = index.search(&request).expect_err("lexical without text");
        assert!(matches!(error, Error::Usage(_)), "{error}");

        let mut request = SearchRequest::new(SearchMode::Vector, 10);
        request.text = Some("rust");
        let error = index
            .search(&request)
            .expect_err("vector without embedding");
        assert!(matches!(error, Error::Usage(_)), "{error}");

        for limit in [0, MAX_SEARCH_LIMIT + 1] {
            let mut request = lexical_request("rust");
            request.limit = limit;
            let error = index.search(&request).expect_err("bad limit");
            assert!(matches!(error, Error::Usage(_)), "{error}");
        }

        let mut request = lexical_request("rust");
        request.offset = MAX_SEARCH_OFFSET + 1;
        let error = index.search(&request).expect_err("bad offset");
        assert!(matches!(error, Error::Usage(_)), "{error}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn searches_are_deterministic() {
        let dir = temp_dir("determinism");
        let store = build_store(&dir);
        let index = Index::build(&store).expect("index");
        let embedding = [1.0, 0.0];
        let mut request = SearchRequest::new(SearchMode::Hybrid, 10);
        request.text = Some("rust language");
        request.embedding = Some(&embedding);
        let first = index.search(&request).expect("first");
        let second = index.search(&request).expect("second");
        assert_eq!(first, second);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn token_cap_is_an_explicit_build_error() {
        let dir = temp_dir("token-cap");
        let mut store = Store::open(&dir).expect("store");
        let text = "word ".repeat(tokenize::MAX_INDEX_TOKENS_PER_CHUNK + 1);
        store
            .put_document(
                document("doc-1", "en"),
                vec![ChunkInput::new(text, vec![1.0, 0.0])],
            )
            .expect("put document");
        let error = Index::build(&store).expect_err("token cap");
        assert!(matches!(error, Error::LimitExceeded(_)), "{error}");
        let _ = fs::remove_dir_all(&dir);
    }
}
