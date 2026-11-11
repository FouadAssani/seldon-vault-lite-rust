//! Exact vector search using cosine similarity.
//!
//! This is the correctness oracle for milestone M3. The stored embeddings are
//! normalized at ingestion; the query is normalized on the fly so an arbitrary
//! finite vector still produces a cosine score. No approximate index (HNSW) is
//! introduced before volume and latency measurements justify it.

use crate::error::{Error, Result};
use crate::index::{Filters, IndexedChunk};
use crate::storage::model::MAX_EMBEDDING_DIMENSION;

/// Ranks chunks by cosine similarity to the query.
///
/// Returns (chunk index, cosine score) sorted by descending score then
/// ascending chunk id. A query whose dimension differs from the store
/// dimension is a usage error; an empty store simply yields no hits.
pub(crate) fn rank(
    query: &[f32],
    chunks: &[IndexedChunk<'_>],
    filters: &Filters,
    dimension: Option<usize>,
) -> Result<Vec<(usize, f64)>> {
    if query.is_empty() {
        return Err(Error::usage(
            "query embedding must have at least one dimension",
        ));
    }
    if query.len() > MAX_EMBEDDING_DIMENSION {
        return Err(Error::limit_exceeded(format!(
            "query embedding has {} dimensions, above the {MAX_EMBEDDING_DIMENSION}-dimension cap",
            query.len()
        )));
    }
    let mut query_sum = 0.0f64;
    for value in query {
        if !value.is_finite() {
            return Err(Error::usage("query embedding contains a non-finite value"));
        }
        query_sum += f64::from(*value) * f64::from(*value);
    }
    if !query_sum.is_finite() || query_sum <= 0.0 {
        return Err(Error::usage(
            "query embedding norm is zero or non-finite; cosine similarity is undefined",
        ));
    }
    let query_norm = query_sum.sqrt();

    if let Some(dimension) = dimension
        && query.len() != dimension
    {
        return Err(Error::usage(format!(
            "query embedding dimension {} does not match the index dimension {dimension}",
            query.len()
        )));
    }

    let mut scored = Vec::new();
    for (index, indexed) in chunks.iter().enumerate() {
        if !filters.matches(indexed.document) {
            continue;
        }
        let vector = indexed.chunk.embedding();
        if vector.len() != query.len() {
            return Err(Error::internal(format!(
                "chunk {} has dimension {}, query has {}",
                indexed.chunk.id(),
                vector.len(),
                query.len()
            )));
        }

        let mut dot = 0.0f64;
        let mut vector_sum = 0.0f64;
        for (query_value, vector_value) in query.iter().zip(vector) {
            let query_value = f64::from(*query_value);
            let vector_value = f64::from(*vector_value);
            dot += query_value * vector_value;
            vector_sum += vector_value * vector_value;
        }
        if !vector_sum.is_finite() || vector_sum <= 0.0 {
            return Err(Error::internal(format!(
                "chunk {} has a zero or non-finite embedding norm",
                indexed.chunk.id()
            )));
        }
        let score = dot / (query_norm * vector_sum.sqrt());
        scored.push((index, score));
    }

    scored.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| chunk_order(chunks, left.0, right.0))
    });
    Ok(scored)
}

fn chunk_order(chunks: &[IndexedChunk<'_>], left: usize, right: usize) -> std::cmp::Ordering {
    chunks[left]
        .chunk
        .id()
        .get()
        .cmp(&chunks[right].chunk.id().get())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::model::{Chunk, ChunkId, Document, DocumentId, Metadata, Modality};

    fn document(id: &str) -> Document {
        Document::new(DocumentId::new(id).expect("id"), Modality::Text)
    }

    fn chunk(document: &Document, id: u64, embedding: Vec<f32>) -> Chunk {
        Chunk::new(
            ChunkId::new(id).expect("chunk id"),
            document.id().clone(),
            0,
            "text".to_owned(),
            embedding,
        )
        .expect("chunk")
    }

    fn indexed<'a>(documents: &'a [Document], chunks: &'a [Chunk]) -> Vec<IndexedChunk<'a>> {
        chunks
            .iter()
            .enumerate()
            .map(|(index, chunk)| IndexedChunk {
                chunk,
                document: &documents[index],
                tokens: Vec::new(),
            })
            .collect()
    }

    #[test]
    fn ranks_by_cosine_similarity() {
        let documents = vec![document("a"), document("b"), document("c")];
        let chunks = vec![
            chunk(&documents[0], 1, vec![1.0, 0.0]),
            chunk(&documents[1], 2, vec![0.0, 1.0]),
            chunk(&documents[2], 3, vec![1.0, 1.0]),
        ];
        let indexed_chunks = indexed(&documents, &chunks);
        let ranked = rank(&[1.0, 0.0], &indexed_chunks, &Filters::new(), Some(2)).expect("rank");
        assert_eq!(ranked.len(), 3);
        assert_eq!(ranked[0].0, 0);
        assert_eq!(ranked[1].0, 2);
        assert_eq!(ranked[2].0, 1);
        assert!((ranked[0].1 - 1.0).abs() < 1e-9);
        assert!((ranked[1].1 - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        assert!(ranked[2].1.abs() < 1e-9);
    }

    #[test]
    fn filters_exclude_documents() {
        let mut metadata = Metadata::new();
        metadata.insert("lang", "fr").expect("metadata");
        let documents = vec![
            document("a"),
            Document::new(DocumentId::new("b").expect("id"), Modality::Text)
                .with_metadata(metadata),
        ];
        let chunks = vec![
            chunk(&documents[0], 1, vec![1.0, 0.0]),
            chunk(&documents[1], 2, vec![1.0, 0.0]),
        ];
        let indexed_chunks = indexed(&documents, &chunks);
        let mut filters = Filters::new();
        filters.add_metadata("lang", "fr").expect("filter metadata");
        let ranked = rank(&[1.0, 0.0], &indexed_chunks, &filters, Some(2)).expect("rank");
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].0, 1);
    }

    #[test]
    fn dimension_mismatch_is_a_usage_error() {
        let documents = vec![document("a")];
        let chunks = vec![chunk(&documents[0], 1, vec![1.0, 0.0])];
        let indexed_chunks = indexed(&documents, &chunks);
        let error = rank(&[1.0, 0.0, 0.0], &indexed_chunks, &Filters::new(), Some(2))
            .expect_err("dimension mismatch");
        assert!(matches!(error, Error::Usage(_)), "{error}");
    }

    #[test]
    fn zero_and_non_finite_queries_are_rejected() {
        let documents = vec![document("a")];
        let chunks = vec![chunk(&documents[0], 1, vec![1.0, 0.0])];
        let indexed_chunks = indexed(&documents, &chunks);
        for query in [&[0.0, 0.0][..], &[f32::NAN, 1.0], &[f32::INFINITY, 0.0]] {
            let error =
                rank(query, &indexed_chunks, &Filters::new(), Some(2)).expect_err("invalid query");
            assert!(matches!(error, Error::Usage(_)), "{error}");
        }
    }

    #[test]
    fn empty_index_yields_no_hits() {
        let ranked = rank(&[1.0, 0.0], &[], &Filters::new(), None).expect("rank");
        assert!(ranked.is_empty());
    }
}
