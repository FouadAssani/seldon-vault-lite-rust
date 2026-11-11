//! Reciprocal Rank Fusion (RRF) for lexical and vector result lists.
//!
//! RRF combines ranks rather than raw scores, so BM25 and cosine values need
//! no calibration. The score of a chunk is the sum of 1 / (k + rank) over the
//! lists that contain it, with k = 60 as in the original RRF paper. Ties are
//! broken by ascending chunk id for deterministic pagination.

use std::collections::BTreeMap;

use crate::index::IndexedChunk;

/// RRF rank constant.
pub const RRF_K: f64 = 60.0;

/// One fused candidate.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Fused {
    /// Index into the indexed chunk vector.
    pub(crate) index: usize,
    /// RRF score.
    pub(crate) score: f64,
    /// One-based lexical rank, when present.
    pub(crate) lexical_rank: Option<usize>,
    /// One-based vector rank, when present.
    pub(crate) vector_rank: Option<usize>,
}

/// Fuses two ranked lists into a deterministic RRF ranking.
pub(crate) fn fuse(
    lexical: &[(usize, f64)],
    vector: &[(usize, f64)],
    chunks: &[IndexedChunk<'_>],
) -> Vec<Fused> {
    let mut fused: BTreeMap<usize, Fused> = BTreeMap::new();

    for (position, (index, _)) in lexical.iter().enumerate() {
        let rank = position + 1;
        let entry = fused.entry(*index).or_insert(Fused {
            index: *index,
            score: 0.0,
            lexical_rank: None,
            vector_rank: None,
        });
        entry.score += 1.0 / (RRF_K + rank as f64);
        entry.lexical_rank = Some(rank);
    }

    for (position, (index, _)) in vector.iter().enumerate() {
        let rank = position + 1;
        let entry = fused.entry(*index).or_insert(Fused {
            index: *index,
            score: 0.0,
            lexical_rank: None,
            vector_rank: None,
        });
        entry.score += 1.0 / (RRF_K + rank as f64);
        entry.vector_rank = Some(rank);
    }

    let mut fused: Vec<Fused> = fused.into_values().collect();
    fused.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| chunk_order(chunks, left.index, right.index))
    });
    fused
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
    use crate::storage::model::{Chunk, ChunkId, Document, DocumentId, Modality};

    fn document(id: &str) -> Document {
        Document::new(DocumentId::new(id).expect("id"), Modality::Text)
    }

    fn chunk(document: &Document, id: u64) -> Chunk {
        Chunk::new(
            ChunkId::new(id).expect("chunk id"),
            document.id().clone(),
            0,
            "text".to_owned(),
            vec![1.0, 0.0],
        )
        .expect("chunk")
    }

    #[test]
    fn fuses_ranks_with_the_standard_constant() {
        let documents = [document("a"), document("b"), document("c")];
        let chunks = [
            chunk(&documents[0], 1),
            chunk(&documents[1], 2),
            chunk(&documents[2], 3),
        ];
        let indexed_chunks: Vec<IndexedChunk<'_>> = chunks
            .iter()
            .enumerate()
            .map(|(index, chunk)| IndexedChunk {
                chunk,
                document: &documents[index],
                tokens: Vec::new(),
            })
            .collect();
        let lexical = vec![(0usize, 2.0), (1, 1.0), (2, 0.5)];
        let vector = vec![(2usize, 0.9), (0, 0.8), (1, 0.7)];
        let fused = fuse(&lexical, &vector, &indexed_chunks);
        assert_eq!(fused.len(), 3);
        assert_eq!(fused[0].index, 0);
        assert_eq!(fused[1].index, 2);
        assert_eq!(fused[2].index, 1);
        assert_eq!(fused[0].lexical_rank, Some(1));
        assert_eq!(fused[0].vector_rank, Some(2));
        let expected = 1.0 / 61.0 + 1.0 / 62.0;
        assert!((fused[0].score - expected).abs() < 1e-12);
    }

    #[test]
    fn a_single_list_preserves_its_order() {
        let documents = [document("a"), document("b")];
        let chunks = [chunk(&documents[0], 1), chunk(&documents[1], 2)];
        let indexed_chunks: Vec<IndexedChunk<'_>> = chunks
            .iter()
            .enumerate()
            .map(|(index, chunk)| IndexedChunk {
                chunk,
                document: &documents[index],
                tokens: Vec::new(),
            })
            .collect();
        let fused = fuse(&[(1, 1.0), (0, 0.5)], &[], &indexed_chunks);
        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].index, 1);
        assert_eq!(fused[0].lexical_rank, Some(1));
        assert_eq!(fused[0].vector_rank, None);
    }

    #[test]
    fn equal_scores_break_ties_by_chunk_id() {
        let documents = [document("a"), document("b")];
        let chunks = [chunk(&documents[0], 2), chunk(&documents[1], 1)];
        let indexed_chunks: Vec<IndexedChunk<'_>> = chunks
            .iter()
            .enumerate()
            .map(|(index, chunk)| IndexedChunk {
                chunk,
                document: &documents[index],
                tokens: Vec::new(),
            })
            .collect();
        let fused = fuse(&[(0, 1.0)], &[(0, 1.0)], &indexed_chunks);
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].index, 0);
    }
}
