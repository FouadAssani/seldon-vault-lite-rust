//! BM25 inverted index over the stored chunks.
//!
//! The index is deterministic: postings are ordered by chunk id and terms are
//! held in a BTreeMap. Scores use the standard BM25 formula with k1 = 1.2 and
//! b = 0.75; no stemming, stop words, or positional scoring is applied at M3.
//! Filters are evaluated before a posting contributes to a score.

use std::collections::{BTreeMap, BTreeSet};

use crate::index::{Filters, IndexedChunk};

/// BM25 term-frequency saturation parameter.
pub const BM25_K1: f64 = 1.2;
/// BM25 document-length normalization parameter.
pub const BM25_B: f64 = 0.75;

#[derive(Debug)]
struct Posting {
    chunk: usize,
    frequency: u32,
}

/// In-memory BM25 index.
#[derive(Debug)]
pub(crate) struct LexicalIndex {
    postings: BTreeMap<String, Vec<Posting>>,
    lengths: Vec<usize>,
    average_length: f64,
}

impl LexicalIndex {
    /// Builds postings and document lengths from tokenized chunks.
    pub(crate) fn build(chunks: &[IndexedChunk<'_>]) -> Self {
        let mut postings: BTreeMap<String, Vec<Posting>> = BTreeMap::new();
        let mut lengths = Vec::with_capacity(chunks.len());
        let mut total_tokens = 0usize;

        for (index, indexed) in chunks.iter().enumerate() {
            lengths.push(indexed.tokens.len());
            total_tokens = total_tokens.saturating_add(indexed.tokens.len());

            let mut frequencies: BTreeMap<&str, u32> = BTreeMap::new();
            for token in &indexed.tokens {
                *frequencies.entry(token.as_str()).or_insert(0) += 1;
            }
            for (term, frequency) in frequencies {
                postings.entry(term.to_owned()).or_default().push(Posting {
                    chunk: index,
                    frequency,
                });
            }
        }

        let average_length = if chunks.is_empty() {
            0.0
        } else {
            total_tokens as f64 / chunks.len() as f64
        };

        Self {
            postings,
            lengths,
            average_length,
        }
    }

    /// Ranks chunks for a query, returning (chunk index, BM25 score).
    ///
    /// The result is sorted by descending score then ascending chunk id, which
    /// makes ties independent of map iteration or insertion order.
    pub(crate) fn rank(
        &self,
        query: &[String],
        chunks: &[IndexedChunk<'_>],
        filters: &Filters,
    ) -> Vec<(usize, f64)> {
        if chunks.is_empty() || query.is_empty() || self.average_length <= 0.0 {
            return Vec::new();
        }

        let document_count = chunks.len() as f64;
        let mut scores = vec![0.0f64; chunks.len()];
        let mut seen: BTreeSet<&str> = BTreeSet::new();

        for term in query {
            if !seen.insert(term.as_str()) {
                continue;
            }
            let Some(postings) = self.postings.get(term) else {
                continue;
            };
            let document_frequency = postings.len() as f64;
            let idf = (1.0
                + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
                .ln();

            for posting in postings {
                let indexed = &chunks[posting.chunk];
                if !filters.matches(indexed.document) {
                    continue;
                }
                let document_length = self.lengths[posting.chunk] as f64;
                let frequency = f64::from(posting.frequency);
                let denominator = frequency
                    + BM25_K1 * (1.0 - BM25_B + BM25_B * document_length / self.average_length);
                scores[posting.chunk] += idf * frequency * (BM25_K1 + 1.0) / denominator;
            }
        }

        let mut ranked: Vec<(usize, f64)> = scores
            .into_iter()
            .enumerate()
            .filter(|(_, score)| *score > 0.0)
            .collect();
        ranked.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| chunk_order(chunks, left.0, right.0))
        });
        ranked
    }
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

    fn chunk(document: &Document, id: u64, text: &str) -> Chunk {
        Chunk::new(
            ChunkId::new(id).expect("chunk id"),
            document.id().clone(),
            0,
            text.to_owned(),
            vec![1.0, 0.0],
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
                tokens: crate::index::tokenize::tokenize(chunk.text(), 1024).expect("tokenize"),
            })
            .collect()
    }

    #[test]
    fn ranks_by_term_frequency() {
        let documents = vec![document("a"), document("b"), document("c")];
        let chunks = vec![
            chunk(&documents[0], 1, "rust language systems"),
            chunk(&documents[1], 2, "python language scripting"),
            chunk(&documents[2], 3, "rust rust rust"),
        ];
        let indexed_chunks = indexed(&documents, &chunks);
        let index = LexicalIndex::build(&indexed_chunks);
        let query = vec!["rust".to_owned()];
        let ranked = index.rank(&query, &indexed_chunks, &Filters::new());
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].0, 2);
        assert_eq!(ranked[1].0, 0);
        assert!(ranked[0].1 > ranked[1].1);
    }

    #[test]
    fn equal_scores_break_ties_by_chunk_id() {
        let documents = vec![document("a"), document("b")];
        let chunks = vec![
            chunk(&documents[0], 2, "language"),
            chunk(&documents[1], 1, "language"),
        ];
        let indexed_chunks = indexed(&documents, &chunks);
        let index = LexicalIndex::build(&indexed_chunks);
        let ranked = index.rank(&["language".to_owned()], &indexed_chunks, &Filters::new());
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].0, 1);
        assert_eq!(ranked[1].0, 0);
        assert_eq!(ranked[0].1, ranked[1].1);
    }

    #[test]
    fn filters_exclude_documents_before_scoring() {
        let mut metadata = Metadata::new();
        metadata.insert("lang", "fr").expect("metadata");
        let documents = vec![
            document("a"),
            Document::new(DocumentId::new("b").expect("id"), Modality::Text)
                .with_metadata(metadata),
        ];
        let chunks = vec![
            chunk(&documents[0], 1, "rust"),
            chunk(&documents[1], 2, "rust"),
        ];
        let indexed_chunks = indexed(&documents, &chunks);
        let index = LexicalIndex::build(&indexed_chunks);
        let mut filters = Filters::new();
        filters.add_metadata("lang", "fr").expect("filter metadata");
        let ranked = index.rank(&["rust".to_owned()], &indexed_chunks, &filters);
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].0, 1);
    }

    #[test]
    fn unknown_terms_and_empty_queries_return_nothing() {
        let documents = vec![document("a")];
        let chunks = vec![chunk(&documents[0], 1, "rust")];
        let indexed_chunks = indexed(&documents, &chunks);
        let index = LexicalIndex::build(&indexed_chunks);
        assert!(
            index
                .rank(&["python".to_owned()], &indexed_chunks, &Filters::new())
                .is_empty()
        );
        assert!(index.rank(&[], &indexed_chunks, &Filters::new()).is_empty());
    }
}
