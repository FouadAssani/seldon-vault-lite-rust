//! Official EmbeddingGemma 2 prompt prefixes.
//!
//! The prefixes are copied from the frozen model's
//! config_sentence_transformers.json. A query and an indexed document are
//! distinct roles: the query prefix must never be applied to a document, and
//! the document prefix must never be applied to a query. Using the wrong
//! prefix silently changes the embedding space.
//!
//! Document titles are part of the official document prompt. The current
//! constant uses the model's default title placeholder; title interpolation
//! will be added with the document ingestion milestone.

/// Prefix for a search query.
pub const SEARCH_QUERY: &str = "task: search result | query: ";

/// Prefix for an indexed document.
pub const DOCUMENT: &str = "title: none | text: ";

/// Prefix for a classification task.
pub const CLASSIFICATION: &str = "task: classification | query: ";

/// Prefix for a clustering task.
pub const CLUSTERING: &str = "task: clustering | query: ";

/// Prefix for a code retrieval task.
pub const CODE_RETRIEVAL: &str = "task: code retrieval | query: ";

/// Prefix for a fact-checking task.
pub const FACT_CHECKING: &str = "task: fact checking | query: ";

/// Prefix for a sentence-similarity task.
pub const SENTENCE_SIMILARITY: &str = "task: sentence similarity | query: ";

/// Prefix for a question-answering task.
pub const QUESTION_ANSWERING: &str = "task: question answering | query: ";

/// A prompt role from the official configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prompt {
    /// Search query.
    SearchQuery,
    /// Indexed document.
    Document,
    /// Classification.
    Classification,
    /// Clustering.
    Clustering,
    /// Code retrieval.
    CodeRetrieval,
    /// Fact checking.
    FactChecking,
    /// Sentence similarity.
    SentenceSimilarity,
    /// Question answering.
    QuestionAnswering,
}

impl Prompt {
    /// Returns the official prefix for this role.
    pub const fn prefix(self) -> &'static str {
        match self {
            Self::SearchQuery => SEARCH_QUERY,
            Self::Document => DOCUMENT,
            Self::Classification => CLASSIFICATION,
            Self::Clustering => CLUSTERING,
            Self::CodeRetrieval => CODE_RETRIEVAL,
            Self::FactChecking => FACT_CHECKING,
            Self::SentenceSimilarity => SENTENCE_SIMILARITY,
            Self::QuestionAnswering => QUESTION_ANSWERING,
        }
    }

    /// Returns true for the query role.
    pub const fn is_query(self) -> bool {
        !matches!(self, Self::Document)
    }
}

/// Prepends the prompt prefix to text.
pub fn apply(prompt: Prompt, text: &str) -> String {
    let mut output = String::with_capacity(prompt.prefix().len() + text.len());
    output.push_str(prompt.prefix());
    output.push_str(text);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefixes_match_the_official_configuration() {
        assert_eq!(SEARCH_QUERY, "task: search result | query: ");
        assert_eq!(DOCUMENT, "title: none | text: ");
        assert_eq!(CLASSIFICATION, "task: classification | query: ");
        assert_eq!(CLUSTERING, "task: clustering | query: ");
        assert_eq!(CODE_RETRIEVAL, "task: code retrieval | query: ");
        assert_eq!(FACT_CHECKING, "task: fact checking | query: ");
        assert_eq!(SENTENCE_SIMILARITY, "task: sentence similarity | query: ");
        assert_eq!(QUESTION_ANSWERING, "task: question answering | query: ");
    }

    #[test]
    fn prompt_roles_map_to_prefixes() {
        assert_eq!(Prompt::SearchQuery.prefix(), SEARCH_QUERY);
        assert_eq!(Prompt::Document.prefix(), DOCUMENT);
        assert!(Prompt::SearchQuery.is_query());
        assert!(!Prompt::Document.is_query());
    }

    #[test]
    fn apply_preserves_the_text() {
        assert_eq!(
            apply(Prompt::SearchQuery, "hello"),
            "task: search result | query: hello"
        );
        assert_eq!(
            apply(Prompt::Document, "hello"),
            "title: none | text: hello"
        );
    }
}
