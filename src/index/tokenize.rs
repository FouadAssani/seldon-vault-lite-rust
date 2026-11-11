//! Deterministic tokenization for lexical search.
//!
//! BM25 needs reproducible terms, not the model tokenizer. This module splits
//! text into Unicode alphanumeric runs, lowercases each scalar with
//! char::to_lowercase, and caps the input size, token count, and token length.
//! Punctuation and whitespace are separators; no stemming or stop-word list is
//! applied, so the index remains predictable.

use crate::error::{Error, Result};

/// Maximum text size accepted by tokenization.
pub const MAX_TOKENIZE_BYTES: usize = 1024 * 1024;
/// Maximum number of tokens accepted in one chunk.
pub const MAX_INDEX_TOKENS_PER_CHUNK: usize = 8192;
/// Maximum number of tokens accepted in one query.
pub const MAX_QUERY_TOKENS: usize = 1024;
/// Maximum size of one token in bytes.
pub const MAX_TOKEN_BYTES: usize = 64;

/// Tokenizes text into lowercase alphanumeric runs.
///
/// The caller chooses the token cap: chunks use MAX_INDEX_TOKENS_PER_CHUNK and
/// queries use MAX_QUERY_TOKENS. Exceeding the cap is an explicit limit error,
/// never a silent truncation.
pub fn tokenize(text: &str, max_tokens: usize) -> Result<Vec<String>> {
    if text.len() > MAX_TOKENIZE_BYTES {
        return Err(Error::limit_exceeded(format!(
            "text is {} bytes, above the {MAX_TOKENIZE_BYTES}-byte tokenization cap",
            text.len()
        )));
    }

    let mut tokens = Vec::new();
    let mut current = String::new();

    for character in text.chars() {
        if character.is_alphanumeric() {
            for lowercase in character.to_lowercase() {
                if current.len() + lowercase.len_utf8() > MAX_TOKEN_BYTES {
                    return Err(Error::limit_exceeded(format!(
                        "token exceeds the {MAX_TOKEN_BYTES}-byte cap"
                    )));
                }
                current.push(lowercase);
            }
        } else if !current.is_empty() {
            push_token(&mut tokens, &mut current, max_tokens)?;
        }
    }
    if !current.is_empty() {
        push_token(&mut tokens, &mut current, max_tokens)?;
    }

    Ok(tokens)
}

fn push_token(tokens: &mut Vec<String>, current: &mut String, max_tokens: usize) -> Result<()> {
    if tokens.len() >= max_tokens {
        return Err(Error::limit_exceeded(format!(
            "token count exceeds the {max_tokens}-token cap"
        )));
    }
    tokens.push(std::mem::take(current));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_punctuation_and_lowercases() {
        let tokens = tokenize("Hello, WORLD! café", MAX_QUERY_TOKENS).expect("tokenize");
        assert_eq!(tokens, vec!["hello", "world", "café"]);
    }

    #[test]
    fn keeps_unicode_alphanumeric_runs() {
        let tokens = tokenize("naïve 日本語 123", MAX_QUERY_TOKENS).expect("tokenize");
        assert_eq!(tokens, vec!["naïve", "日本語", "123"]);
    }

    #[test]
    fn empty_and_separator_only_text_yield_no_tokens() {
        for text in ["", "   ", "!!! ---", "\n\t"] {
            assert!(
                tokenize(text, MAX_QUERY_TOKENS)
                    .expect("tokenize")
                    .is_empty()
            );
        }
    }

    #[test]
    fn enforces_the_token_count_cap() {
        let text = "a ".repeat(MAX_QUERY_TOKENS + 1);
        let error = tokenize(&text, MAX_QUERY_TOKENS).expect_err("token cap");
        assert!(matches!(error, Error::LimitExceeded(_)));
    }

    #[test]
    fn enforces_the_input_size_cap() {
        let text = "a".repeat(MAX_TOKENIZE_BYTES + 1);
        let error = tokenize(&text, MAX_QUERY_TOKENS).expect_err("input cap");
        assert!(matches!(error, Error::LimitExceeded(_)));
    }

    #[test]
    fn enforces_the_token_length_cap() {
        let text = "a".repeat(MAX_TOKEN_BYTES + 1);
        let error = tokenize(&text, MAX_QUERY_TOKENS).expect_err("token length cap");
        assert!(matches!(error, Error::LimitExceeded(_)));
    }
}
