//! Model assets: the frozen tokenizer and the official prompt prefixes.
//!
//! This module does not run inference yet. It provides the exact text input
//! contract: token ids must match the official tokenizer, and prompts must be
//! applied with the correct role.

pub mod prompt;
pub mod tokenizer;
