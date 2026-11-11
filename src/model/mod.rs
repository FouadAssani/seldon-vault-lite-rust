//! Model assets and text inference for the frozen EmbeddingGemma 2 model.
//!
//! The module exposes the strict text configuration, the validated FP32 weight
//! loader, the exact tokenizer, the official prompt prefixes, and the text
//! encoder. Image and audio encoders are not implemented at milestone M2 and
//! are not declared supported.

pub mod config;
pub mod prompt;
pub mod text;
pub mod tokenizer;
pub mod weights;

pub use config::{LayerConfig, LayerType, TextConfig};
pub use text::{ForwardOutput, TextEmbedder, TextModel};
pub use weights::TextWeights;
