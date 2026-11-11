//! Model assets and inference for the frozen EmbeddingGemma 2 model.
//!
//! The module exposes the strict text and vision configurations, the validated
//! FP32 weight loaders, the exact tokenizer, the official prompt prefixes, the
//! text encoder, the vision tower, and the image embedder that connects the
//! two. Audio remains unimplemented at milestone M4 and is not declared
//! supported.

pub mod config;
pub mod image;
pub mod prompt;
pub mod text;
pub mod tokenizer;
pub mod vision;
pub mod weights;

pub use config::{LayerConfig, LayerType, TextConfig};
pub use image::{ImageEmbedder, ImageEmbeddingOutput, ImageSpecialTokens};
pub use text::{ForwardOutput, TextEmbedder, TextModel};
pub use vision::{VisionConfig, VisionModel, VisionOutput, VisionWeights};
pub use weights::TextWeights;
