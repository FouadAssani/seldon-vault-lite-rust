//! Model assets and inference for the frozen EmbeddingGemma 2 model.
//!
//! The module exposes the strict text, vision, and audio configurations, the
//! validated FP32 weight loaders, the exact tokenizer, the official prompt
//! prefixes, the text encoder, the vision tower, the audio tower, and the
//! image and audio embedders that connect them to the text encoder.

pub mod audio;
pub mod audio_embedder;
pub mod config;
pub mod image;
pub mod prompt;
pub mod text;
pub mod tokenizer;
pub mod vision;
pub mod weights;

pub use audio::{AudioConfig, AudioModel, AudioOutput, AudioWeights};
pub use audio_embedder::{AudioEmbedder, AudioEmbeddingOutput, AudioSpecialTokens};
pub use config::{LayerConfig, LayerType, TextConfig};
pub use image::{ImageEmbedder, ImageEmbeddingOutput, ImageSpecialTokens};
pub use text::{ForwardOutput, TextEmbedder, TextModel};
pub use vision::{VisionConfig, VisionModel, VisionOutput, VisionWeights};
pub use weights::TextWeights;
