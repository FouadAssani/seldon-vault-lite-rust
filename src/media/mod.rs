//! Native media decoders and preprocessing for multimodal embeddings.
//!
//! Milestone M4 implements the image path. This module exposes the bounded
//! zlib/DEFLATE decoder used by PNG, the strict PNG and baseline JPEG
//! decoders, a shared decoded-image type, and the official image
//! preprocessing that produces vision patches and positions. The vision
//! encoder itself lives in the model module. No external process or image
//! library is used.

pub mod deflate;
pub mod image;
pub mod jpeg;
pub mod png;
pub mod preprocess;

pub use image::Image;
pub use preprocess::{ImagePatches, PreprocessOptions};
