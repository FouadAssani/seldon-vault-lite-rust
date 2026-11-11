//! Native media decoders and preprocessing for multimodal embeddings.
//!
//! Milestone M4 implements the image path. This module exposes the bounded
//! zlib/DEFLATE decoder used by PNG, the strict PNG and baseline JPEG
//! decoders, and a shared decoded-image type. Official image preprocessing
//! and the vision encoder are not implemented yet, so no image embedding is
//! declared supported. No external process or image library is used.

pub mod deflate;
pub mod image;
pub mod jpeg;
pub mod png;

pub use image::Image;
