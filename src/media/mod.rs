//! Native media decoders and preprocessing for multimodal embeddings.
//!
//! Milestone M4 starts with the image path. This module currently exposes the
//! bounded zlib/DEFLATE decoder used by PNG. The PNG parser and official image
//! preprocessing will be added as the image path progresses. No external
//! process or image library is used.

pub mod deflate;
pub mod png;
