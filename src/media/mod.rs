//! Native media decoders and preprocessing for multimodal embeddings.
//!
//! Milestone M4 implements the image path: the bounded zlib/DEFLATE decoder
//! used by PNG, the strict PNG and baseline JPEG decoders, a shared
//! decoded-image type, and the official image preprocessing that produces
//! vision patches and positions. The vision encoder itself lives in the model
//! module.
//!
//! Milestone M5 adds the audio path: a strict RIFF/WAVE decoder for PCM and
//! IEEE-float samples, mono conversion, deterministic windowed-sinc
//! resampling, and the official log-mel spectrogram. The audio encoder lives
//! in the model module. No external process, image library, or audio library
//! is used.

pub mod audio;
pub mod audio_preprocess;
pub mod deflate;
pub mod image;
pub mod jpeg;
pub mod png;
pub mod preprocess;
pub mod wav;

pub use audio::{Audio, MAX_AUDIO_SAMPLES, TARGET_SAMPLE_RATE};
pub use audio_preprocess::{MelSpectrogram, log_mel};
pub use image::Image;
pub use preprocess::{ImagePatches, PreprocessOptions};
