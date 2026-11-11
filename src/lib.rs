//! Foundations of Seldon Vault Lite.
//!
//! The crate exposes the CLI, the central error type, binary format readers,
//! strict JSON, the FP32 tensor core, the frozen model tokenizer, and the
//! milestone M3 persistent store. Inference, media, and the HTTP API will be
//! added over milestones M4 to M9; no empty module is created in advance.
//!
//! Project contract: no third-party crate, only std and core::arch
//! intrinsics.

#![warn(missing_docs)]

pub mod binary;
pub mod cli;
pub mod error;
pub mod index;
pub mod json;
pub mod model;
pub mod storage;
pub mod tensor;

/// Crate version, from Cargo.toml.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Program name displayed by the CLI.
///
/// It is derived from the package name so that help and the version line
/// cannot diverge if the package is renamed.
pub const PROGRAM_NAME: &str = env!("CARGO_PKG_NAME");

/// Frozen model revision.
///
/// The value is the Hugging Face commit recorded in
/// fixtures/model/manifest.json. It is exposed so the CLI and tests do not
/// duplicate the revision string.
pub const MODEL_REVISION: &str = "f9972ac22e696a105c482202646979e33e4b5a1e";

/// Development model cache directory, ignored by Git.
pub const MODEL_CACHE_DIR: &str = ".model-cache";

/// Build target triple.
///
/// Phase 1 only validates aarch64-apple-darwin. Other targets return
/// "unsupported" until their port has been validated; this avoids announcing
/// compatibility that has not been tested.
pub const TARGET_TRIPLE: &str = if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
    "aarch64-apple-darwin"
} else {
    "unsupported"
};
