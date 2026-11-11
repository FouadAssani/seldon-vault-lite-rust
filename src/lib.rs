//! Foundations of Seldon Vault Lite.
//!
//! The crate exposes the CLI and the central error type. The inference,
//! storage, and media modules will be added over milestones M1 to M9; no
//! empty module is created in advance.
//!
//! Project contract: no third-party crate, only `std` and `core::arch`
//! intrinsics.

#![warn(missing_docs)]

pub mod cli;
pub mod error;

/// Crate version, from `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Program name displayed by the CLI.
///
/// It is derived from the package name so that help and the version line
/// cannot diverge if the package is renamed.
pub const PROGRAM_NAME: &str = env!("CARGO_PKG_NAME");

/// Build target triple.
///
/// Phase 1 only validates `aarch64-apple-darwin`. Other targets return
/// `"unsupported"` until their port has been validated; this avoids
/// announcing compatibility that has not been tested.
pub const TARGET_TRIPLE: &str = if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
    "aarch64-apple-darwin"
} else {
    "unsupported"
};
