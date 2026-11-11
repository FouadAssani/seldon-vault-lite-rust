//! Model file sources: a development directory or a parsed model bundle.
//!
//! The text, image, and audio loaders use this abstraction so the same
//! validation and decoding code runs on a directory tree or on the sections of
//! an embedded bundle. Directory reads remain bounded by the caller's cap.
//! Bundle reads borrow the section bytes directly, so loading the 1.5 GB
//! Safetensors section does not copy it before the tensor decoder reads it.

use std::borrow::Cow;
use std::path::{Path, PathBuf};

#[cfg(feature = "embedded-model")]
use std::sync::OnceLock;

use crate::binary::read_file_bounded;
use crate::bundle::Bundle;
#[cfg(not(feature = "embedded-model"))]
use crate::bundle::MAX_BUNDLE_BYTES;
use crate::error::{Error, Result};
use crate::json::Value;
use crate::model::config::MAX_CONFIG_BYTES;
use crate::model::text::TextModel;
use crate::model::tokenizer::Tokenizer;

/// A source of model files.
#[derive(Debug)]
pub enum ModelFiles<'a> {
    /// A directory on disk. Files are read with `read_file_bounded`.
    Directory(PathBuf),
    /// A parsed model bundle. Section bytes are borrowed from the bundle.
    Bundle(&'a Bundle<'a>),
}

impl<'a> ModelFiles<'a> {
    /// Builds a directory source.
    pub fn directory(path: impl Into<PathBuf>) -> Self {
        Self::Directory(path.into())
    }

    /// Builds a bundle source.
    pub fn bundle(bundle: &'a Bundle<'a>) -> Self {
        Self::Bundle(bundle)
    }

    /// Reads a file by relative path, bounded by `cap` bytes.
    ///
    /// # Errors
    ///
    /// Returns Error::Corrupt for a missing file or section, and
    /// Error::LimitExceeded when the content is larger than `cap`.
    pub fn read(&self, path: &str, cap: u64) -> Result<Cow<'a, [u8]>> {
        if !is_safe_relative(path) {
            return Err(Error::internal(format!(
                "model file path {path} is not a safe relative path"
            )));
        }
        match self {
            Self::Directory(root) => Ok(Cow::Owned(read_file_bounded(&root.join(path), cap)?)),
            Self::Bundle(bundle) => {
                let data = bundle
                    .data(path)
                    .ok_or_else(|| Error::corrupt(format!("model bundle has no section {path}")))?;
                if data.len() as u64 > cap {
                    return Err(Error::limit_exceeded(format!(
                        "bundle section {path} is {} bytes, above the {cap}-byte cap",
                        data.len()
                    )));
                }
                Ok(Cow::Borrowed(data))
            }
        }
    }

    /// Reads a UTF-8 file by relative path, bounded by `cap` bytes.
    ///
    /// # Errors
    ///
    /// Returns Error::Corrupt when the content is not valid UTF-8.
    pub fn read_utf8(&self, path: &str, cap: u64) -> Result<Cow<'a, str>> {
        let bytes = self.read(path, cap)?;
        match bytes {
            Cow::Borrowed(bytes) => std::str::from_utf8(bytes)
                .map(Cow::Borrowed)
                .map_err(|_| Error::corrupt(format!("bundle section {path} is not valid UTF-8"))),
            Cow::Owned(bytes) => {
                let text = String::from_utf8(bytes)
                    .map_err(|_| Error::corrupt(format!("{path} is not valid UTF-8")))?;
                Ok(Cow::Owned(text))
            }
        }
    }

    /// Returns the bundle section SHA-256, or None for a directory source.
    ///
    /// The digest is already recorded in the bundle header, so a loader can
    /// reuse it as the weights fingerprint instead of hashing the section a
    /// second time.
    pub fn section_sha256(&self, path: &str) -> Option<[u8; 32]> {
        match self {
            Self::Directory(_) => None,
            Self::Bundle(bundle) => bundle.section(path).map(|section| section.sha256()),
        }
    }

    /// Returns true when the relative path exists.
    pub fn contains(&self, path: &str) -> bool {
        if !is_safe_relative(path) {
            return false;
        }
        match self {
            Self::Directory(root) => root.join(path).is_file(),
            Self::Bundle(bundle) => bundle.section(path).is_some(),
        }
    }
}

/// Validates that a path is relative and normalized.
fn is_safe_relative(path: &str) -> bool {
    if path.is_empty() || path.starts_with('/') || path.contains('\\') || path.contains('\0') {
        return false;
    }
    path.split('/')
        .all(|component| !component.is_empty() && component != "." && component != "..")
}

#[cfg(feature = "embedded-model")]
/// Model bundle embedded at compile time.
///
/// The build must set `SELDON_MODEL_BUNDLE` to the absolute path of a
/// verified `.svlm` file. The bytes are included by the Rust compiler and are
/// not read from disk at runtime.
pub static EMBEDDED_MODEL_BUNDLE: &[u8] = include_bytes!(env!("SELDON_MODEL_BUNDLE"));

/// Returns the parsed and verified embedded bundle, parsing it once.
#[cfg(feature = "embedded-model")]
fn embedded_bundle() -> Result<&'static Bundle<'static>> {
    static BUNDLE: OnceLock<Bundle<'static>> = OnceLock::new();
    if let Some(bundle) = BUNDLE.get() {
        return Ok(bundle);
    }
    let bundle = Bundle::parse(EMBEDDED_MODEL_BUNDLE)?;
    bundle.verify()?;
    let _ = BUNDLE.set(bundle);
    BUNDLE
        .get()
        .ok_or_else(|| Error::internal("embedded model bundle initialization failed"))
}

/// Text model, optional tokenizer, and raw configuration loaded from one source.
#[derive(Debug)]
pub struct TextModelFiles {
    /// Loaded FP32 text model.
    pub model: TextModel,
    /// Tokenizer when the source contains tokenizer.json.
    pub tokenizer: Option<Tokenizer>,
    /// Parsed configuration object, including the vision and audio sections.
    pub config: Value,
}

/// Loads the text model, optional tokenizer, and raw configuration from a
/// directory or a bundle file.
///
/// A bundle file is read, structurally parsed, and fully verified once; the
/// returned model, tokenizer, and configuration own their data, so the bundle
/// bytes can be dropped after this call.
///
/// # Errors
///
/// Returns a typed error for a missing or corrupt model, a bundle that fails
/// structural or section verification, or an invalid tokenizer.
pub fn load_text_model_files(path: &Path) -> Result<TextModelFiles> {
    with_source(path, load_text_model_files_from)
}

fn load_text_model_files_from(files: &ModelFiles<'_>) -> Result<TextModelFiles> {
    let config_text = files.read_utf8("config.json", MAX_CONFIG_BYTES)?;
    let config = crate::json::parse(&config_text)?;
    let model = TextModel::from_files(files)?;
    let tokenizer = if files.contains("tokenizer.json") {
        Some(Tokenizer::from_files(files)?)
    } else {
        None
    };
    Ok(TextModelFiles {
        model,
        tokenizer,
        config,
    })
}

/// Loads a text embedder from a directory or a bundle file.
///
/// # Errors
///
/// Returns a typed error for a missing or corrupt model, a bundle that fails
/// structural or section verification, or an invalid tokenizer.
pub fn load_text_embedder(path: &Path) -> Result<crate::model::text::TextEmbedder> {
    with_source(path, crate::model::text::TextEmbedder::from_files)
}

/// Loads a text model from a directory or a bundle file.
///
/// # Errors
///
/// Returns a typed error for a missing or corrupt model, or a bundle that fails
/// structural or section verification.
pub fn load_text_model(path: &Path) -> Result<crate::model::text::TextModel> {
    with_source(path, crate::model::text::TextModel::from_files)
}

/// Loads an image embedder from a directory or a bundle file.
///
/// # Errors
///
/// Returns a typed error for a missing or corrupt model, or a bundle that fails
/// structural or section verification.
pub fn load_image_embedder(path: &Path) -> Result<crate::model::image::ImageEmbedder> {
    with_source(path, crate::model::image::ImageEmbedder::from_files)
}

/// Loads an audio embedder from a directory or a bundle file.
///
/// # Errors
///
/// Returns a typed error for a missing or corrupt model, or a bundle that fails
/// structural or section verification.
pub fn load_audio_embedder(path: &Path) -> Result<crate::model::audio_embedder::AudioEmbedder> {
    with_source(
        path,
        crate::model::audio_embedder::AudioEmbedder::from_files,
    )
}

/// Loads from the embedded bundle when the feature is enabled, or from a
/// directory or bundle file otherwise.
fn with_source<T>(path: &Path, load: impl FnOnce(&ModelFiles<'_>) -> Result<T>) -> Result<T> {
    #[cfg(feature = "embedded-model")]
    let result = {
        let _ = path;
        let bundle = embedded_bundle()?;
        load(&ModelFiles::bundle(bundle))
    };
    #[cfg(not(feature = "embedded-model"))]
    let result = if path.is_file() {
        with_bundle(path, load)
    } else {
        load(&ModelFiles::directory(path))
    };
    result
}

/// Reads a bundle file, parses it, verifies every section, and runs `load`.
#[cfg(not(feature = "embedded-model"))]
fn with_bundle<T>(path: &Path, load: impl FnOnce(&ModelFiles<'_>) -> Result<T>) -> Result<T> {
    let bytes = read_file_bounded(path, MAX_BUNDLE_BYTES)?;
    let bundle = Bundle::parse(&bytes)?;
    bundle.verify()?;
    load(&ModelFiles::bundle(&bundle))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_paths_are_validated() {
        assert!(is_safe_relative("config.json"));
        assert!(is_safe_relative("1_Pooling/config.json"));
        assert!(!is_safe_relative(""));
        assert!(!is_safe_relative("/config.json"));
        assert!(!is_safe_relative("a//b"));
        assert!(!is_safe_relative("a/../b"));
        assert!(!is_safe_relative("a\\b"));
        assert!(!is_safe_relative("a\0b"));
    }
}
