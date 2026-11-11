//! Model loading and embedding dispatch for the HTTP API.
//!
//! The text model is loaded once at startup, which gives the server an honest
//! readiness signal and the model's embedding dimension before it accepts a
//! request. The tokenizer is loaded only when `tokenizer.json` exists; the tiny
//! diagnostic fixtures intentionally have no tokenizer and therefore support
//! only the explicit token-id path.
//!
//! The vision and audio towers are large. They are loaded lazily on the first
//! request for their modality, behind a mutex, and reused afterwards. The
//! readiness response reports which modalities are declared by the frozen
//! configuration and which have already been loaded. This keeps a text-only
//! deployment from allocating media weights it will never use while still
//! refusing a media request when the model does not declare that modality.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::binary::reader::read_file_bounded;
use crate::error::{Error, Result};
use crate::json;
use crate::media::preprocess::PreprocessOptions;
use crate::media::{self, Audio};
use crate::model::audio::AudioConfig;
use crate::model::config::{MAX_CONFIG_BYTES, MAX_TEXT_TOKENS, TextConfig};
use crate::model::prompt::{self, Prompt};
use crate::model::text::TextModel;
use crate::model::tokenizer::Tokenizer;
use crate::model::vision::VisionConfig;
use crate::model::{AudioEmbedder, ImageEmbedder};
use crate::{MODEL_REVISION, VERSION};

/// Media format detected from a byte prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    /// PNG signature.
    Png,
    /// JPEG SOI marker.
    Jpeg,
}

/// Loaded model set and embedding entry points.
#[derive(Debug)]
pub struct Engine {
    model_dir: PathBuf,
    model: TextModel,
    tokenizer: Option<Tokenizer>,
    image: Mutex<Option<ImageEmbedder>>,
    audio: Mutex<Option<AudioEmbedder>>,
    image_available: bool,
    audio_available: bool,
    text_input_available: bool,
    embedding_dimension: usize,
    matryoshka_dimensions: Vec<usize>,
}

impl Engine {
    /// Loads the text model and validates the frozen configuration.
    ///
    /// The vision and audio configurations are validated when declared, but
    /// their towers are loaded lazily. A missing tokenizer is not an error: the
    /// model can still serve token-id embeddings, which is the reproducible
    /// diagnostic path used by the tiny fixtures.
    ///
    /// # Errors
    ///
    /// Returns a typed error for an invalid or oversized configuration, a
    /// missing or inconsistent weight tensor, an invalid tokenizer, or a
    /// declared media configuration that cannot be parsed.
    pub fn load(model_dir: &Path, vision_budget: usize) -> Result<Self> {
        let config_bytes = read_file_bounded(&model_dir.join("config.json"), MAX_CONFIG_BYTES)?;
        let config_text = std::str::from_utf8(&config_bytes)
            .map_err(|_| Error::corrupt("config.json is not valid UTF-8"))?;
        let config_value = json::parse(config_text)?;
        let text_config = TextConfig::from_json_value(&config_value)?;
        let image_available = config_value.get("vision_config").is_some();
        if image_available {
            VisionConfig::from_json_value(&config_value)?;
        }
        let audio_available = config_value.get("audio_config").is_some();
        if audio_available {
            AudioConfig::from_json_value(&config_value)?;
        }
        PreprocessOptions::new(vision_budget)?;

        let model = TextModel::load(model_dir)?;
        let embedding_dimension = model.config().embedding_dim();
        if text_config.embedding_dim() != embedding_dimension {
            return Err(Error::corrupt(format!(
                "text configuration declares embedding dimension {}, loaded model reports {}",
                text_config.embedding_dim(),
                embedding_dimension
            )));
        }

        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = match std::fs::metadata(&tokenizer_path) {
            Ok(metadata) if metadata.is_file() => {
                let tokenizer = Tokenizer::load(&tokenizer_path)?;
                if tokenizer.vocab_size() != model.config().vocab_size() {
                    return Err(Error::corrupt(format!(
                        "tokenizer vocabulary has {} entries, model declares {}",
                        tokenizer.vocab_size(),
                        model.config().vocab_size()
                    )));
                }
                Some(tokenizer)
            }
            Ok(_) => None,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(Error::Io(error)),
        };
        let text_input_available = tokenizer.is_some();
        let matryoshka_dimensions = [768usize, 512, 256, 128]
            .into_iter()
            .filter(|dimension| *dimension <= embedding_dimension)
            .collect();

        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            model,
            tokenizer,
            image: Mutex::new(None),
            audio: Mutex::new(None),
            image_available,
            audio_available,
            text_input_available,
            embedding_dimension,
            matryoshka_dimensions,
        })
    }

    /// Model identifier frozen by the project.
    pub fn model_id(&self) -> &'static str {
        "google/embeddinggemma-2"
    }

    /// Frozen model revision.
    pub fn revision(&self) -> &'static str {
        MODEL_REVISION
    }

    /// Crate version reported by the API.
    pub fn version(&self) -> &'static str {
        VERSION
    }

    /// Model embedding dimension.
    pub fn embedding_dimension(&self) -> usize {
        self.embedding_dimension
    }

    /// Official Matryoshka dimensions supported by this model.
    pub fn matryoshka_dimensions(&self) -> &[usize] {
        &self.matryoshka_dimensions
    }

    /// Returns true when a tokenizer is present and text input is available.
    pub fn text_input_available(&self) -> bool {
        self.text_input_available
    }

    /// Returns true when the frozen configuration declares a vision tower.
    pub fn image_available(&self) -> bool {
        self.image_available
    }

    /// Returns true when the frozen configuration declares an audio tower.
    pub fn audio_available(&self) -> bool {
        self.audio_available
    }

    /// Returns true when the vision tower has been loaded.
    ///
    /// # Errors
    ///
    /// Returns an internal error if the image model lock is poisoned.
    pub fn image_loaded(&self) -> Result<bool> {
        self.image
            .lock()
            .map(|guard| guard.is_some())
            .map_err(|_| Error::internal("image model lock is poisoned"))
    }

    /// Returns true when the audio tower has been loaded.
    ///
    /// # Errors
    ///
    /// Returns an internal error if the audio model lock is poisoned.
    pub fn audio_loaded(&self) -> Result<bool> {
        self.audio
            .lock()
            .map(|guard| guard.is_some())
            .map_err(|_| Error::internal("audio model lock is poisoned"))
    }

    /// Modalities declared by the frozen model.
    pub fn available_modalities(&self) -> Vec<&'static str> {
        let mut modalities = vec!["text"];
        if self.image_available {
            modalities.push("image");
        }
        if self.audio_available {
            modalities.push("audio");
        }
        modalities
    }

    /// Validates a requested Matryoshka dimension.
    ///
    /// # Errors
    ///
    /// Returns a usage error when the dimension is zero or above the model
    /// embedding dimension.
    pub fn validate_dimension(&self, dimension: usize) -> Result<()> {
        if dimension == 0 || dimension > self.embedding_dimension {
            return Err(Error::usage(format!(
                "dimension {dimension} is outside 1..={}",
                self.embedding_dimension
            )));
        }
        Ok(())
    }

    /// Embeds explicit token ids without tokenization or a prompt prefix.
    ///
    /// # Errors
    ///
    /// Returns a usage error for an empty or oversized id list, an invalid
    /// dimension, or an out-of-vocabulary id, and an internal error for a model
    /// invariant violation.
    pub fn embed_tokens(&self, ids: &[u32], dimension: usize) -> Result<Vec<f32>> {
        self.validate_dimension(dimension)?;
        if ids.is_empty() {
            return Err(Error::usage("token id input must not be empty"));
        }
        if ids.len() > MAX_TEXT_TOKENS {
            return Err(Error::limit_exceeded(format!(
                "token id input has {} ids, above the {MAX_TEXT_TOKENS}-token context",
                ids.len()
            )));
        }
        self.model.embed_ids(ids, dimension)
    }

    /// Embeds raw text with the official prompt prefix.
    ///
    /// # Errors
    ///
    /// Returns an unsupported error when no tokenizer is present, a usage error
    /// for an invalid dimension or prompt, or a limit error when the tokenized
    /// input exceeds the model context.
    pub fn embed_text(&self, text: &str, prompt: Prompt, dimension: usize) -> Result<Vec<f32>> {
        self.validate_dimension(dimension)?;
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            Error::unsupported(
                "this model directory has no tokenizer.json; use the token-id input type",
            )
        })?;
        let full = prompt::apply(prompt, text);
        let ids = tokenizer.encode(&full, true)?;
        if ids.len() > MAX_TEXT_TOKENS {
            return Err(Error::limit_exceeded(format!(
                "tokenized input has {} tokens, above the {MAX_TEXT_TOKENS}-token context",
                ids.len()
            )));
        }
        self.model.embed_ids(&ids, dimension)
    }

    /// Detects a supported image format from the leading bytes.
    ///
    /// # Errors
    ///
    /// Returns a corruption error when the bytes are neither PNG nor JPEG.
    pub fn detect_image_format(bytes: &[u8]) -> Result<ImageFormat> {
        if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
            Ok(ImageFormat::Png)
        } else if bytes.len() >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF {
            Ok(ImageFormat::Jpeg)
        } else {
            Err(Error::corrupt(
                "media bytes are neither a PNG nor a JPEG image",
            ))
        }
    }

    /// Decodes and embeds one image.
    ///
    /// The vision tower is loaded on first use. The official preprocessing and
    /// visual token budget are applied before the tower, so the result is a
    /// real image embedding rather than a filename or caption vector.
    ///
    /// # Errors
    ///
    /// Returns an unsupported error when the model declares no vision tower, a
    /// corruption error for an invalid image, a usage error for an invalid
    /// dimension or budget, or a model error for an invariant violation.
    pub fn embed_image(
        &self,
        bytes: &[u8],
        format: ImageFormat,
        dimension: usize,
        vision_budget: usize,
    ) -> Result<Vec<f32>> {
        if !self.image_available {
            return Err(Error::unsupported(
                "this model declares no vision_config; image embeddings are unavailable",
            ));
        }
        self.validate_dimension(dimension)?;
        let options = PreprocessOptions::new(vision_budget)?;
        let image = match format {
            ImageFormat::Png => media::png::decode(bytes)?,
            ImageFormat::Jpeg => media::jpeg::decode(bytes)?,
        };
        let mut guard = self
            .image
            .lock()
            .map_err(|_| Error::internal("image model lock is poisoned"))?;
        if guard.is_none() {
            *guard = Some(ImageEmbedder::load(&self.model_dir)?);
        }
        guard
            .as_ref()
            .ok_or_else(|| Error::internal("image embedder was not loaded"))?
            .embed(&image, dimension, options)
    }

    /// Decodes and embeds one WAV clip.
    ///
    /// The audio tower is loaded on first use. The native decoder converts to
    /// mono, resamples to 16 kHz, and truncates at the official 30-second cap
    /// before the log-mel processor and the tower run.
    ///
    /// # Errors
    ///
    /// Returns an unsupported error when the model declares no audio tower, a
    /// corruption error for invalid WAV data, a usage error for an invalid
    /// dimension, or a model error for an invariant violation.
    pub fn embed_audio(&self, bytes: &[u8], dimension: usize) -> Result<Vec<f32>> {
        if !self.audio_available {
            return Err(Error::unsupported(
                "this model declares no audio_config; audio embeddings are unavailable",
            ));
        }
        self.validate_dimension(dimension)?;
        let audio: Audio = media::wav::decode_bytes(bytes)?;
        let mut guard = self
            .audio
            .lock()
            .map_err(|_| Error::internal("audio model lock is poisoned"))?;
        if guard.is_none() {
            *guard = Some(AudioEmbedder::load(&self.model_dir)?);
        }
        guard
            .as_ref()
            .ok_or_else(|| Error::internal("audio embedder was not loaded"))?
            .embed(&audio, dimension)
    }
}
