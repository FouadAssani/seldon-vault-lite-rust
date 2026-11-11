//! Audio embedding path for EmbeddingGemma 2.
//!
//! The audio embedder connects the native WAV decoder, mono conversion and
//! resampling, the official log-mel preprocessing, the audio tower, and the
//! text encoder. An audio-only input becomes the token layout
//! <bos><|audio><|audio|>*soft_tokens<audio|><eos>; the projected audio soft
//! tokens replace the placeholder embeddings, and the text encoder produces the
//! final 768-dimensional vector. Mean pooling covers every token, including the
//! begin/end audio markers and the text special tokens, exactly as the
//! reference model does.
//!
//! The product never falls back to a file name, a transcription, or a textual
//! marker: a failure anywhere in the path is returned as a typed error.

use std::path::Path;

use crate::binary::sha256::hex_digest;
use crate::error::{Error, Result};
use crate::json;
use crate::media::Audio;
use crate::media::audio_preprocess::log_mel;
use crate::model::audio::{AudioConfig, AudioModel, AudioOutput, AudioWeights};
use crate::model::config::{MAX_CONFIG_BYTES, MAX_TEXT_TOKENS, TextConfig};
use crate::model::prompt::{self, Prompt};
use crate::model::source::ModelFiles;
use crate::model::text::{ForwardOutput, TextModel};
use crate::model::tokenizer::Tokenizer;
use crate::model::weights::{MAX_MODEL_FILE_BYTES, TextWeights};

/// Special token ids used by the audio-only prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioSpecialTokens {
    /// Beginning-of-sequence token.
    pub bos: u32,
    /// End-of-sequence token.
    pub eos: u32,
    /// Padding token, used to build placeholder embeddings before the audio
    /// features overwrite them.
    pub pad: u32,
    /// Begin-of-audio marker.
    pub boa: u32,
    /// One audio soft-token placeholder.
    pub audio: u32,
    /// End-of-audio marker.
    pub eoa: u32,
}

/// One audio embedding with optional diagnostics.
#[derive(Debug)]
pub struct AudioEmbeddingOutput {
    embedding: Vec<f32>,
    audio: AudioOutput,
    forward: ForwardOutput,
}

impl AudioEmbeddingOutput {
    /// Final normalized embedding.
    pub fn embedding(&self) -> &[f32] {
        &self.embedding
    }

    /// Audio tower output, including captured layers when requested.
    pub fn audio(&self) -> &AudioOutput {
        &self.audio
    }

    /// Text encoder output, including captured layers when requested.
    pub fn forward(&self) -> &ForwardOutput {
        &self.forward
    }
}

/// Audio embedder: WAV preparation, log-mel preprocessing, audio tower, and
/// text encoder.
#[derive(Debug)]
pub struct AudioEmbedder {
    model: TextModel,
    audio: AudioModel,
    tokenizer: Option<Tokenizer>,
    tokens: AudioSpecialTokens,
}

impl AudioEmbedder {
    /// Loads the audio embedder from a model directory containing config.json
    /// and model.safetensors, plus tokenizer.json when available.
    ///
    /// The Safetensors file is read once and shared by the text and audio
    /// weight loaders. Both loaders validate their own tensor inventory. When
    /// tokenizer.json is present, every audio special token id is checked
    /// against it. When it is absent, the ids declared by config.json are used
    /// for audio-only embedding, and text embedding through this embedder
    /// returns an unsupported error instead of pretending to tokenize.
    pub fn load(model_dir: &Path) -> Result<Self> {
        Self::from_files(&ModelFiles::directory(model_dir))
    }

    /// Loads the audio embedder from a directory or bundle source.
    ///
    /// The Safetensors section is read once and shared by the text and audio
    /// weight loaders. When tokenizer.json is present, every audio special
    /// token id is checked against it. When it is absent, audio-only embedding
    /// uses the ids declared by config.json and text embedding through this
    /// embedder returns an unsupported error.
    ///
    /// # Errors
    ///
    /// Returns a typed error for a missing or invalid configuration, a missing
    /// or inconsistent weight tensor, an oversized file, or a tokenizer whose
    /// vocabulary does not match the model.
    pub fn from_files(files: &ModelFiles<'_>) -> Result<Self> {
        let config_text = files.read_utf8("config.json", MAX_CONFIG_BYTES)?;
        let config_value = json::parse(&config_text)?;
        let text_config = TextConfig::from_json_value(&config_value)?;
        let audio_config = AudioConfig::from_json_value(&config_value)?;

        let bytes = files.read("model.safetensors", MAX_MODEL_FILE_BYTES)?;
        let fingerprint = files
            .section_sha256("model.safetensors")
            .map(|digest| hex_digest(&digest));
        let text_weights = match &fingerprint {
            Some(fingerprint) => {
                TextWeights::from_bytes_with_fingerprint(&bytes, &text_config, fingerprint.clone())?
            }
            None => TextWeights::from_bytes(&bytes, &text_config)?,
        };
        let audio_weights = match &fingerprint {
            Some(fingerprint) => AudioWeights::from_bytes_with_fingerprint(
                &bytes,
                &audio_config,
                text_config.hidden_size(),
                fingerprint.clone(),
            )?,
            None => AudioWeights::from_bytes(&bytes, &audio_config, text_config.hidden_size())?,
        };
        let model = TextModel::new(text_config, text_weights)?;
        let audio = AudioModel::new(audio_config, audio_weights, model.config().hidden_size())?;

        let tokenizer = if files.contains("tokenizer.json") {
            let tokenizer = Tokenizer::from_files(files)?;
            if tokenizer.vocab_size() != model.config().vocab_size() {
                return Err(Error::corrupt(format!(
                    "tokenizer vocabulary has {} entries, model declares {}",
                    tokenizer.vocab_size(),
                    model.config().vocab_size()
                )));
            }
            Some(tokenizer)
        } else {
            None
        };
        let tokens = parse_special_tokens(&config_value, tokenizer.as_ref())?;
        Ok(Self {
            model,
            audio,
            tokenizer,
            tokens,
        })
    }

    /// Text model.
    pub fn model(&self) -> &TextModel {
        &self.model
    }

    /// Audio tower.
    pub fn audio(&self) -> &AudioModel {
        &self.audio
    }

    /// Frozen tokenizer, when the model directory provides one.
    pub fn tokenizer(&self) -> Option<&Tokenizer> {
        self.tokenizer.as_ref()
    }

    /// Special token ids used for audio prompts.
    pub fn special_tokens(&self) -> AudioSpecialTokens {
        self.tokens
    }

    /// Embeds raw text with the official prompt prefix.
    ///
    /// This uses the same text model and tokenizer as the audio path, so a
    /// mixed text/audio index shares one embedding space and one loaded model.
    pub fn embed_text(&self, text: &str, prompt: Prompt, dimension: usize) -> Result<Vec<f32>> {
        let embedding_dim = self.model.config().embedding_dim();
        if dimension == 0 || dimension > embedding_dim {
            return Err(Error::internal(format!(
                "embedding dimension {dimension} is outside 1..={embedding_dim}"
            )));
        }
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            Error::unsupported(
                "this model directory has no tokenizer.json; text embedding is unavailable",
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

    /// Embeds one decoded audio clip and returns a Matryoshka-dimension vector.
    pub fn embed(&self, audio: &Audio, dimension: usize) -> Result<Vec<f32>> {
        self.embed_impl(audio, dimension, false)
            .map(|output| output.embedding)
    }

    /// Embeds one decoded audio clip and captures the audio and text layer
    /// states.
    pub fn embed_with_capture(
        &self,
        audio: &Audio,
        dimension: usize,
    ) -> Result<AudioEmbeddingOutput> {
        self.embed_impl(audio, dimension, true)
    }

    fn embed_impl(
        &self,
        audio: &Audio,
        dimension: usize,
        capture: bool,
    ) -> Result<AudioEmbeddingOutput> {
        let embedding_dim = self.model.config().embedding_dim();
        if dimension == 0 || dimension > embedding_dim {
            return Err(Error::internal(format!(
                "embedding dimension {dimension} is outside 1..={embedding_dim}"
            )));
        }
        let samples = audio.samples_16k()?;
        let mel = log_mel(&samples)?;
        let tower = self.audio.forward(&mel, capture)?;
        let soft_tokens = tower.audio_tokens();
        let sequence = soft_tokens
            .checked_add(4)
            .ok_or_else(|| Error::limit_exceeded("audio prompt length overflow"))?;
        if sequence > MAX_TEXT_TOKENS {
            return Err(Error::limit_exceeded(format!(
                "audio prompt has {sequence} tokens, above the {MAX_TEXT_TOKENS}-token context"
            )));
        }
        let mut input_ids = Vec::with_capacity(sequence);
        input_ids.push(self.tokens.bos);
        input_ids.push(self.tokens.boa);
        input_ids.extend(std::iter::repeat_n(self.tokens.audio, soft_tokens));
        input_ids.push(self.tokens.eoa);
        input_ids.push(self.tokens.eos);

        let forward = self.model.forward_with_features(
            &input_ids,
            tower.pooled(),
            2,
            soft_tokens,
            capture,
        )?;
        let embedding = forward.embedding(dimension)?;
        Ok(AudioEmbeddingOutput {
            embedding,
            audio: tower,
            forward,
        })
    }
}

fn parse_special_tokens(
    config: &json::Value,
    tokenizer: Option<&Tokenizer>,
) -> Result<AudioSpecialTokens> {
    let text = config
        .get("text_config")
        .ok_or_else(|| Error::corrupt("model configuration has no text_config"))?;
    let bos = required_token(text, "bos_token_id")?;
    let eos = required_token(text, "eos_token_id")?;
    let pad = required_token(text, "pad_token_id")?;
    let boa = required_token(config, "boa_token_id")?;
    let audio = required_token(config, "audio_token_id")?;
    let eoa = required_token(config, "eoa_token_index")?;

    if let Some(tokenizer) = tokenizer {
        validate_token(tokenizer, "<bos>", bos, "bos_token_id")?;
        validate_token(tokenizer, "<eos>", eos, "eos_token_id")?;
        validate_token(tokenizer, "<pad>", pad, "pad_token_id")?;
        validate_token(tokenizer, "<|audio>", boa, "boa_token_id")?;
        validate_token(tokenizer, "<|audio|>", audio, "audio_token_id")?;
        validate_token(tokenizer, "<audio|>", eoa, "eoa_token_index")?;
    }

    Ok(AudioSpecialTokens {
        bos,
        eos,
        pad,
        boa,
        audio,
        eoa,
    })
}

fn required_token(config: &json::Value, key: &str) -> Result<u32> {
    let value = config
        .get(key)
        .ok_or_else(|| Error::unsupported(format!("model configuration has no {key}")))?;
    let number = value
        .as_number()
        .ok_or_else(|| Error::corrupt(format!("{key} must be a number")))?;
    let id = number
        .as_u64()
        .ok_or_else(|| Error::corrupt(format!("{key} must be a non-negative integer")))?;
    u32::try_from(id).map_err(|_| Error::corrupt(format!("{key} does not fit u32")))
}

fn validate_token(tokenizer: &Tokenizer, token: &str, expected: u32, key: &str) -> Result<()> {
    let actual = tokenizer.token_id(token).ok_or_else(|| {
        Error::corrupt(format!(
            "tokenizer does not define {token}, required by {key}"
        ))
    })?;
    if actual != expected {
        return Err(Error::corrupt(format!(
            "{key} declares {expected}, but the tokenizer maps {token} to {actual}"
        )));
    }
    Ok(())
}
