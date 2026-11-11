//! Image embedding path for EmbeddingGemma 2.
//!
//! The image embedder connects the native decoders, the official
//! preprocessing, the vision tower, and the text encoder. An image-only input
//! becomes the token layout <bos><|image><|image|>*soft_tokens<image|><eos>;
//! the projected vision soft tokens replace the placeholder embeddings, and
//! the text encoder produces the final 768-dimensional vector. Mean pooling
//! covers every token, including the begin/end image markers and the text
//! special tokens, exactly as the reference model does.
//!
//! The product never falls back to a file name, a caption, or a textual
//! marker: a failure anywhere in the path is returned as a typed error.

use std::path::Path;

use crate::binary::read_file_bounded;
use crate::error::{Error, Result};
use crate::json;
use crate::media::Image;
use crate::media::preprocess::{PreprocessOptions, preprocess};
use crate::model::config::{MAX_CONFIG_BYTES, MAX_TEXT_TOKENS, TextConfig};
use crate::model::text::{ForwardOutput, TextModel};
use crate::model::tokenizer::Tokenizer;
use crate::model::vision::{VisionConfig, VisionModel, VisionOutput, VisionWeights};
use crate::model::weights::{MAX_MODEL_FILE_BYTES, TextWeights};

/// Special token ids used by the image-only prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSpecialTokens {
    /// Beginning-of-sequence token.
    pub bos: u32,
    /// End-of-sequence token.
    pub eos: u32,
    /// Padding token, used to build placeholder embeddings before the vision
    /// features overwrite them.
    pub pad: u32,
    /// Begin-of-image marker.
    pub boi: u32,
    /// One image soft-token placeholder.
    pub image: u32,
    /// End-of-image marker.
    pub eoi: u32,
}

/// One image embedding with optional diagnostics.
#[derive(Debug)]
pub struct ImageEmbeddingOutput {
    embedding: Vec<f32>,
    vision: VisionOutput,
    forward: ForwardOutput,
}

impl ImageEmbeddingOutput {
    /// Final normalized embedding.
    pub fn embedding(&self) -> &[f32] {
        &self.embedding
    }

    /// Vision tower output, including captured layers when requested.
    pub fn vision(&self) -> &VisionOutput {
        &self.vision
    }

    /// Text encoder output, including captured layers when requested.
    pub fn forward(&self) -> &ForwardOutput {
        &self.forward
    }
}

/// Image embedder: preprocessing, vision tower, and text encoder.
#[derive(Debug)]
pub struct ImageEmbedder {
    model: TextModel,
    vision: VisionModel,
    tokenizer: Tokenizer,
    tokens: ImageSpecialTokens,
}

impl ImageEmbedder {
    /// Loads the image embedder from a model directory containing config.json,
    /// model.safetensors, and tokenizer.json.
    ///
    /// The Safetensors file is read once and shared by the text and vision
    /// weight loaders. Both loaders validate their own tensor inventory.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config_bytes = read_file_bounded(&model_dir.join("config.json"), MAX_CONFIG_BYTES)?;
        let config_text = std::str::from_utf8(&config_bytes)
            .map_err(|_| Error::corrupt("config.json is not valid UTF-8"))?;
        let config_value = json::parse(config_text)?;
        let text_config = TextConfig::from_json_value(&config_value)?;
        let vision_config = VisionConfig::from_json_value(&config_value)?;

        let bytes = read_file_bounded(&model_dir.join("model.safetensors"), MAX_MODEL_FILE_BYTES)?;
        let text_weights = TextWeights::from_bytes(&bytes, &text_config)?;
        let vision_weights =
            VisionWeights::from_bytes(&bytes, &vision_config, text_config.hidden_size())?;
        let model = TextModel::new(text_config, text_weights)?;
        let vision = VisionModel::new(vision_config, vision_weights, model.config().hidden_size())?;

        let tokenizer = Tokenizer::load(&model_dir.join("tokenizer.json"))?;
        if tokenizer.vocab_size() != model.config().vocab_size() {
            return Err(Error::corrupt(format!(
                "tokenizer vocabulary has {} entries, model declares {}",
                tokenizer.vocab_size(),
                model.config().vocab_size()
            )));
        }
        let tokens = parse_special_tokens(&config_value, &tokenizer)?;
        Ok(Self {
            model,
            vision,
            tokenizer,
            tokens,
        })
    }

    /// Text model.
    pub fn model(&self) -> &TextModel {
        &self.model
    }

    /// Vision tower.
    pub fn vision(&self) -> &VisionModel {
        &self.vision
    }

    /// Frozen tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Special token ids used for image prompts.
    pub fn special_tokens(&self) -> ImageSpecialTokens {
        self.tokens
    }

    /// Embeds one decoded image and returns a Matryoshka-dimension vector.
    pub fn embed(
        &self,
        image: &Image,
        dimension: usize,
        options: PreprocessOptions,
    ) -> Result<Vec<f32>> {
        self.embed_impl(image, dimension, options, false)
            .map(|output| output.embedding)
    }

    /// Embeds one decoded image and captures the vision and text layer states.
    pub fn embed_with_capture(
        &self,
        image: &Image,
        dimension: usize,
        options: PreprocessOptions,
    ) -> Result<ImageEmbeddingOutput> {
        self.embed_impl(image, dimension, options, true)
    }

    fn embed_impl(
        &self,
        image: &Image,
        dimension: usize,
        options: PreprocessOptions,
        capture: bool,
    ) -> Result<ImageEmbeddingOutput> {
        let embedding_dim = self.model.config().embedding_dim();
        if dimension == 0 || dimension > embedding_dim {
            return Err(Error::internal(format!(
                "embedding dimension {dimension} is outside 1..={embedding_dim}"
            )));
        }
        let patches = preprocess(image, options)?;
        let vision = self.vision.forward(&patches, capture)?;
        let soft_tokens = vision.soft_tokens();
        let sequence = soft_tokens
            .checked_add(4)
            .ok_or_else(|| Error::limit_exceeded("image prompt length overflow"))?;
        if sequence > MAX_TEXT_TOKENS {
            return Err(Error::limit_exceeded(format!(
                "image prompt has {sequence} tokens, above the {MAX_TEXT_TOKENS}-token context"
            )));
        }
        let mut input_ids = Vec::with_capacity(sequence);
        input_ids.push(self.tokens.bos);
        input_ids.push(self.tokens.boi);
        input_ids.extend(std::iter::repeat_n(self.tokens.image, soft_tokens));
        input_ids.push(self.tokens.eoi);
        input_ids.push(self.tokens.eos);

        let forward = self.model.forward_with_features(
            &input_ids,
            vision.pooled(),
            2,
            soft_tokens,
            capture,
        )?;
        let embedding = forward.embedding(dimension)?;
        Ok(ImageEmbeddingOutput {
            embedding,
            vision,
            forward,
        })
    }
}

fn parse_special_tokens(config: &json::Value, tokenizer: &Tokenizer) -> Result<ImageSpecialTokens> {
    let text = config
        .get("text_config")
        .ok_or_else(|| Error::corrupt("model configuration has no text_config"))?;
    let bos = required_token(text, "bos_token_id")?;
    let eos = required_token(text, "eos_token_id")?;
    let pad = required_token(text, "pad_token_id")?;
    let boi = required_token(config, "boi_token_id")?;
    let image = required_token(config, "image_token_id")?;
    let eoi = required_token(config, "eoi_token_id")?;

    validate_token(tokenizer, "<bos>", bos, "bos_token_id")?;
    validate_token(tokenizer, "<eos>", eos, "eos_token_id")?;
    validate_token(tokenizer, "<pad>", pad, "pad_token_id")?;
    validate_token(tokenizer, "<|image>", boi, "boi_token_id")?;
    validate_token(tokenizer, "<|image|>", image, "image_token_id")?;
    validate_token(tokenizer, "<image|>", eoi, "eoi_token_id")?;

    Ok(ImageSpecialTokens {
        bos,
        eos,
        pad,
        boi,
        image,
        eoi,
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
