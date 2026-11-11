//! Gemma 4 audio tower inference for EmbeddingGemma 2.
//!
//! The tower consumes the official 128-bin log-mel features and their frame
//! mask. It applies two stride-2 convolution/normalization blocks, sinusoidal
//! relative positional encoding, 12 chunked-local-attention encoder layers,
//! and the 1536-dimensional output projection. The projection into the text
//! hidden space is performed by the embedder, mirroring the vision path.
//!
//! The implementation is strict: an unknown activation, a missing clipping
//! bound, a non-silence activation, an inconsistent head layout, a missing
//! tensor, or a shape mismatch is rejected instead of approximated. The
//! attention is computed only for valid (non-padding) queries; trailing
//! padding cannot influence a valid query because the tower is causal along
//! time.

use std::borrow::Cow;
use std::path::Path;

use crate::binary::read_file_bounded;
use crate::binary::safetensors::SafetensorsFile;
use crate::binary::sha256::{hex_digest, sha256_hex};
use crate::error::{Error, Result};
use crate::json::{self, Limits, Number, Value};
use crate::media::audio_preprocess::{FEATURE_SIZE, MelSpectrogram};
use crate::model::source::ModelFiles;
use crate::model::text::MAX_ACTIVATION_ELEMENTS;
use crate::tensor::{Shape, Tensor, kernels};

/// Maximum audio hidden size.
pub const MAX_AUDIO_HIDDEN: usize = 16_384;

/// Maximum audio feed-forward intermediate size.
pub const MAX_AUDIO_INTERMEDIATE: usize = 65_536;

/// Maximum number of audio transformer layers.
pub const MAX_AUDIO_LAYERS: usize = 128;

/// Maximum number of attention heads.
pub const MAX_AUDIO_HEADS: usize = 128;

/// Maximum attention head dimension.
pub const MAX_AUDIO_HEAD_DIM: usize = 1_024;

/// Maximum number of subsampling convolution channels.
pub const MAX_AUDIO_CONV_CHANNELS: usize = 4_096;

/// Maximum output projection width.
pub const MAX_AUDIO_OUTPUT_DIMS: usize = 16_384;

/// Maximum relative-position context size.
pub const MAX_AUDIO_CONTEXT: usize = 4_096;

/// Maximum size of an audio configuration file.
pub const MAX_AUDIO_CONFIG_BYTES: u64 = 1024 * 1024;

/// Maximum size of the Safetensors file accepted by the loader.
pub const MAX_AUDIO_FILE_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Maximum total number of FP32 audio weight elements kept in memory.
pub const MAX_AUDIO_WEIGHT_ELEMENTS: usize = 1 << 30;

/// Validated Gemma 4 audio configuration.
#[derive(Debug, Clone)]
pub struct AudioConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    head_dim: usize,
    conv_kernel_size: usize,
    conv_channels: [usize; 2],
    attention_chunk_size: usize,
    attention_context_left: usize,
    attention_context_right: usize,
    attention_logit_cap: f32,
    rms_norm_eps: f32,
    output_proj_dims: usize,
    residual_weight: f32,
    gradient_clipping: f32,
}

impl AudioConfig {
    /// Parses the audio section of a composite model configuration.
    pub fn from_json_str(input: &str) -> Result<Self> {
        let limits = Limits {
            max_input_bytes: MAX_AUDIO_CONFIG_BYTES as usize,
            ..Limits::default()
        };
        let value = json::parse_with_limits(input, &limits)?;
        Self::from_json_value(&value)
    }

    /// Parses the audio configuration from an already parsed JSON value.
    pub fn from_json_value(value: &Value) -> Result<Self> {
        let root = value
            .as_object()
            .ok_or_else(|| Error::corrupt("model configuration must be a JSON object"))?;
        let audio = match root.get("audio_config") {
            Some(nested) => nested
                .as_object()
                .ok_or_else(|| Error::corrupt("audio_config must be a JSON object"))?,
            None => {
                return Err(Error::unsupported(
                    "the model has no audio_config; audio embeddings are unavailable",
                ));
            }
        };

        if let Some(model_type) = audio.get("model_type") {
            let model_type = model_type
                .as_str()
                .ok_or_else(|| Error::corrupt("audio model_type must be a string"))?;
            if model_type != "gemma4_audio" {
                return Err(Error::unsupported(format!(
                    "audio model type {model_type} is not gemma4_audio"
                )));
            }
        }
        if let Some(dtype) = audio.get("dtype") {
            let dtype = dtype
                .as_str()
                .ok_or_else(|| Error::corrupt("audio dtype must be a string"))?;
            match dtype {
                "bfloat16" | "float32" => {}
                "float16" => {
                    return Err(Error::unsupported(
                        "float16 is forbidden for EmbeddingGemma 2 activations",
                    ));
                }
                other => {
                    return Err(Error::unsupported(format!(
                        "audio dtype {other} is not supported"
                    )));
                }
            }
        }

        let hidden_size = unsigned(audio, "hidden_size")?;
        let num_hidden_layers = unsigned(audio, "num_hidden_layers")?;
        let num_attention_heads = unsigned(audio, "num_attention_heads")?;
        let conv_kernel_size = unsigned(audio, "conv_kernel_size")?;
        let attention_chunk_size = unsigned(audio, "attention_chunk_size")?;
        let attention_context_left = unsigned(audio, "attention_context_left")?;
        let attention_context_right = unsigned(audio, "attention_context_right")?;
        let output_proj_dims = unsigned(audio, "output_proj_dims")?;
        let rms_norm_eps = float(audio, "rms_norm_eps")? as f32;
        let attention_logit_cap = float(audio, "attention_logit_cap")? as f32;
        let residual_weight = float(audio, "residual_weight")? as f32;
        let gradient_clipping = float(audio, "gradient_clipping")? as f32;

        let hidden_activation = string(audio, "hidden_act")?;
        if hidden_activation != "silu" {
            return Err(Error::unsupported(format!(
                "audio activation {hidden_activation} is not silu"
            )));
        }
        if !optional_bool(audio, "use_clipped_linears")?.unwrap_or(false) {
            return Err(Error::unsupported(
                "audio clipped linear layers are required by the frozen model",
            ));
        }
        if let Some(dropout) = optional_float(audio, "attention_dropout")?
            && dropout != 0.0
        {
            return Err(Error::unsupported(
                "audio attention dropout is not implemented",
            ));
        }

        if !hidden_size.is_multiple_of(2) {
            return Err(Error::corrupt(
                "audio hidden size must be even for sinusoidal relative positions",
            ));
        }
        if num_attention_heads == 0 || !hidden_size.is_multiple_of(num_attention_heads) {
            return Err(Error::corrupt(
                "audio hidden size must be divisible by a positive head count",
            ));
        }
        let head_dim = hidden_size / num_attention_heads;
        if head_dim == 0 || head_dim > MAX_AUDIO_HEAD_DIM {
            return Err(Error::limit_exceeded(format!(
                "audio head dimension {head_dim} is outside 1..={MAX_AUDIO_HEAD_DIM}"
            )));
        }
        let intermediate_size = hidden_size
            .checked_mul(4)
            .ok_or_else(|| Error::limit_exceeded("audio intermediate size overflow"))?;
        if hidden_size > MAX_AUDIO_HIDDEN
            || intermediate_size > MAX_AUDIO_INTERMEDIATE
            || num_hidden_layers > MAX_AUDIO_LAYERS
            || num_attention_heads > MAX_AUDIO_HEADS
            || output_proj_dims == 0
            || output_proj_dims > MAX_AUDIO_OUTPUT_DIMS
        {
            return Err(Error::limit_exceeded(
                "audio dimensions exceed their documented caps",
            ));
        }
        if conv_kernel_size == 0 || conv_kernel_size > 31 {
            return Err(Error::limit_exceeded(format!(
                "audio convolution kernel {conv_kernel_size} is outside 1..=31"
            )));
        }
        if attention_chunk_size == 0 || attention_context_left < 2 {
            return Err(Error::corrupt(
                "audio chunked attention requires a positive chunk and at least two context positions on the left",
            ));
        }
        let context_size = attention_chunk_size
            .checked_add(attention_context_left - 1)
            .and_then(|value| value.checked_add(attention_context_right))
            .ok_or_else(|| Error::limit_exceeded("audio attention context overflow"))?;
        if context_size == 0 || context_size > MAX_AUDIO_CONTEXT {
            return Err(Error::limit_exceeded(format!(
                "audio attention context {context_size} is outside 1..={MAX_AUDIO_CONTEXT}"
            )));
        }
        if !rms_norm_eps.is_finite() || rms_norm_eps < 0.0 {
            return Err(Error::corrupt(
                "audio RMSNorm epsilon must be finite and non-negative",
            ));
        }
        if !attention_logit_cap.is_finite() || attention_logit_cap <= 0.0 {
            return Err(Error::corrupt(
                "audio attention logit cap must be finite and positive",
            ));
        }
        if !residual_weight.is_finite() {
            return Err(Error::corrupt("audio residual weight must be finite"));
        }
        if !gradient_clipping.is_finite() || gradient_clipping <= 0.0 {
            return Err(Error::corrupt(
                "audio gradient clipping must be finite and positive",
            ));
        }

        let channels = audio
            .get("subsampling_conv_channels")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::corrupt("audio subsampling_conv_channels must be an array"))?;
        if channels.len() != 2 {
            return Err(Error::unsupported(
                "audio subsampling requires exactly two convolution channels",
            ));
        }
        let first = channels[0]
            .as_number()
            .and_then(Number::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| Error::corrupt("audio conv channel must be a non-negative integer"))?;
        let second = channels[1]
            .as_number()
            .and_then(Number::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| Error::corrupt("audio conv channel must be a non-negative integer"))?;
        if first == 0
            || second == 0
            || first > MAX_AUDIO_CONV_CHANNELS
            || second > MAX_AUDIO_CONV_CHANNELS
            || !first.is_multiple_of(4)
        {
            return Err(Error::limit_exceeded(
                "audio subsampling channels exceed their documented caps or are not divisible by four",
            ));
        }
        let projection_input = (first / 4)
            .checked_mul(second)
            .ok_or_else(|| Error::limit_exceeded("audio subsampling projection overflow"))?;
        if projection_input == 0 {
            return Err(Error::corrupt(
                "audio subsampling projection has no input features",
            ));
        }

        Ok(Self {
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            head_dim,
            conv_kernel_size,
            conv_channels: [first, second],
            attention_chunk_size,
            attention_context_left,
            attention_context_right,
            attention_logit_cap,
            rms_norm_eps,
            output_proj_dims,
            residual_weight,
            gradient_clipping,
        })
    }

    /// Loads the audio configuration from a file with an explicit size cap.
    pub fn from_file(path: &Path) -> Result<Self> {
        let bytes = read_file_bounded(path, MAX_AUDIO_CONFIG_BYTES)?;
        let text = std::str::from_utf8(&bytes)
            .map_err(|_| Error::corrupt(format!("{} is not valid UTF-8", path.display())))?;
        Self::from_json_str(text)
    }

    /// Audio hidden size.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Feed-forward intermediate size.
    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }

    /// Number of encoder layers.
    pub fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }

    /// Number of attention heads.
    pub fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }

    /// Attention head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Causal depthwise convolution kernel size.
    pub fn conv_kernel_size(&self) -> usize {
        self.conv_kernel_size
    }

    /// Subsampling convolution channel counts.
    pub fn conv_channels(&self) -> [usize; 2] {
        self.conv_channels
    }

    /// Chunk size of the local attention.
    pub fn attention_chunk_size(&self) -> usize {
        self.attention_chunk_size
    }

    /// Number of past context positions (including the current position).
    pub fn attention_context_left(&self) -> usize {
        self.attention_context_left
    }

    /// Number of future context positions.
    pub fn attention_context_right(&self) -> usize {
        self.attention_context_right
    }

    /// Number of context positions in one attention window.
    pub fn context_size(&self) -> usize {
        self.attention_chunk_size + self.attention_context_left - 1 + self.attention_context_right
    }

    /// Number of sinusoidal relative-position entries.
    pub fn position_count(&self) -> usize {
        self.context_size() / 2 + 1
    }

    /// Attention logit soft cap.
    pub fn attention_logit_cap(&self) -> f32 {
        self.attention_logit_cap
    }

    /// RMSNorm epsilon.
    pub fn rms_norm_eps(&self) -> f32 {
        self.rms_norm_eps
    }

    /// Output projection width.
    pub fn output_proj_dims(&self) -> usize {
        self.output_proj_dims
    }

    /// Residual weight applied to the feed-forward branch.
    pub fn residual_weight(&self) -> f32 {
        self.residual_weight
    }

    /// Gradient clipping bound.
    pub fn gradient_clipping(&self) -> f32 {
        self.gradient_clipping
    }

    /// Input width of the subsampling projection.
    pub fn subsampling_projection_input(&self) -> usize {
        (self.conv_channels[0] / 4) * self.conv_channels[1]
    }

    /// Expected tensor names and shapes for the audio weights.
    pub fn tensor_shapes(&self, text_hidden: usize) -> Vec<(String, Vec<usize>)> {
        let hidden = self.hidden_size;
        let intermediate = self.intermediate_size;
        let head_dim = self.head_dim;
        let [first, second] = self.conv_channels;
        let projection_input = self.subsampling_projection_input();
        let mut shapes = Vec::with_capacity(8 + self.num_hidden_layers * 62);
        shapes.push((
            "audio_tower.subsample_conv_projection.layer0.conv.weight".to_owned(),
            vec![first, 1, 3, 3],
        ));
        shapes.push((
            "audio_tower.subsample_conv_projection.layer0.norm.weight".to_owned(),
            vec![first],
        ));
        shapes.push((
            "audio_tower.subsample_conv_projection.layer1.conv.weight".to_owned(),
            vec![second, first, 3, 3],
        ));
        shapes.push((
            "audio_tower.subsample_conv_projection.layer1.norm.weight".to_owned(),
            vec![second],
        ));
        shapes.push((
            "audio_tower.subsample_conv_projection.input_proj_linear.weight".to_owned(),
            vec![hidden, projection_input],
        ));
        for index in 0..self.num_hidden_layers {
            let prefix = format!("audio_tower.layers.{index}");
            for feed_forward in ["feed_forward1", "feed_forward2"] {
                let branch = format!("{prefix}.{feed_forward}");
                shapes.push((format!("{branch}.pre_layer_norm.weight"), vec![hidden]));
                shapes.push((format!("{branch}.post_layer_norm.weight"), vec![hidden]));
                shapes.push((
                    format!("{branch}.ffw_layer_1.linear.weight"),
                    vec![intermediate, hidden],
                ));
                shapes.push((
                    format!("{branch}.ffw_layer_2.linear.weight"),
                    vec![hidden, intermediate],
                ));
                for linear in ["ffw_layer_1", "ffw_layer_2"] {
                    for bound in ["input_min", "input_max", "output_min", "output_max"] {
                        shapes.push((format!("{branch}.{linear}.{bound}"), Vec::new()));
                    }
                }
            }
            for projection in ["q_proj", "k_proj", "v_proj", "post"] {
                shapes.push((
                    format!("{prefix}.self_attn.{projection}.linear.weight"),
                    vec![hidden, hidden],
                ));
                for bound in ["input_min", "input_max", "output_min", "output_max"] {
                    shapes.push((
                        format!("{prefix}.self_attn.{projection}.{bound}"),
                        Vec::new(),
                    ));
                }
            }
            shapes.push((
                format!("{prefix}.self_attn.relative_k_proj.weight"),
                vec![hidden, hidden],
            ));
            shapes.push((format!("{prefix}.self_attn.per_dim_scale"), vec![head_dim]));
            shapes.push((
                format!("{prefix}.lconv1d.pre_layer_norm.weight"),
                vec![hidden],
            ));
            shapes.push((
                format!("{prefix}.lconv1d.linear_start.linear.weight"),
                vec![hidden * 2, hidden],
            ));
            shapes.push((
                format!("{prefix}.lconv1d.linear_end.linear.weight"),
                vec![hidden, hidden],
            ));
            for linear in ["linear_start", "linear_end"] {
                for bound in ["input_min", "input_max", "output_min", "output_max"] {
                    shapes.push((format!("{prefix}.lconv1d.{linear}.{bound}"), Vec::new()));
                }
            }
            shapes.push((
                format!("{prefix}.lconv1d.depthwise_conv1d.weight"),
                vec![hidden, 1, self.conv_kernel_size],
            ));
            shapes.push((format!("{prefix}.lconv1d.conv_norm.weight"), vec![hidden]));
            shapes.push((format!("{prefix}.norm_pre_attn.weight"), vec![hidden]));
            shapes.push((format!("{prefix}.norm_post_attn.weight"), vec![hidden]));
            shapes.push((format!("{prefix}.norm_out.weight"), vec![hidden]));
        }
        shapes.push((
            "audio_tower.output_proj.weight".to_owned(),
            vec![self.output_proj_dims, hidden],
        ));
        shapes.push((
            "audio_tower.output_proj.bias".to_owned(),
            vec![self.output_proj_dims],
        ));
        shapes.push((
            "embed_audio.embedding_projection.weight".to_owned(),
            vec![text_hidden, self.output_proj_dims],
        ));
        shapes
    }
}

fn unsigned(object: &json::Object, key: &str) -> Result<usize> {
    let value = object
        .get(key)
        .ok_or_else(|| Error::corrupt(format!("audio configuration is missing {key}")))?;
    let number = value
        .as_number()
        .ok_or_else(|| Error::corrupt(format!("{key} must be a number")))?;
    let value = number
        .as_u64()
        .ok_or_else(|| Error::corrupt(format!("{key} must be a non-negative integer")))?;
    usize::try_from(value).map_err(|_| Error::limit_exceeded(format!("{key} does not fit usize")))
}

fn float(object: &json::Object, key: &str) -> Result<f64> {
    object
        .get(key)
        .and_then(|value| value.as_number())
        .map(Number::as_f64)
        .ok_or_else(|| Error::corrupt(format!("{key} must be a number")))
}

fn string<'a>(object: &'a json::Object, key: &str) -> Result<&'a str> {
    object
        .get(key)
        .and_then(|value| value.as_str())
        .ok_or_else(|| Error::corrupt(format!("{key} must be a string")))
}

fn optional_bool(object: &json::Object, key: &str) -> Result<Option<bool>> {
    match object.get(key) {
        None => Ok(None),
        Some(value) => value
            .as_bool()
            .map(Some)
            .ok_or_else(|| Error::corrupt(format!("{key} must be a boolean"))),
    }
}

fn optional_float(object: &json::Object, key: &str) -> Result<Option<f64>> {
    match object.get(key) {
        None => Ok(None),
        Some(value) => value
            .as_number()
            .map(|number| Some(number.as_f64()))
            .ok_or_else(|| Error::corrupt(format!("{key} must be a number"))),
    }
}

/// A linear layer with the frozen model's input/output clipping bounds.
#[derive(Debug)]
struct ClippedLinear {
    weight: Vec<f32>,
    input_min: f32,
    input_max: f32,
    output_min: f32,
    output_max: f32,
}

impl ClippedLinear {
    fn apply(&self, input: &[f32], rows: usize, in_features: usize) -> Result<Vec<f32>> {
        let out_features = self.weight.len() / in_features;
        if self.weight.len() != out_features * in_features {
            return Err(Error::internal("audio linear weight has an invalid length"));
        }
        let input_len = rows
            .checked_mul(in_features)
            .ok_or_else(|| Error::limit_exceeded("audio linear input size overflow"))?;
        let output_len = rows
            .checked_mul(out_features)
            .ok_or_else(|| Error::limit_exceeded("audio linear output size overflow"))?;
        if input.len() != input_len {
            return Err(Error::internal(format!(
                "audio linear input has {} elements, expected {input_len}",
                input.len()
            )));
        }
        if output_len > MAX_ACTIVATION_ELEMENTS {
            return Err(Error::limit_exceeded(format!(
                "audio linear output would allocate {output_len} elements"
            )));
        }
        let mut output = vec![0.0f32; output_len];
        for row in 0..rows {
            let input_row = &input[row * in_features..(row + 1) * in_features];
            let output_row = &mut output[row * out_features..(row + 1) * out_features];
            for (column, destination) in output_row.iter_mut().enumerate() {
                let weight_row = &self.weight[column * in_features..(column + 1) * in_features];
                let mut sum = 0.0f32;
                for (value, weight_value) in input_row.iter().zip(weight_row) {
                    sum += clamp_value(*value, self.input_min, self.input_max) * weight_value;
                }
                *destination = clamp_value(sum, self.output_min, self.output_max);
            }
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(Error::internal(
                "audio clipped linear produced a non-finite value",
            ));
        }
        Ok(output)
    }
}

/// Weights of one feed-forward branch.
#[derive(Debug)]
struct FeedForwardWeights {
    pre_layer_norm: Vec<f32>,
    ffw_layer_1: ClippedLinear,
    ffw_layer_2: ClippedLinear,
    post_layer_norm: Vec<f32>,
}

/// Weights of the chunked local attention.
#[derive(Debug)]
struct AttentionWeights {
    q_proj: ClippedLinear,
    k_proj: ClippedLinear,
    v_proj: ClippedLinear,
    post: ClippedLinear,
    relative_k_proj: Vec<f32>,
    per_dim_scale: Vec<f32>,
}

/// Weights of the light convolution block.
#[derive(Debug)]
struct LightConvWeights {
    pre_layer_norm: Vec<f32>,
    linear_start: ClippedLinear,
    depthwise_conv1d: Vec<f32>,
    conv_norm: Vec<f32>,
    linear_end: ClippedLinear,
}

/// Weights of one audio encoder layer.
#[derive(Debug)]
struct AudioLayerWeights {
    feed_forward1: FeedForwardWeights,
    feed_forward2: FeedForwardWeights,
    attention: AttentionWeights,
    light_conv: LightConvWeights,
    norm_pre_attn: Vec<f32>,
    norm_post_attn: Vec<f32>,
    norm_out: Vec<f32>,
}

/// FP32 weights of the audio tower and its text-space projection.
#[derive(Debug)]
pub struct AudioWeights {
    layer0_conv: Vec<f32>,
    layer0_norm: Vec<f32>,
    layer1_conv: Vec<f32>,
    layer1_norm: Vec<f32>,
    input_proj_linear: Vec<f32>,
    layers: Vec<AudioLayerWeights>,
    output_proj: Vec<f32>,
    output_proj_bias: Vec<f32>,
    embedding_projection: Vec<f32>,
    fingerprint: String,
    parameters: usize,
}

impl AudioWeights {
    /// Loads the audio weights from a Safetensors file.
    pub fn from_file(path: &Path, config: &AudioConfig, text_hidden: usize) -> Result<Self> {
        let bytes = read_file_bounded(path, MAX_AUDIO_FILE_BYTES)?;
        Self::from_bytes(&bytes, config, text_hidden)
    }

    /// Loads the audio weights from a directory or bundle source.
    pub fn from_files(
        files: &ModelFiles<'_>,
        config: &AudioConfig,
        text_hidden: usize,
    ) -> Result<Self> {
        let bytes = files.read("model.safetensors", MAX_AUDIO_FILE_BYTES)?;
        match files.section_sha256("model.safetensors") {
            Some(digest) => {
                Self::from_bytes_with_fingerprint(&bytes, config, text_hidden, hex_digest(&digest))
            }
            None => Self::from_bytes(&bytes, config, text_hidden),
        }
    }

    /// Loads the audio weights from an in-memory Safetensors file.
    pub fn from_bytes(bytes: &[u8], config: &AudioConfig, text_hidden: usize) -> Result<Self> {
        let fingerprint = sha256_hex(bytes);
        Self::from_bytes_with_fingerprint(bytes, config, text_hidden, fingerprint)
    }

    /// Loads the audio weights with an already verified SHA-256 fingerprint.
    ///
    /// The caller must have verified that `fingerprint` belongs to `bytes`.
    pub fn from_bytes_with_fingerprint(
        bytes: &[u8],
        config: &AudioConfig,
        text_hidden: usize,
        fingerprint: String,
    ) -> Result<Self> {
        let file = SafetensorsFile::parse(bytes)?;
        Self::decode(&file, config, text_hidden, fingerprint)
    }

    /// SHA-256 fingerprint of the source file.
    pub fn fingerprint_hex(&self) -> &str {
        &self.fingerprint
    }

    /// Number of FP32 parameters held in memory.
    pub fn parameters(&self) -> usize {
        self.parameters
    }

    /// Number of encoder layers.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    fn decode(
        file: &SafetensorsFile<'_>,
        config: &AudioConfig,
        text_hidden: usize,
        fingerprint: String,
    ) -> Result<Self> {
        let expected = config.tensor_shapes(text_hidden);
        let expected_names: std::collections::BTreeSet<&str> =
            expected.iter().map(|(name, _)| name.as_str()).collect();
        for entry in file.entries() {
            let name = entry.name();
            let is_audio = name.starts_with("audio_tower.") || name.starts_with("embed_audio.");
            if is_audio && !expected_names.contains(name) {
                return Err(Error::corrupt(format!(
                    "unexpected audio tensor {name} in the weight file"
                )));
            }
        }

        let mut decoded: std::collections::BTreeMap<String, Tensor> =
            std::collections::BTreeMap::new();
        let mut parameters: usize = 0;
        for (name, shape) in &expected {
            let entry = file.tensor(name).ok_or_else(|| {
                Error::corrupt(format!(
                    "audio weight {name} is missing from the Safetensors file"
                ))
            })?;
            if entry.shape().dims() != shape.as_slice() {
                return Err(Error::corrupt(format!(
                    "audio weight {name} has shape {}, expected {shape:?}",
                    entry.shape()
                )));
            }
            parameters = parameters
                .checked_add(entry.shape().numel())
                .ok_or_else(|| Error::limit_exceeded("audio weight element count overflow"))?;
            if parameters > MAX_AUDIO_WEIGHT_ELEMENTS {
                return Err(Error::limit_exceeded(format!(
                    "audio weights exceed the {MAX_AUDIO_WEIGHT_ELEMENTS}-element cap"
                )));
            }
            let tensor = entry.to_f32()?;
            tensor.validate_finite()?;
            decoded.insert(name.clone(), tensor);
        }

        let layers = build_layers(&mut decoded, config)?;
        Ok(Self {
            layer0_conv: take(
                &mut decoded,
                "audio_tower.subsample_conv_projection.layer0.conv.weight",
            )?,
            layer0_norm: take(
                &mut decoded,
                "audio_tower.subsample_conv_projection.layer0.norm.weight",
            )?,
            layer1_conv: take(
                &mut decoded,
                "audio_tower.subsample_conv_projection.layer1.conv.weight",
            )?,
            layer1_norm: take(
                &mut decoded,
                "audio_tower.subsample_conv_projection.layer1.norm.weight",
            )?,
            input_proj_linear: take(
                &mut decoded,
                "audio_tower.subsample_conv_projection.input_proj_linear.weight",
            )?,
            layers,
            output_proj: take(&mut decoded, "audio_tower.output_proj.weight")?,
            output_proj_bias: take(&mut decoded, "audio_tower.output_proj.bias")?,
            embedding_projection: take(&mut decoded, "embed_audio.embedding_projection.weight")?,
            fingerprint,
            parameters,
        })
    }
}

fn build_layers(
    decoded: &mut std::collections::BTreeMap<String, Tensor>,
    config: &AudioConfig,
) -> Result<Vec<AudioLayerWeights>> {
    let mut layers = Vec::with_capacity(config.num_hidden_layers());
    for index in 0..config.num_hidden_layers() {
        let prefix = format!("audio_tower.layers.{index}");
        let feed_forward1 = build_feed_forward(decoded, &format!("{prefix}.feed_forward1"))?;
        let feed_forward2 = build_feed_forward(decoded, &format!("{prefix}.feed_forward2"))?;
        let attention_prefix = format!("{prefix}.self_attn");
        let attention = AttentionWeights {
            q_proj: build_clipped_linear(decoded, &format!("{attention_prefix}.q_proj"))?,
            k_proj: build_clipped_linear(decoded, &format!("{attention_prefix}.k_proj"))?,
            v_proj: build_clipped_linear(decoded, &format!("{attention_prefix}.v_proj"))?,
            post: build_clipped_linear(decoded, &format!("{attention_prefix}.post"))?,
            relative_k_proj: take(
                decoded,
                &format!("{attention_prefix}.relative_k_proj.weight"),
            )?,
            per_dim_scale: take(decoded, &format!("{attention_prefix}.per_dim_scale"))?,
        };
        let conv_prefix = format!("{prefix}.lconv1d");
        let light_conv = LightConvWeights {
            pre_layer_norm: take(decoded, &format!("{conv_prefix}.pre_layer_norm.weight"))?,
            linear_start: build_clipped_linear(decoded, &format!("{conv_prefix}.linear_start"))?,
            depthwise_conv1d: take(decoded, &format!("{conv_prefix}.depthwise_conv1d.weight"))?,
            conv_norm: take(decoded, &format!("{conv_prefix}.conv_norm.weight"))?,
            linear_end: build_clipped_linear(decoded, &format!("{conv_prefix}.linear_end"))?,
        };
        layers.push(AudioLayerWeights {
            feed_forward1,
            feed_forward2,
            attention,
            light_conv,
            norm_pre_attn: take(decoded, &format!("{prefix}.norm_pre_attn.weight"))?,
            norm_post_attn: take(decoded, &format!("{prefix}.norm_post_attn.weight"))?,
            norm_out: take(decoded, &format!("{prefix}.norm_out.weight"))?,
        });
    }
    Ok(layers)
}

fn build_feed_forward(
    decoded: &mut std::collections::BTreeMap<String, Tensor>,
    prefix: &str,
) -> Result<FeedForwardWeights> {
    Ok(FeedForwardWeights {
        pre_layer_norm: take(decoded, &format!("{prefix}.pre_layer_norm.weight"))?,
        ffw_layer_1: build_clipped_linear(decoded, &format!("{prefix}.ffw_layer_1"))?,
        ffw_layer_2: build_clipped_linear(decoded, &format!("{prefix}.ffw_layer_2"))?,
        post_layer_norm: take(decoded, &format!("{prefix}.post_layer_norm.weight"))?,
    })
}

fn build_clipped_linear(
    decoded: &mut std::collections::BTreeMap<String, Tensor>,
    prefix: &str,
) -> Result<ClippedLinear> {
    let input_min = take_scalar(decoded, &format!("{prefix}.input_min"))?;
    let input_max = take_scalar(decoded, &format!("{prefix}.input_max"))?;
    let output_min = take_scalar(decoded, &format!("{prefix}.output_min"))?;
    let output_max = take_scalar(decoded, &format!("{prefix}.output_max"))?;
    if input_min > input_max || output_min > output_max {
        return Err(Error::corrupt(format!(
            "audio clipping bounds for {prefix} are inverted"
        )));
    }
    Ok(ClippedLinear {
        weight: take(decoded, &format!("{prefix}.linear.weight"))?,
        input_min,
        input_max,
        output_min,
        output_max,
    })
}

fn take(decoded: &mut std::collections::BTreeMap<String, Tensor>, name: &str) -> Result<Vec<f32>> {
    let tensor = decoded
        .remove(name)
        .ok_or_else(|| Error::internal(format!("decoded tensor {name} is missing")))?;
    Ok(tensor.into_parts().1)
}

fn take_scalar(
    decoded: &mut std::collections::BTreeMap<String, Tensor>,
    name: &str,
) -> Result<f32> {
    let tensor = decoded
        .remove(name)
        .ok_or_else(|| Error::internal(format!("decoded scalar {name} is missing")))?;
    let (shape, values) = tensor.into_parts();
    if !shape.is_scalar() || values.len() != 1 {
        return Err(Error::internal(format!("{name} is not a scalar")));
    }
    Ok(values[0])
}

/// Result of one audio tower pass.
#[derive(Debug)]
pub struct AudioOutput {
    pooled: Vec<f32>,
    hidden_states: Vec<Tensor>,
    sscp_output: Option<Tensor>,
    audio_tokens: usize,
    text_hidden: usize,
}

impl AudioOutput {
    /// Projected audio soft tokens into the text hidden space, row-major.
    pub fn pooled(&self) -> &[f32] {
        &self.pooled
    }

    /// Captured encoder layer states over valid audio tokens.
    pub fn hidden_states(&self) -> &[Tensor] {
        &self.hidden_states
    }

    /// Captured subsampling output over every token, including padding.
    pub fn sscp_output(&self) -> Option<&Tensor> {
        self.sscp_output.as_ref()
    }

    /// Number of valid audio soft tokens.
    pub fn audio_tokens(&self) -> usize {
        self.audio_tokens
    }

    /// Width of each projected soft token.
    pub fn text_hidden(&self) -> usize {
        self.text_hidden
    }
}

/// FP32 audio tower and its projection into the text hidden space.
#[derive(Debug)]
pub struct AudioModel {
    config: AudioConfig,
    weights: AudioWeights,
    text_hidden: usize,
}

impl AudioModel {
    /// Builds an audio model from a validated configuration and weight set.
    pub fn new(config: AudioConfig, weights: AudioWeights, text_hidden: usize) -> Result<Self> {
        if weights.layer_count() != config.num_hidden_layers() {
            return Err(Error::internal(format!(
                "audio weight set has {} layers, configuration declares {}",
                weights.layer_count(),
                config.num_hidden_layers()
            )));
        }
        let expected_projection = config
            .output_proj_dims()
            .checked_mul(text_hidden)
            .ok_or_else(|| Error::limit_exceeded("audio projection size overflow"))?;
        if weights.embedding_projection.len() != expected_projection {
            return Err(Error::internal(format!(
                "audio projection has {} elements, expected {expected_projection}",
                weights.embedding_projection.len()
            )));
        }
        Ok(Self {
            config,
            weights,
            text_hidden,
        })
    }

    /// Audio configuration.
    pub fn config(&self) -> &AudioConfig {
        &self.config
    }

    /// SHA-256 fingerprint of the source weight file.
    pub fn fingerprint_hex(&self) -> &str {
        self.weights.fingerprint_hex()
    }

    /// Number of FP32 audio parameters loaded in memory.
    pub fn parameters(&self) -> usize {
        self.weights.parameters()
    }

    /// Runs the audio tower and returns the projected soft tokens.
    pub fn forward(&self, mel: &MelSpectrogram, capture_hidden: bool) -> Result<AudioOutput> {
        let hidden_size = self.config.hidden_size();
        let frames = mel.frames();
        if mel.features().len() != frames * FEATURE_SIZE {
            return Err(Error::internal(
                "audio mel features do not match the frame count",
            ));
        }
        if mel.mask().len() != frames {
            return Err(Error::internal(
                "audio mel mask does not match the frame count",
            ));
        }

        let (sscp, output_mask) = self.subsample(mel)?;
        let tokens = output_mask.iter().filter(|value| **value).count();
        if tokens == 0 {
            return Err(Error::corrupt(
                "audio tower received no valid subsampled token",
            ));
        }
        if tokens > sscp.len() / hidden_size {
            return Err(Error::internal(
                "audio subsampling mask exceeds the token count",
            ));
        }

        let positions = self.config.position_count();
        let position_embeddings = self.position_embeddings(positions)?;
        let mut hidden = sscp;
        let mut hidden_states = Vec::new();
        let mut sscp_output = None;
        if capture_hidden {
            let selected = select_valid(&hidden, &output_mask, hidden_size)?;
            sscp_output = Some(tensor_from(tokens, hidden_size, &selected)?);
        }

        for layer in &self.weights.layers {
            hidden = feed_forward(
                &hidden,
                &layer.feed_forward1,
                self.config.hidden_size(),
                self.config.rms_norm_eps(),
                self.config.gradient_clipping(),
                self.config.residual_weight(),
            )?;
            let residual = hidden.clone();
            hidden = rms_norm_rows_checked(
                &hidden,
                &layer.norm_pre_attn,
                hidden.len() / hidden_size,
                hidden_size,
                self.config.rms_norm_eps(),
            )?;
            let relative_keys = kernels::linear(
                &position_embeddings,
                &layer.attention.relative_k_proj,
                positions,
                hidden_size,
                hidden_size,
            )?;
            let attention = chunked_attention(
                &hidden,
                &output_mask,
                &layer.attention,
                &self.config,
                &relative_keys,
            )?;
            let attention = rms_norm_rows_checked(
                &attention,
                &layer.norm_post_attn,
                attention.len() / hidden_size,
                hidden_size,
                self.config.rms_norm_eps(),
            )?;
            hidden = kernels::add(&residual, &attention)?;
            hidden = light_conv(
                &hidden,
                &layer.light_conv,
                self.config.hidden_size(),
                self.config.rms_norm_eps(),
                self.config.gradient_clipping(),
            )?;
            hidden = feed_forward(
                &hidden,
                &layer.feed_forward2,
                self.config.hidden_size(),
                self.config.rms_norm_eps(),
                self.config.gradient_clipping(),
                self.config.residual_weight(),
            )?;
            hidden = rms_norm_rows_checked(
                &hidden,
                &layer.norm_out,
                hidden.len() / hidden_size,
                hidden_size,
                self.config.rms_norm_eps(),
            )?;
            if capture_hidden {
                let selected = select_valid(&hidden, &output_mask, hidden_size)?;
                hidden_states.push(tensor_from(tokens, hidden_size, &selected)?);
            }
        }

        let rows = hidden.len() / hidden_size;
        let projected = kernels::linear(
            &hidden,
            &self.weights.output_proj,
            rows,
            hidden_size,
            self.config.output_proj_dims(),
        )?;
        if self.weights.output_proj_bias.len() != self.config.output_proj_dims() {
            return Err(Error::internal(
                "audio output projection bias has an invalid length",
            ));
        }
        let mut projected = projected;
        for row in 0..rows {
            for (column, value) in projected
                [row * self.config.output_proj_dims()..(row + 1) * self.config.output_proj_dims()]
                .iter_mut()
                .enumerate()
            {
                *value += self.weights.output_proj_bias[column];
            }
        }
        if projected.iter().any(|value| !value.is_finite()) {
            return Err(Error::internal(
                "audio output projection produced a non-finite value",
            ));
        }
        let normalized = kernels::rms_norm_rows_unscaled(
            &projected,
            rows,
            self.config.output_proj_dims(),
            self.config.rms_norm_eps(),
        )?;
        let pooled_all = kernels::linear(
            &normalized,
            &self.weights.embedding_projection,
            rows,
            self.config.output_proj_dims(),
            self.text_hidden,
        )?;
        let pooled = select_valid(&pooled_all, &output_mask, self.text_hidden)?;
        Ok(AudioOutput {
            pooled,
            hidden_states,
            sscp_output,
            audio_tokens: tokens,
            text_hidden: self.text_hidden,
        })
    }

    fn subsample(&self, mel: &MelSpectrogram) -> Result<(Vec<f32>, Vec<bool>)> {
        let hidden = self.config.hidden_size();
        let [first, second] = self.config.conv_channels();
        let frames = mel.frames();
        let first_frames = frames.div_ceil(2);
        let second_frames = first_frames.div_ceil(2);
        let first_freq = FEATURE_SIZE.div_ceil(2);
        let second_freq = first_freq.div_ceil(2);
        if second_frames == 0 {
            return Err(Error::corrupt("audio subsampling produced no frames"));
        }
        let layer0_len = first_frames
            .checked_mul(first_freq)
            .and_then(|value| value.checked_mul(first))
            .ok_or_else(|| Error::limit_exceeded("audio subsampling layer 0 size overflow"))?;
        let layer1_len = second_frames
            .checked_mul(second_freq)
            .and_then(|value| value.checked_mul(second))
            .ok_or_else(|| Error::limit_exceeded("audio subsampling layer 1 size overflow"))?;
        if layer0_len > MAX_ACTIVATION_ELEMENTS || layer1_len > MAX_ACTIVATION_ELEMENTS {
            return Err(Error::limit_exceeded(format!(
                "audio subsampling would allocate more than the {MAX_ACTIVATION_ELEMENTS}-element cap"
            )));
        }

        let mut layer0 = vec![0.0f32; layer0_len];
        for out_t in 0..first_frames {
            for out_f in 0..first_freq {
                for out_c in 0..first {
                    let mut sum = 0.0f32;
                    for kernel_t in 0..3usize {
                        for kernel_f in 0..3usize {
                            let in_t = out_t as i64 * 2 + kernel_t as i64 - 1;
                            let in_f = out_f as i64 * 2 + kernel_f as i64 - 1;
                            if in_t < 0
                                || in_t >= frames as i64
                                || in_f < 0
                                || in_f >= FEATURE_SIZE as i64
                                || !mel.mask()[in_t as usize]
                            {
                                continue;
                            }
                            let value =
                                mel.features()[in_t as usize * FEATURE_SIZE + in_f as usize];
                            let weight =
                                self.weights.layer0_conv[out_c * 9 + kernel_t * 3 + kernel_f];
                            sum += value * weight;
                        }
                    }
                    layer0[(out_t * first_freq + out_f) * first + out_c] = sum;
                }
                normalize_subsample(
                    &mut layer0[(out_t * first_freq + out_f) * first
                        ..(out_t * first_freq + out_f + 1) * first],
                    &self.weights.layer0_norm,
                    self.config.rms_norm_eps(),
                )?;
            }
        }
        let mut mask0 = Vec::with_capacity(first_frames);
        for index in 0..first_frames {
            mask0.push(mel.mask().get(index * 2).copied().unwrap_or(false));
        }

        let mut layer1 = vec![0.0f32; layer1_len];
        for out_t in 0..second_frames {
            for out_f in 0..second_freq {
                for out_c in 0..second {
                    let mut sum = 0.0f32;
                    for in_c in 0..first {
                        for kernel_t in 0..3usize {
                            for kernel_f in 0..3usize {
                                let in_t = out_t as i64 * 2 + kernel_t as i64 - 1;
                                let in_f = out_f as i64 * 2 + kernel_f as i64 - 1;
                                if in_t < 0
                                    || in_t >= first_frames as i64
                                    || in_f < 0
                                    || in_f >= first_freq as i64
                                    || !mask0[in_t as usize]
                                {
                                    continue;
                                }
                                let value = layer0
                                    [(in_t as usize * first_freq + in_f as usize) * first + in_c];
                                let weight = self.weights.layer1_conv
                                    [((out_c * first + in_c) * 3 + kernel_t) * 3 + kernel_f];
                                sum += value * weight;
                            }
                        }
                    }
                    layer1[(out_t * second_freq + out_f) * second + out_c] = sum;
                }
                normalize_subsample(
                    &mut layer1[(out_t * second_freq + out_f) * second
                        ..(out_t * second_freq + out_f + 1) * second],
                    &self.weights.layer1_norm,
                    self.config.rms_norm_eps(),
                )?;
            }
        }
        let mut mask1 = Vec::with_capacity(second_frames);
        for index in 0..second_frames {
            mask1.push(mask0.get(index * 2).copied().unwrap_or(false));
        }

        let projection_input = self.config.subsampling_projection_input();
        if layer1.len() != second_frames * projection_input {
            return Err(Error::internal(
                "audio subsampling output does not match the projection input width",
            ));
        }
        let hidden = kernels::linear(
            &layer1,
            &self.weights.input_proj_linear,
            second_frames,
            projection_input,
            hidden,
        )?;
        Ok((hidden, mask1))
    }

    fn position_embeddings(&self, positions: usize) -> Result<Vec<f32>> {
        let hidden = self.config.hidden_size();
        if !hidden.is_multiple_of(2) {
            return Err(Error::internal(
                "audio hidden size must be even for sinusoidal positions",
            ));
        }
        let timescales = hidden / 2;
        if timescales == 0 {
            return Err(Error::internal("audio position encoding has no timescales"));
        }
        let increment = if timescales > 1 {
            (10_000.0f64).ln() / (timescales - 1) as f64
        } else {
            0.0
        };
        let mut inverse = Vec::with_capacity(timescales);
        for index in 0..timescales {
            inverse.push((-(index as f64) * increment).exp() as f32);
        }
        let mut positions_embedding = vec![0.0f32; positions * hidden];
        for position in 0..positions {
            let position_id = (positions - 1 - position) as f32;
            for (index, inverse_value) in inverse.iter().enumerate() {
                let scaled = position_id * inverse_value;
                positions_embedding[position * hidden + index] = scaled.sin();
                positions_embedding[position * hidden + timescales + index] = scaled.cos();
            }
        }
        Ok(positions_embedding)
    }
}

fn normalize_subsample(values: &mut [f32], weight: &[f32], epsilon: f32) -> Result<()> {
    let width = values.len();
    if width == 0 || weight.len() != width {
        return Err(Error::internal(
            "audio subsampling norm has an invalid width",
        ));
    }
    let mut sum = 0.0f32;
    for value in values.iter() {
        if !value.is_finite() {
            return Err(Error::internal(
                "audio subsampling produced a non-finite value",
            ));
        }
        sum += *value;
    }
    let mean = sum / width as f32;
    let mut variance = 0.0f32;
    for value in values.iter() {
        let difference = *value - mean;
        variance += difference * difference;
    }
    variance /= width as f32;
    let inverse = 1.0 / (variance + epsilon).sqrt();
    if !inverse.is_finite() {
        return Err(Error::internal("audio subsampling norm is not finite"));
    }
    for (value, scale) in values.iter_mut().zip(weight) {
        *value = (*value - mean) * inverse * scale;
        *value = value.max(0.0);
    }
    Ok(())
}

fn rms_norm_rows_checked(
    input: &[f32],
    weight: &[f32],
    rows: usize,
    width: usize,
    epsilon: f32,
) -> Result<Vec<f32>> {
    if input.len() != rows * width {
        return Err(Error::internal(
            "audio RMSNorm input does not match the row layout",
        ));
    }
    kernels::rms_norm_rows(input, weight, rows, width, epsilon)
}

fn clamp_value(value: f32, minimum: f32, maximum: f32) -> f32 {
    if value < minimum {
        minimum
    } else if value > maximum {
        maximum
    } else {
        value
    }
}

fn silu_value(value: f32) -> f32 {
    if value >= 0.0 {
        value / (1.0 + (-value).exp())
    } else {
        value * value.exp() / (1.0 + value.exp())
    }
}

fn sigmoid_value(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

fn softplus_value(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        (1.0 + value.exp()).ln()
    }
}

fn feed_forward(
    hidden: &[f32],
    weights: &FeedForwardWeights,
    hidden_size: usize,
    epsilon: f32,
    gradient_clipping: f32,
    residual_weight: f32,
) -> Result<Vec<f32>> {
    let rows = hidden.len() / hidden_size;
    if rows == 0 || hidden.len() != rows * hidden_size {
        return Err(Error::internal(
            "audio feed-forward input has an invalid layout",
        ));
    }
    let residual = hidden;
    let clamped = clamp_if_needed(hidden, gradient_clipping);
    let normalized = rms_norm_rows_checked(
        &clamped,
        &weights.pre_layer_norm,
        rows,
        hidden_size,
        epsilon,
    )?;
    let first = weights.ffw_layer_1.apply(&normalized, rows, hidden_size)?;
    let mut activated = Vec::with_capacity(first.len());
    for value in &first {
        let silu = silu_value(*value);
        if !silu.is_finite() {
            return Err(Error::internal(
                "audio feed-forward activation is not finite",
            ));
        }
        activated.push(silu);
    }
    let intermediate = weights.ffw_layer_1.weight.len() / hidden_size;
    if intermediate == 0 || weights.ffw_layer_1.weight.len() != intermediate * hidden_size {
        return Err(Error::internal(
            "audio feed-forward first projection has an invalid width",
        ));
    }
    let second = weights.ffw_layer_2.apply(&activated, rows, intermediate)?;
    let clamped = clamp_if_needed(&second, gradient_clipping);
    let normalized = rms_norm_rows_checked(
        &clamped,
        &weights.post_layer_norm,
        rows,
        hidden_size,
        epsilon,
    )?;
    let mut output = Vec::with_capacity(hidden.len());
    for (residual_value, value) in residual.iter().zip(&normalized) {
        let scaled = *value * residual_weight;
        let sum = *residual_value + scaled;
        if !sum.is_finite() {
            return Err(Error::internal("audio feed-forward residual is not finite"));
        }
        output.push(sum);
    }
    Ok(output)
}

fn clamp_if_needed(values: &[f32], bound: f32) -> Cow<'_, [f32]> {
    if values.iter().any(|value| value.abs() > bound) {
        Cow::Owned(
            values
                .iter()
                .map(|value| clamp_value(*value, -bound, bound))
                .collect(),
        )
    } else {
        Cow::Borrowed(values)
    }
}

fn light_conv(
    hidden: &[f32],
    weights: &LightConvWeights,
    hidden_size: usize,
    epsilon: f32,
    gradient_clipping: f32,
) -> Result<Vec<f32>> {
    let rows = hidden.len() / hidden_size;
    if rows == 0 || hidden.len() != rows * hidden_size {
        return Err(Error::internal(
            "audio light-conv input has an invalid layout",
        ));
    }
    let residual = hidden;
    let normalized =
        rms_norm_rows_checked(hidden, &weights.pre_layer_norm, rows, hidden_size, epsilon)?;
    let start = weights.linear_start.apply(&normalized, rows, hidden_size)?;
    if start.len() != rows * hidden_size * 2 {
        return Err(Error::internal(
            "audio light-conv start projection has an invalid length",
        ));
    }
    let mut gated = vec![0.0f32; rows * hidden_size];
    for row in 0..rows {
        for column in 0..hidden_size {
            let first = start[row * hidden_size * 2 + column];
            let second = start[row * hidden_size * 2 + hidden_size + column];
            let value = first * sigmoid_value(second);
            if !value.is_finite() {
                return Err(Error::internal(
                    "audio light-conv gate produced a non-finite value",
                ));
            }
            gated[row * hidden_size + column] = value;
        }
    }

    let kernel = weights.depthwise_conv1d.len() / hidden_size;
    if kernel == 0 || weights.depthwise_conv1d.len() != hidden_size * kernel {
        return Err(Error::internal(
            "audio depthwise convolution has an invalid weight layout",
        ));
    }
    let mut convolved = vec![0.0f32; rows * hidden_size];
    for row in 0..rows {
        for column in 0..hidden_size {
            let mut sum = 0.0f32;
            for tap in 0..kernel {
                let source = row as i64 - (kernel as i64 - 1) + tap as i64;
                if source < 0 || source >= rows as i64 {
                    continue;
                }
                let weight = weights.depthwise_conv1d[column * kernel + tap];
                sum += gated[source as usize * hidden_size + column] * weight;
            }
            convolved[row * hidden_size + column] = sum;
        }
    }
    let clamped = clamp_if_needed(&convolved, gradient_clipping);
    let normalized =
        rms_norm_rows_checked(&clamped, &weights.conv_norm, rows, hidden_size, epsilon)?;
    let mut activated = Vec::with_capacity(normalized.len());
    for value in &normalized {
        let silu = silu_value(*value);
        if !silu.is_finite() {
            return Err(Error::internal("audio light-conv activation is not finite"));
        }
        activated.push(silu);
    }
    let end = weights.linear_end.apply(&activated, rows, hidden_size)?;
    let mut output = Vec::with_capacity(hidden.len());
    for (residual_value, value) in residual.iter().zip(&end) {
        let sum = *residual_value + *value;
        if !sum.is_finite() {
            return Err(Error::internal("audio light-conv residual is not finite"));
        }
        output.push(sum);
    }
    Ok(output)
}

fn chunked_attention(
    hidden: &[f32],
    mask: &[bool],
    weights: &AttentionWeights,
    config: &AudioConfig,
    relative_keys: &[f32],
) -> Result<Vec<f32>> {
    let hidden_size = config.hidden_size();
    let heads = config.num_attention_heads();
    let head_dim = config.head_dim();
    let rows = hidden.len() / hidden_size;
    if rows == 0 || hidden.len() != rows * hidden_size || mask.len() != rows {
        return Err(Error::internal(
            "audio attention input has an invalid layout",
        ));
    }
    let chunk = config.attention_chunk_size();
    let context = config.context_size();
    let positions = config.position_count();
    let max_past = config.attention_context_left() - 1;
    let left_window = config.attention_context_left() - 1;
    let q_scale = ((head_dim as f64).powf(-0.5) / std::f64::consts::LN_2) as f32;
    let k_scale = ((1.0 + std::f64::consts::E).ln() / std::f64::consts::LN_2) as f32;
    let softcap = config.attention_logit_cap();
    let invalid = -1.0e9f32;

    let mut query = weights.q_proj.apply(hidden, rows, hidden_size)?;
    let mut key = weights.k_proj.apply(hidden, rows, hidden_size)?;
    let value = weights.v_proj.apply(hidden, rows, hidden_size)?;
    if weights.per_dim_scale.len() != head_dim {
        return Err(Error::internal(
            "audio per-dimension scale has an invalid length",
        ));
    }
    for row in 0..rows {
        for head in 0..heads {
            for dimension in 0..head_dim {
                let index = row * hidden_size + head * head_dim + dimension;
                let scale = softplus_value(weights.per_dim_scale[dimension]);
                if !scale.is_finite() {
                    return Err(Error::internal("audio per-dimension scale is not finite"));
                }
                query[index] *= q_scale * scale;
                key[index] *= k_scale;
            }
        }
    }

    let blocks = rows.div_ceil(chunk);
    let mut attention = vec![0.0f32; rows * hidden_size];
    let mut scores = vec![0.0f32; heads * context];
    let mut denominators = vec![0.0f32; heads];
    for block in 0..blocks {
        for offset in 0..chunk {
            let query_index = block * chunk + offset;
            if query_index >= rows || !mask[query_index] {
                continue;
            }
            for head in 0..heads {
                let mut maximum = f32::NEG_INFINITY;
                for context_index in 0..context {
                    let key_index =
                        block as i64 * chunk as i64 + context_index as i64 - max_past as i64;
                    let valid = key_index >= 0
                        && key_index < rows as i64
                        && mask[key_index as usize]
                        && query_index as i64 - key_index >= 0
                        && query_index as i64 - key_index < left_window as i64;
                    let score = if valid {
                        let key_index = key_index as usize;
                        let mut dot = 0.0f32;
                        for dimension in 0..head_dim {
                            dot += query[query_index * hidden_size + head * head_dim + dimension]
                                * key[key_index * hidden_size + head * head_dim + dimension];
                        }
                        let relative = context_index as i64 - offset as i64;
                        if relative >= 0 && (relative as usize) < positions {
                            let relative_index = relative as usize;
                            for dimension in 0..head_dim {
                                dot += query
                                    [query_index * hidden_size + head * head_dim + dimension]
                                    * relative_keys
                                        [(relative_index * heads + head) * head_dim + dimension];
                            }
                        }
                        (dot / softcap).tanh() * softcap
                    } else {
                        invalid
                    };
                    scores[head * context + context_index] = score;
                    maximum = maximum.max(score);
                }
                if maximum <= invalid / 2.0 {
                    denominators[head] = 0.0;
                    continue;
                }
                let mut sum = 0.0f32;
                for context_index in 0..context {
                    let value = (scores[head * context + context_index] - maximum).exp();
                    scores[head * context + context_index] = value;
                    sum += value;
                }
                if !sum.is_finite() || sum <= 0.0 {
                    return Err(Error::internal(
                        "audio attention softmax has a non-finite denominator",
                    ));
                }
                denominators[head] = sum;
            }
            for head in 0..heads {
                let sum = denominators[head];
                if sum == 0.0 {
                    continue;
                }
                for context_index in 0..context {
                    let weight = scores[head * context + context_index] / sum;
                    if weight == 0.0 {
                        continue;
                    }
                    let key_index =
                        block as i64 * chunk as i64 + context_index as i64 - max_past as i64;
                    if key_index < 0 || key_index >= rows as i64 {
                        continue;
                    }
                    let key_index = key_index as usize;
                    for dimension in 0..head_dim {
                        attention[query_index * hidden_size + head * head_dim + dimension] +=
                            weight * value[key_index * hidden_size + head * head_dim + dimension];
                    }
                }
            }
        }
    }
    if attention.iter().any(|value| !value.is_finite()) {
        return Err(Error::internal(
            "audio attention produced a non-finite value",
        ));
    }
    let post = weights.post.apply(&attention, rows, hidden_size)?;
    Ok(post)
}

fn select_valid(values: &[f32], mask: &[bool], width: usize) -> Result<Vec<f32>> {
    let rows = mask.len();
    if values.len() != rows * width {
        return Err(Error::internal(
            "audio selection input does not match the mask layout",
        ));
    }
    let mut selected = Vec::with_capacity(mask.iter().filter(|value| **value).count() * width);
    for (row, valid) in mask.iter().enumerate() {
        if *valid {
            selected.extend_from_slice(&values[row * width..(row + 1) * width]);
        }
    }
    Ok(selected)
}

fn tensor_from(rows: usize, width: usize, data: &[f32]) -> Result<Tensor> {
    let shape = Shape::new(vec![rows, width])?;
    Tensor::from_slice(shape, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_a_configuration_without_audio() {
        let value = json::parse(r#"{"text_config":{}}"#).expect("json");
        let error = AudioConfig::from_json_value(&value).expect_err("missing audio config");
        assert!(matches!(error, Error::Unsupported(_)));
    }

    #[test]
    fn rejects_float16_audio() {
        let value = json::parse(
            r#"{"audio_config":{"model_type":"gemma4_audio","dtype":"float16","hidden_size":16,"num_hidden_layers":1,"num_attention_heads":2,"conv_kernel_size":5,"subsampling_conv_channels":[8,8],"attention_chunk_size":2,"attention_context_left":2,"attention_context_right":0,"attention_logit_cap":50.0,"rms_norm_eps":1e-6,"output_proj_dims":16,"residual_weight":0.5,"gradient_clipping":1e10,"hidden_act":"silu","use_clipped_linears":true}}"#,
        )
        .expect("json");
        let error = AudioConfig::from_json_value(&value).expect_err("float16 must fail");
        assert!(matches!(error, Error::Unsupported(_)));
    }

    #[test]
    fn computes_the_frozen_context_and_position_count() {
        let value = json::parse(
            r#"{"audio_config":{"model_type":"gemma4_audio","dtype":"bfloat16","hidden_size":1024,"num_hidden_layers":12,"num_attention_heads":8,"conv_kernel_size":5,"subsampling_conv_channels":[128,32],"attention_chunk_size":12,"attention_context_left":13,"attention_context_right":0,"attention_logit_cap":50.0,"rms_norm_eps":1e-6,"output_proj_dims":1536,"residual_weight":0.5,"gradient_clipping":1e10,"hidden_act":"silu","use_clipped_linears":true}}"#,
        )
        .expect("json");
        let config = AudioConfig::from_json_value(&value).expect("audio config");
        assert_eq!(config.hidden_size(), 1024);
        assert_eq!(config.head_dim(), 128);
        assert_eq!(config.context_size(), 24);
        assert_eq!(config.position_count(), 13);
        assert_eq!(config.subsampling_projection_input(), 1024);
        assert_eq!(config.tensor_shapes(512).len(), 8 + 12 * 62);
    }

    #[test]
    fn silu_and_softplus_are_finite_at_extremes() {
        for value in [-100.0f32, -1.0, 0.0, 1.0, 100.0] {
            assert!(silu_value(value).is_finite());
            assert!(softplus_value(value).is_finite());
            assert!(sigmoid_value(value).is_finite());
        }
    }
}
