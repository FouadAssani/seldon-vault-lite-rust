//! Strict parser for the EmbeddingGemma 2 text configuration.
//!
//! The engine accepts only the architecture it implements. A configuration
//! with a different activation, a different attention layout, FP16, or an
//! inconsistent per-layer head configuration is rejected instead of being
//! approximated: silently running a different architecture would produce
//! embeddings that are not comparable with the reference.
//!
//! Every dimension is validated against an explicit cap and every product that
//! feeds a weight shape uses checked arithmetic.

use std::path::Path;

use crate::binary::read_file_bounded;
use crate::error::{Error, Result};
use crate::json::{self, Limits, Number, Object, Value};
use crate::model::source::ModelFiles;

/// Maximum size of a configuration file.
pub const MAX_CONFIG_BYTES: u64 = 1024 * 1024;

/// Maximum number of text tokens accepted by the engine.
///
/// The model card documents an 8,192-token context. The configuration allows
/// longer positions, but the product refuses to run beyond the documented
/// limit until a longer-context parity corpus exists.
pub const MAX_TEXT_TOKENS: usize = 8_192;

/// Maximum hidden size.
pub const MAX_HIDDEN_SIZE: usize = 16_384;

/// Maximum number of transformer layers.
pub const MAX_LAYERS: usize = 128;

/// Maximum number of attention heads.
pub const MAX_HEADS: usize = 128;

/// Maximum head dimension.
pub const MAX_HEAD_DIM: usize = 1_024;

/// Maximum feed-forward intermediate size.
pub const MAX_INTERMEDIATE_SIZE: usize = 65_536;

/// Maximum vocabulary size.
pub const MAX_VOCAB_SIZE: usize = 1_000_000;

/// Maximum embedding dimension.
pub const MAX_EMBEDDING_DIM: usize = 16_384;

/// Maximum position count.
pub const MAX_POSITIONS: usize = 1_048_576;

/// Attention pattern of one layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    /// Bidirectional sliding-window attention with a radius of
    /// `sliding_window`.
    SlidingAttention,
    /// Bidirectional full attention.
    FullAttention,
}

/// Per-layer attention configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerConfig {
    /// Attention pattern of this layer.
    pub layer_type: LayerType,
    /// Dimension of one query/key/value head.
    pub head_dim: usize,
    /// Number of key/value heads before grouped-query expansion.
    pub num_key_value_heads: usize,
}

/// Validated text model configuration.
#[derive(Debug, Clone)]
pub struct TextConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    hidden_size_per_layer_input: usize,
    embedding_dim: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f32,
    sliding_window: usize,
    pad_token_id: Option<u32>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    embedding_scale: f32,
    ple_scale: f32,
    layers: Vec<LayerConfig>,
    full_rope_theta: f32,
    sliding_rope_theta: f32,
}

impl TextConfig {
    /// Parses a configuration from JSON text.
    pub fn from_json_str(input: &str) -> Result<Self> {
        let limits = Limits {
            max_input_bytes: MAX_CONFIG_BYTES as usize,
            ..Limits::default()
        };
        let value = json::parse_with_limits(input, &limits)?;
        Self::from_json_value(&value)
    }

    /// Parses a configuration from an already parsed JSON value.
    ///
    /// The top-level object may either be the text configuration itself or the
    /// composite model configuration with a `text_config` member.
    pub fn from_json_value(value: &Value) -> Result<Self> {
        let root = value
            .as_object()
            .ok_or_else(|| Error::corrupt("model configuration must be a JSON object"))?;
        let text = match root.get("text_config") {
            Some(nested) => nested
                .as_object()
                .ok_or_else(|| Error::corrupt("text_config must be a JSON object"))?,
            None => root,
        };

        if let Some(model_type) = text.get("model_type") {
            let model_type = model_type
                .as_str()
                .ok_or_else(|| Error::corrupt("model_type must be a string"))?;
            if model_type != "embedding_gemma2_text" {
                return Err(Error::unsupported(format!(
                    "text model type {model_type} is not embedding_gemma2_text"
                )));
            }
        }

        if let Some(dtype) = text.get("dtype") {
            let dtype = dtype
                .as_str()
                .ok_or_else(|| Error::corrupt("dtype must be a string"))?;
            match dtype {
                "bfloat16" | "float32" => {}
                "float16" => {
                    return Err(Error::unsupported(
                        "float16 is forbidden for EmbeddingGemma 2 activations",
                    ));
                }
                other => {
                    return Err(Error::unsupported(format!(
                        "model dtype {other} is not supported"
                    )));
                }
            }
        }

        let hidden_size = unsigned(text, "hidden_size")?;
        let intermediate_size = unsigned(text, "intermediate_size")?;
        let num_hidden_layers = unsigned(text, "num_hidden_layers")?;
        let num_attention_heads = unsigned(text, "num_attention_heads")?;
        let global_num_key_value_heads = unsigned(text, "num_key_value_heads")?;
        let global_head_dim = unsigned(text, "head_dim")?;
        let hidden_size_per_layer_input = unsigned(text, "hidden_size_per_layer_input")?;
        let embedding_dim = unsigned(text, "embedding_dim")?;
        let vocab_size = unsigned(text, "vocab_size")?;
        let max_position_embeddings = unsigned(text, "max_position_embeddings")?;
        let sliding_window = unsigned(text, "sliding_window")?;
        let rms_norm_eps = float(text, "rms_norm_eps")?;
        let rms_norm_eps_f32 = rms_norm_eps as f32;
        if !rms_norm_eps_f32.is_finite() {
            return Err(Error::corrupt(
                "RMSNorm epsilon does not fit a finite f32 value",
            ));
        }

        let hidden_activation = string(text, "hidden_activation")?;
        if hidden_activation != "gelu_pytorch_tanh" {
            return Err(Error::unsupported(format!(
                "hidden activation {hidden_activation} is not gelu_pytorch_tanh"
            )));
        }

        let attention_bias = optional_bool(text, "attention_bias")?.unwrap_or(false);
        if attention_bias {
            return Err(Error::unsupported(
                "attention bias is not part of the frozen model",
            ));
        }
        if let Some(dropout) = optional_float(text, "attention_dropout")?
            && dropout != 0.0
        {
            return Err(Error::unsupported(
                "attention dropout is not implemented by the inference engine",
            ));
        }

        validate_caps(&ConfigNumbers {
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads: global_num_key_value_heads,
            head_dim: global_head_dim,
            hidden_size_per_layer_input,
            embedding_dim,
            vocab_size,
            max_position_embeddings,
            sliding_window,
            rms_norm_eps,
        })?;

        let layer_types = parse_layer_types(text, num_hidden_layers)?;
        let overrides = parse_per_layer_config(text, num_hidden_layers)?;
        let mut layers = Vec::with_capacity(num_hidden_layers);
        for (index, layer_type) in layer_types.into_iter().enumerate() {
            let (head_dim, num_key_value_heads) = match overrides.get(&index) {
                Some(overridden) => (overridden.head_dim, overridden.num_key_value_heads),
                None => (global_head_dim, global_num_key_value_heads),
            };
            validate_head_layout(num_attention_heads, num_key_value_heads, head_dim)?;
            layers.push(LayerConfig {
                layer_type,
                head_dim,
                num_key_value_heads,
            });
        }

        validate_uniform_head_dim(&layers)?;
        let (full_rope_theta, sliding_rope_theta) = parse_rope_parameters(text)?;

        let pad_token_id = optional_token_id(text, "pad_token_id")?;
        let bos_token_id = optional_token_id(text, "bos_token_id")?;
        let eos_token_id = optional_token_id(text, "eos_token_id")?;
        for id in [pad_token_id, bos_token_id, eos_token_id]
            .into_iter()
            .flatten()
        {
            if id as usize >= vocab_size {
                return Err(Error::corrupt(format!(
                    "special token id {id} is outside the vocabulary of {vocab_size} entries"
                )));
            }
        }

        let embedding_scale = (hidden_size as f32).sqrt();
        let ple_scale = 1.0 / embedding_scale;
        if !embedding_scale.is_finite() || embedding_scale <= 0.0 {
            return Err(Error::corrupt(
                "hidden size does not yield a finite embedding scale",
            ));
        }

        Ok(Self {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            hidden_size_per_layer_input,
            embedding_dim,
            max_position_embeddings,
            rms_norm_eps: rms_norm_eps_f32,
            sliding_window,
            pad_token_id,
            bos_token_id,
            eos_token_id,
            embedding_scale,
            ple_scale,
            layers,
            full_rope_theta,
            sliding_rope_theta,
        })
    }

    /// Loads a configuration file with an explicit size cap.
    pub fn from_file(path: &Path) -> Result<Self> {
        let bytes = read_file_bounded(path, MAX_CONFIG_BYTES)?;
        let text = std::str::from_utf8(&bytes)
            .map_err(|_| Error::corrupt(format!("{} is not valid UTF-8", path.display())))?;
        Self::from_json_str(text)
    }

    /// Loads config.json from a directory or bundle source.
    pub fn from_files(files: &ModelFiles<'_>) -> Result<Self> {
        let text = files.read_utf8("config.json", MAX_CONFIG_BYTES)?;
        Self::from_json_str(&text)
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Hidden size of the transformer.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Feed-forward intermediate size.
    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }

    /// Number of transformer layers.
    pub fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }

    /// Number of query heads.
    pub fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }

    /// Dimension of the per-layer input signal.
    pub fn hidden_size_per_layer_input(&self) -> usize {
        self.hidden_size_per_layer_input
    }

    /// Output embedding dimension after the final projection.
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Maximum number of positions declared by the model.
    pub fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }

    /// RMSNorm epsilon.
    pub fn rms_norm_eps(&self) -> f32 {
        self.rms_norm_eps
    }

    /// Inclusive sliding-window radius.
    pub fn sliding_window(&self) -> usize {
        self.sliding_window
    }

    /// Padding token id, when declared.
    pub fn pad_token_id(&self) -> Option<u32> {
        self.pad_token_id
    }

    /// Beginning-of-sequence token id, when declared.
    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    /// End-of-sequence token id, when declared.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    /// Multiplicative scale applied to the token embedding table.
    pub fn embedding_scale(&self) -> f32 {
        self.embedding_scale
    }

    /// Multiplicative scale applied to the per-layer projection.
    pub fn ple_scale(&self) -> f32 {
        self.ple_scale
    }

    /// Per-layer attention configurations.
    pub fn layers(&self) -> &[LayerConfig] {
        &self.layers
    }

    /// RoPE inverse frequencies for a layer type.
    pub fn rope_inv_freq(&self, layer_type: LayerType) -> Result<Vec<f32>> {
        let (theta, head_dim) = match layer_type {
            LayerType::FullAttention => (self.full_rope_theta, self.full_head_dim()),
            LayerType::SlidingAttention => (self.sliding_rope_theta, self.sliding_head_dim()),
        };
        crate::tensor::kernels::rope_inv_freq(head_dim, theta)
    }

    fn full_head_dim(&self) -> usize {
        self.layers
            .iter()
            .find(|layer| layer.layer_type == LayerType::FullAttention)
            .map(|layer| layer.head_dim)
            .unwrap_or(0)
    }

    fn sliding_head_dim(&self) -> usize {
        self.layers
            .iter()
            .find(|layer| layer.layer_type == LayerType::SlidingAttention)
            .map(|layer| layer.head_dim)
            .unwrap_or(0)
    }

    /// Expected tensor names and shapes for the text weights.
    ///
    /// The loader uses this inventory to reject missing, extra, or
    /// shape-incompatible tensors before running any computation.
    pub fn tensor_shapes(&self) -> Vec<(String, Vec<usize>)> {
        let hidden = self.hidden_size;
        let ple = self.hidden_size_per_layer_input;
        let intermediate = self.intermediate_size;
        let mut shapes = Vec::with_capacity(5 + self.num_hidden_layers * 17);
        shapes.push((
            "language_model.embed_tokens.weight".to_owned(),
            vec![self.vocab_size, hidden],
        ));
        shapes.push((
            "language_model.embedding_projection.weight".to_owned(),
            vec![self.embedding_dim, hidden],
        ));
        shapes.push(("language_model.norm.weight".to_owned(), vec![hidden]));
        shapes.push((
            "language_model.ple.per_layer_model_projection.weight".to_owned(),
            vec![self.num_hidden_layers * ple, hidden],
        ));
        shapes.push((
            "language_model.ple.per_layer_projection_norm.weight".to_owned(),
            vec![ple],
        ));
        for (index, layer) in self.layers.iter().enumerate() {
            let prefix = format!("language_model.layers.{index}");
            let q_dim = self.num_attention_heads * layer.head_dim;
            let kv_dim = layer.num_key_value_heads * layer.head_dim;
            shapes.push((format!("{prefix}.input_layernorm.weight"), vec![hidden]));
            shapes.push((
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![hidden],
            ));
            shapes.push((
                format!("{prefix}.pre_feedforward_layernorm.weight"),
                vec![hidden],
            ));
            shapes.push((
                format!("{prefix}.post_feedforward_layernorm.weight"),
                vec![hidden],
            ));
            shapes.push((format!("{prefix}.layer_scalar"), vec![1]));
            shapes.push((
                format!("{prefix}.mlp.gate_proj.weight"),
                vec![intermediate, hidden],
            ));
            shapes.push((
                format!("{prefix}.mlp.up_proj.weight"),
                vec![intermediate, hidden],
            ));
            shapes.push((
                format!("{prefix}.mlp.down_proj.weight"),
                vec![hidden, intermediate],
            ));
            shapes.push((
                format!("{prefix}.ple_block.per_layer_input_gate.weight"),
                vec![ple, hidden],
            ));
            shapes.push((
                format!("{prefix}.ple_block.per_layer_projection.weight"),
                vec![hidden, ple],
            ));
            shapes.push((
                format!("{prefix}.ple_block.post_per_layer_input_norm.weight"),
                vec![hidden],
            ));
            shapes.push((
                format!("{prefix}.self_attn.q_proj.weight"),
                vec![q_dim, hidden],
            ));
            shapes.push((
                format!("{prefix}.self_attn.k_proj.weight"),
                vec![kv_dim, hidden],
            ));
            shapes.push((
                format!("{prefix}.self_attn.v_proj.weight"),
                vec![kv_dim, hidden],
            ));
            shapes.push((
                format!("{prefix}.self_attn.o_proj.weight"),
                vec![hidden, q_dim],
            ));
            shapes.push((
                format!("{prefix}.self_attn.q_norm.weight"),
                vec![layer.head_dim],
            ));
            shapes.push((
                format!("{prefix}.self_attn.k_norm.weight"),
                vec![layer.head_dim],
            ));
        }
        shapes
    }
}

#[derive(Debug, Clone, Copy)]
struct ConfigNumbers {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    hidden_size_per_layer_input: usize,
    embedding_dim: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    sliding_window: usize,
    rms_norm_eps: f64,
}

fn validate_caps(numbers: &ConfigNumbers) -> Result<()> {
    let ConfigNumbers {
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        hidden_size_per_layer_input,
        embedding_dim,
        vocab_size,
        max_position_embeddings,
        sliding_window,
        rms_norm_eps,
    } = *numbers;
    if hidden_size == 0 || hidden_size > MAX_HIDDEN_SIZE {
        return Err(Error::limit_exceeded(format!(
            "hidden size {hidden_size} is outside 1..={MAX_HIDDEN_SIZE}"
        )));
    }
    if intermediate_size == 0 || intermediate_size > MAX_INTERMEDIATE_SIZE {
        return Err(Error::limit_exceeded(format!(
            "intermediate size {intermediate_size} is outside 1..={MAX_INTERMEDIATE_SIZE}"
        )));
    }
    if num_hidden_layers == 0 || num_hidden_layers > MAX_LAYERS {
        return Err(Error::limit_exceeded(format!(
            "layer count {num_hidden_layers} is outside 1..={MAX_LAYERS}"
        )));
    }
    if num_attention_heads == 0 || num_attention_heads > MAX_HEADS {
        return Err(Error::limit_exceeded(format!(
            "attention head count {num_attention_heads} is outside 1..={MAX_HEADS}"
        )));
    }
    if hidden_size_per_layer_input == 0 || hidden_size_per_layer_input > MAX_HIDDEN_SIZE {
        return Err(Error::limit_exceeded(format!(
            "per-layer input size {hidden_size_per_layer_input} is outside 1..={MAX_HIDDEN_SIZE}"
        )));
    }
    if embedding_dim == 0 || embedding_dim > MAX_EMBEDDING_DIM {
        return Err(Error::limit_exceeded(format!(
            "embedding dimension {embedding_dim} is outside 1..={MAX_EMBEDDING_DIM}"
        )));
    }
    if vocab_size == 0 || vocab_size > MAX_VOCAB_SIZE {
        return Err(Error::limit_exceeded(format!(
            "vocabulary size {vocab_size} is outside 1..={MAX_VOCAB_SIZE}"
        )));
    }
    if max_position_embeddings == 0 || max_position_embeddings > MAX_POSITIONS {
        return Err(Error::limit_exceeded(format!(
            "position count {max_position_embeddings} is outside 1..={MAX_POSITIONS}"
        )));
    }
    if sliding_window == 0 || sliding_window > max_position_embeddings {
        return Err(Error::limit_exceeded(format!(
            "sliding window {sliding_window} is outside 1..={max_position_embeddings}"
        )));
    }
    if !rms_norm_eps.is_finite() || rms_norm_eps < 0.0 {
        return Err(Error::corrupt(
            "RMSNorm epsilon must be finite and non-negative",
        ));
    }
    validate_head_layout(num_attention_heads, num_key_value_heads, head_dim)
}

fn validate_head_layout(
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
) -> Result<()> {
    if num_key_value_heads == 0 || num_key_value_heads > num_attention_heads {
        return Err(Error::corrupt(format!(
            "key/value head count {num_key_value_heads} is not in 1..={num_attention_heads}"
        )));
    }
    if !num_attention_heads.is_multiple_of(num_key_value_heads) {
        return Err(Error::corrupt(format!(
            "attention head count {num_attention_heads} is not a multiple of key/value head count {num_key_value_heads}"
        )));
    }
    if head_dim == 0 || head_dim > MAX_HEAD_DIM || !head_dim.is_multiple_of(2) {
        return Err(Error::corrupt(format!(
            "head dimension {head_dim} must be a positive even value at most {MAX_HEAD_DIM}"
        )));
    }
    num_attention_heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error::limit_exceeded("query projection dimension overflows"))?;
    num_key_value_heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error::limit_exceeded("key/value projection dimension overflows"))?;
    Ok(())
}

fn validate_uniform_head_dim(layers: &[LayerConfig]) -> Result<()> {
    for layer_type in [LayerType::SlidingAttention, LayerType::FullAttention] {
        let mut dimension = None;
        for layer in layers.iter().filter(|layer| layer.layer_type == layer_type) {
            match dimension {
                None => dimension = Some(layer.head_dim),
                Some(expected) if expected != layer.head_dim => {
                    return Err(Error::unsupported(
                        "layers of the same attention type must share one head dimension",
                    ));
                }
                Some(_) => {}
            }
        }
    }
    Ok(())
}

fn parse_layer_types(text: &Object, num_hidden_layers: usize) -> Result<Vec<LayerType>> {
    let value = text
        .get("layer_types")
        .ok_or_else(|| Error::corrupt("layer_types is missing"))?;
    let array = value
        .as_array()
        .ok_or_else(|| Error::corrupt("layer_types must be an array"))?;
    if array.len() != num_hidden_layers {
        return Err(Error::corrupt(format!(
            "layer_types has {} entries, expected {num_hidden_layers}",
            array.len()
        )));
    }
    let mut layers = Vec::with_capacity(num_hidden_layers);
    for (index, entry) in array.iter().enumerate() {
        let name = entry
            .as_str()
            .ok_or_else(|| Error::corrupt(format!("layer_types[{index}] must be a string")))?;
        let layer_type = match name {
            "sliding_attention" => LayerType::SlidingAttention,
            "full_attention" => LayerType::FullAttention,
            other => {
                return Err(Error::unsupported(format!(
                    "layer type {other} is not sliding_attention or full_attention"
                )));
            }
        };
        layers.push(layer_type);
    }
    if layers.last() != Some(&LayerType::FullAttention) {
        return Err(Error::unsupported("the last layer must use full attention"));
    }
    Ok(layers)
}

#[derive(Debug, Clone, Copy)]
struct LayerOverride {
    head_dim: usize,
    num_key_value_heads: usize,
}

fn parse_per_layer_config(
    text: &Object,
    num_hidden_layers: usize,
) -> Result<std::collections::BTreeMap<usize, LayerOverride>> {
    let Some(value) = text.get("per_layer_config") else {
        return Ok(std::collections::BTreeMap::new());
    };
    let object = value
        .as_object()
        .ok_or_else(|| Error::corrupt("per_layer_config must be an object"))?;
    let mut overrides = std::collections::BTreeMap::new();
    for (key, entry) in object {
        let index: usize = key
            .parse()
            .map_err(|_| Error::corrupt(format!("per_layer_config key {key} is not an index")))?;
        if index >= num_hidden_layers {
            return Err(Error::corrupt(format!(
                "per_layer_config index {index} is outside {num_hidden_layers} layers"
            )));
        }
        let entry = entry.as_object().ok_or_else(|| {
            Error::corrupt(format!("per_layer_config[{index}] must be an object"))
        })?;
        let head_dim = unsigned(entry, "head_dim")?;
        let num_key_value_heads = unsigned(entry, "num_key_value_heads")?;
        overrides.insert(
            index,
            LayerOverride {
                head_dim,
                num_key_value_heads,
            },
        );
    }
    Ok(overrides)
}

fn parse_rope_parameters(text: &Object) -> Result<(f32, f32)> {
    let value = text
        .get("rope_parameters")
        .ok_or_else(|| Error::corrupt("rope_parameters is missing"))?;
    let object = value
        .as_object()
        .ok_or_else(|| Error::corrupt("rope_parameters must be an object"))?;
    let mut full = None;
    let mut sliding = None;
    for (key, entry) in object {
        let entry = entry
            .as_object()
            .ok_or_else(|| Error::corrupt(format!("rope_parameters.{key} must be an object")))?;
        let rope_type = string(entry, "rope_type")?;
        if rope_type != "default" {
            return Err(Error::unsupported(format!(
                "RoPE type {rope_type} is not supported"
            )));
        }
        let theta = float(entry, "rope_theta")?;
        if !theta.is_finite() || theta <= 1.0 {
            return Err(Error::corrupt(format!(
                "RoPE theta {theta} must be finite and greater than one"
            )));
        }
        match key.as_str() {
            "full_attention" => full = Some(theta as f32),
            "sliding_attention" => sliding = Some(theta as f32),
            other => {
                return Err(Error::unsupported(format!(
                    "RoPE parameters for {other} are not supported"
                )));
            }
        }
    }
    match (full, sliding) {
        (Some(full), Some(sliding)) => Ok((full, sliding)),
        _ => Err(Error::corrupt(
            "rope_parameters must declare full_attention and sliding_attention",
        )),
    }
}

fn unsigned(object: &Object, key: &str) -> Result<usize> {
    let value = object
        .get(key)
        .ok_or_else(|| Error::corrupt(format!("configuration is missing {key}")))?;
    let value = value
        .as_number()
        .and_then(Number::as_u64)
        .ok_or_else(|| Error::corrupt(format!("{key} must be a non-negative integer")))?;
    usize::try_from(value).map_err(|_| Error::limit_exceeded(format!("{key} does not fit usize")))
}

fn float(object: &Object, key: &str) -> Result<f64> {
    let value = object
        .get(key)
        .ok_or_else(|| Error::corrupt(format!("configuration is missing {key}")))?;
    value
        .as_number()
        .map(Number::as_f64)
        .ok_or_else(|| Error::corrupt(format!("{key} must be a number")))
}

fn optional_float(object: &Object, key: &str) -> Result<Option<f64>> {
    match object.get(key) {
        Some(value) => value
            .as_number()
            .map(Number::as_f64)
            .map(Some)
            .ok_or_else(|| Error::corrupt(format!("{key} must be a number"))),
        None => Ok(None),
    }
}

fn optional_bool(object: &Object, key: &str) -> Result<Option<bool>> {
    match object.get(key) {
        Some(value) => value
            .as_bool()
            .map(Some)
            .ok_or_else(|| Error::corrupt(format!("{key} must be a boolean"))),
        None => Ok(None),
    }
}

fn optional_token_id(object: &Object, key: &str) -> Result<Option<u32>> {
    match object.get(key) {
        Some(Value::Null) => Ok(None),
        Some(value) => {
            let id = value
                .as_number()
                .and_then(Number::as_u64)
                .ok_or_else(|| Error::corrupt(format!("{key} must be a non-negative integer")))?;
            u32::try_from(id)
                .map(Some)
                .map_err(|_| Error::limit_exceeded(format!("{key} does not fit u32")))
        }
        None => Ok(None),
    }
}

fn string<'a>(object: &'a Object, key: &str) -> Result<&'a str> {
    object
        .get(key)
        .ok_or_else(|| Error::corrupt(format!("configuration is missing {key}")))?
        .as_str()
        .ok_or_else(|| Error::corrupt(format!("{key} must be a string")))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINI: &str = r#"{
        "model_type": "embedding_gemma2_text",
        "dtype": "bfloat16",
        "vocab_size": 64,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 3,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "hidden_size_per_layer_input": 8,
        "embedding_dim": 8,
        "hidden_activation": "gelu_pytorch_tanh",
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "attention_bias": false,
        "attention_dropout": 0.0,
        "sliding_window": 2,
        "layer_types": ["sliding_attention", "sliding_attention", "full_attention"],
        "per_layer_config": {"2": {"head_dim": 16, "num_key_value_heads": 1}},
        "rope_parameters": {
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
            "full_attention": {"rope_theta": 1000000.0, "rope_type": "default"}
        }
    }"#;

    #[test]
    fn parses_a_consistent_text_configuration() {
        let config = TextConfig::from_json_str(MINI).expect("configuration must parse");
        assert_eq!(config.vocab_size(), 64);
        assert_eq!(config.hidden_size(), 16);
        assert_eq!(config.num_hidden_layers(), 3);
        assert_eq!(config.embedding_dim(), 8);
        assert_eq!(config.sliding_window(), 2);
        assert_eq!(config.layers().len(), 3);
        assert_eq!(config.layers()[0].head_dim, 8);
        assert_eq!(config.layers()[1].head_dim, 8);
        assert_eq!(config.layers()[2].head_dim, 16);
        assert_eq!(config.layers()[2].num_key_value_heads, 1);
        assert_eq!(config.embedding_scale(), 4.0);
        assert_eq!(config.ple_scale(), 0.25);
        assert_eq!(config.tensor_shapes().len(), 5 + 3 * 17);
    }

    #[test]
    fn accepts_a_composite_model_configuration() {
        let composite = format!(r#"{{"text_config": {MINI}}}"#);
        let config = TextConfig::from_json_str(&composite).expect("composite configuration");
        assert_eq!(config.hidden_size(), 16);
    }

    #[test]
    fn rejects_unsupported_architecture_choices() {
        for (document, label) in [
            (
                MINI.replace(r#""gelu_pytorch_tanh""#, r#""silu""#),
                "activation",
            ),
            (MINI.replace(r#""bfloat16""#, r#""float16""#), "float16"),
            (
                MINI.replace(r#""attention_bias": false"#, r#""attention_bias": true"#),
                "bias",
            ),
            (
                MINI.replace(r#""attention_dropout": 0.0"#, r#""attention_dropout": 0.1"#),
                "dropout",
            ),
            (
                MINI.replace(r#""sliding_attention""#, r#""causal""#),
                "layer type",
            ),
            (
                MINI.replace(r#""rope_type": "default""#, r#""rope_type": "linear""#),
                "rope type",
            ),
        ] {
            let error = TextConfig::from_json_str(&document).expect_err("must be rejected");
            assert!(
                matches!(error, Error::Unsupported(_) | Error::Corrupt(_)),
                "{label}: {error}"
            );
        }
    }

    #[test]
    fn rejects_missing_and_inconsistent_fields() {
        for document in [
            MINI.replace(r#""hidden_size": 16"#, ""),
            MINI.replace(r#""num_hidden_layers": 3"#, r#""num_hidden_layers": 4"#),
            MINI.replace(r#""sliding_window": 2"#, r#""sliding_window": 0"#),
            MINI.replace(r#""head_dim": 8"#, r#""head_dim": 7"#),
            MINI.replace(r#""num_key_value_heads": 1"#, r#""num_key_value_heads": 3"#),
            MINI.replace(
                r#""2": {"head_dim": 16, "num_key_value_heads": 1}"#,
                r#""3": {"head_dim": 16, "num_key_value_heads": 1}"#,
            ),
            MINI.replace(r#""rms_norm_eps": 1e-6"#, r#""rms_norm_eps": 1e300"#),
        ] {
            let error = TextConfig::from_json_str(&document).expect_err("must be rejected");
            assert!(
                matches!(
                    error,
                    Error::Corrupt(_) | Error::Unsupported(_) | Error::LimitExceeded(_)
                ),
                "unexpected error: {error}"
            );
        }
    }

    #[test]
    fn rejects_a_non_full_last_layer() {
        let document = MINI.replace(
            r#""layer_types": ["sliding_attention", "sliding_attention", "full_attention"]"#,
            r#""layer_types": ["sliding_attention", "sliding_attention", "sliding_attention"]"#,
        );
        let error = TextConfig::from_json_str(&document).expect_err("last layer must be full");
        assert!(matches!(error, Error::Unsupported(_)));
    }

    #[test]
    fn tensor_inventory_matches_the_frozen_model_shape() {
        let config = TextConfig::from_json_str(MINI).expect("configuration");
        let shapes = config.tensor_shapes();
        let shape_of = |name: &str| {
            shapes
                .iter()
                .find(|(candidate, _)| candidate == name)
                .map(|(_, shape)| shape.clone())
                .expect("tensor name")
        };
        assert_eq!(shape_of("language_model.embed_tokens.weight"), vec![64, 16]);
        assert_eq!(
            shape_of("language_model.embedding_projection.weight"),
            vec![8, 16]
        );
        assert_eq!(shape_of("language_model.norm.weight"), vec![16]);
        assert_eq!(
            shape_of("language_model.ple.per_layer_model_projection.weight"),
            vec![24, 16]
        );
        assert_eq!(
            shape_of("language_model.ple.per_layer_projection_norm.weight"),
            vec![8]
        );
        assert_eq!(
            shape_of("language_model.layers.0.self_attn.q_proj.weight"),
            vec![16, 16]
        );
        assert_eq!(
            shape_of("language_model.layers.0.self_attn.k_proj.weight"),
            vec![8, 16]
        );
        assert_eq!(
            shape_of("language_model.layers.2.self_attn.q_proj.weight"),
            vec![32, 16]
        );
    }
}
