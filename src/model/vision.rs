//! Gemma 4 vision tower inference for EmbeddingGemma 2.
//!
//! The tower accepts the padded 16x16 patches and (x, y) grid positions
//! produced by the image preprocessing module. It applies the patch linear
//! projection, learned two-axis position embeddings, 16 bidirectional
//! encoder layers with axial 2-D RoPE, 3x3 spatial pooling, and the projection
//! into the text hidden space. Only real (non-padding) patches are processed:
//! padding patches are masked out of attention and discarded by the pooler,
//! so compressing them away is mathematically equivalent and avoids a large
//! amount of useless work.
//!
//! The implementation is strict: an unknown activation, a clipped-linear
//! configuration, a non-axial RoPE, an inconsistent head layout, a missing
//! tensor, or a shape mismatch is rejected instead of approximated.

use std::path::Path;

use crate::binary::read_file_bounded;
use crate::binary::safetensors::SafetensorsFile;
use crate::binary::sha256::sha256_hex;
use crate::error::{Error, Result};
use crate::json::{self, Limits, Number, Value};
use crate::media::preprocess::{ImagePatches, PATCH_SIZE, PATCH_VALUES, POOLING_KERNEL_SIZE};
use crate::model::text::MAX_ACTIVATION_ELEMENTS;
use crate::tensor::kernels;
use crate::tensor::{Shape, Tensor};

/// Maximum vision hidden size.
pub const MAX_VISION_HIDDEN: usize = 16_384;

/// Maximum vision feed-forward intermediate size.
pub const MAX_VISION_INTERMEDIATE: usize = 65_536;

/// Maximum number of vision transformer layers.
pub const MAX_VISION_LAYERS: usize = 128;

/// Maximum number of attention heads.
pub const MAX_VISION_HEADS: usize = 128;

/// Maximum attention head dimension.
pub const MAX_VISION_HEAD_DIM: usize = 1_024;

/// Maximum learned position-embedding entries per axis.
pub const MAX_POSITION_EMBEDDINGS: usize = 1_048_576;

/// Maximum size of a configuration file.
pub const MAX_VISION_CONFIG_BYTES: u64 = 1024 * 1024;

/// Maximum size of the Safetensors file accepted by the loader.
pub const MAX_VISION_FILE_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Maximum total number of FP32 vision weight elements kept in memory.
pub const MAX_VISION_WEIGHT_ELEMENTS: usize = 1 << 30;

/// Validated Gemma 4 vision configuration.
#[derive(Debug, Clone)]
pub struct VisionConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    patch_size: usize,
    pooling_kernel_size: usize,
    position_embedding_size: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
}

impl VisionConfig {
    /// Parses the vision section of a composite model configuration.
    pub fn from_json_str(input: &str) -> Result<Self> {
        let limits = Limits {
            max_input_bytes: MAX_VISION_CONFIG_BYTES as usize,
            ..Limits::default()
        };
        let value = json::parse_with_limits(input, &limits)?;
        Self::from_json_value(&value)
    }

    /// Parses the vision configuration from an already parsed JSON value.
    pub fn from_json_value(value: &Value) -> Result<Self> {
        let root = value
            .as_object()
            .ok_or_else(|| Error::corrupt("model configuration must be a JSON object"))?;
        let vision = match root.get("vision_config") {
            Some(nested) => nested
                .as_object()
                .ok_or_else(|| Error::corrupt("vision_config must be a JSON object"))?,
            None => {
                return Err(Error::unsupported(
                    "the model has no vision_config; image embeddings are unavailable",
                ));
            }
        };

        if let Some(model_type) = vision.get("model_type") {
            let model_type = model_type
                .as_str()
                .ok_or_else(|| Error::corrupt("vision model_type must be a string"))?;
            if model_type != "gemma4_vision" {
                return Err(Error::unsupported(format!(
                    "vision model type {model_type} is not gemma4_vision"
                )));
            }
        }

        if let Some(dtype) = vision.get("dtype") {
            let dtype = dtype
                .as_str()
                .ok_or_else(|| Error::corrupt("vision dtype must be a string"))?;
            match dtype {
                "bfloat16" | "float32" => {}
                "float16" => {
                    return Err(Error::unsupported(
                        "float16 is forbidden for EmbeddingGemma 2 activations",
                    ));
                }
                other => {
                    return Err(Error::unsupported(format!(
                        "vision dtype {other} is not supported"
                    )));
                }
            }
        }

        let hidden_size = unsigned(vision, "hidden_size")?;
        let intermediate_size = unsigned(vision, "intermediate_size")?;
        let num_hidden_layers = unsigned(vision, "num_hidden_layers")?;
        let num_attention_heads = unsigned(vision, "num_attention_heads")?;
        let num_key_value_heads = unsigned(vision, "num_key_value_heads")?;
        let head_dim = unsigned(vision, "head_dim")?;
        let patch_size = unsigned(vision, "patch_size")?;
        let pooling_kernel_size = unsigned(vision, "pooling_kernel_size")?;
        let position_embedding_size = unsigned(vision, "position_embedding_size")?;
        let rms_norm_eps = float(vision, "rms_norm_eps")?;
        let rms_norm_eps_f32 = rms_norm_eps as f32;
        if !rms_norm_eps_f32.is_finite() || rms_norm_eps_f32 < 0.0 {
            return Err(Error::corrupt(
                "vision RMSNorm epsilon must be finite and non-negative",
            ));
        }

        let hidden_activation = string(vision, "hidden_activation")?;
        if hidden_activation != "gelu_pytorch_tanh" {
            return Err(Error::unsupported(format!(
                "vision activation {hidden_activation} is not gelu_pytorch_tanh"
            )));
        }
        if optional_bool(vision, "attention_bias")?.unwrap_or(false) {
            return Err(Error::unsupported(
                "vision attention bias is not part of the frozen model",
            ));
        }
        if let Some(dropout) = optional_float(vision, "attention_dropout")?
            && dropout != 0.0
        {
            return Err(Error::unsupported(
                "vision attention dropout is not implemented",
            ));
        }
        if optional_bool(vision, "use_clipped_linears")?.unwrap_or(false) {
            return Err(Error::unsupported(
                "clipped vision linear layers are not part of the frozen model",
            ));
        }
        if optional_bool(vision, "standardize")?.unwrap_or(false) {
            return Err(Error::unsupported(
                "vision feature standardization is not part of the frozen model",
            ));
        }
        if patch_size != PATCH_SIZE {
            return Err(Error::unsupported(format!(
                "vision patch size {patch_size} is not the frozen {PATCH_SIZE}"
            )));
        }
        if pooling_kernel_size != POOLING_KERNEL_SIZE {
            return Err(Error::unsupported(format!(
                "vision pooling kernel {pooling_kernel_size} is not the frozen {POOLING_KERNEL_SIZE}"
            )));
        }
        if head_dim == 0 || !head_dim.is_multiple_of(4) {
            return Err(Error::unsupported(
                "vision head dimension must be positive and divisible by four for axial RoPE",
            ));
        }
        if num_attention_heads == 0 || num_key_value_heads == 0 {
            return Err(Error::corrupt(
                "vision attention head counts must be positive",
            ));
        }
        if num_attention_heads % num_key_value_heads != 0 {
            return Err(Error::corrupt(
                "vision query head count is not a multiple of the key/value head count",
            ));
        }

        let q_dim = num_attention_heads
            .checked_mul(head_dim)
            .ok_or_else(|| Error::limit_exceeded("vision query projection width overflow"))?;
        let kv_dim = num_key_value_heads
            .checked_mul(head_dim)
            .ok_or_else(|| Error::limit_exceeded("vision key/value projection width overflow"))?;
        if hidden_size > MAX_VISION_HIDDEN
            || q_dim > MAX_VISION_HIDDEN
            || kv_dim > MAX_VISION_HIDDEN
        {
            return Err(Error::limit_exceeded(format!(
                "vision dimensions exceed the {MAX_VISION_HIDDEN}-element hidden cap"
            )));
        }
        if intermediate_size > MAX_VISION_INTERMEDIATE {
            return Err(Error::limit_exceeded(format!(
                "vision intermediate size {intermediate_size} exceeds the {MAX_VISION_INTERMEDIATE} cap"
            )));
        }
        if num_hidden_layers > MAX_VISION_LAYERS {
            return Err(Error::limit_exceeded(format!(
                "vision layer count {num_hidden_layers} exceeds the {MAX_VISION_LAYERS} cap"
            )));
        }
        if num_attention_heads > MAX_VISION_HEADS || head_dim > MAX_VISION_HEAD_DIM {
            return Err(Error::limit_exceeded(
                "vision attention layout exceeds its documented cap",
            ));
        }
        if position_embedding_size == 0 || position_embedding_size > MAX_POSITION_EMBEDDINGS {
            return Err(Error::limit_exceeded(format!(
                "vision position table size {position_embedding_size} is outside 1..={MAX_POSITION_EMBEDDINGS}"
            )));
        }

        let rope = vision
            .get("rope_parameters")
            .and_then(|value| value.as_object())
            .ok_or_else(|| Error::corrupt("vision rope_parameters must be an object"))?;
        let rope_type = rope
            .get("rope_type")
            .and_then(|value| value.as_str())
            .ok_or_else(|| Error::corrupt("vision rope_type must be a string"))?;
        if rope_type != "axial" {
            return Err(Error::unsupported(format!(
                "vision RoPE type {rope_type} is not axial"
            )));
        }
        let rope_theta = rope
            .get("rope_theta")
            .and_then(|value| value.as_number())
            .map(Number::as_f64)
            .ok_or_else(|| Error::corrupt("vision rope_theta must be a number"))?;
        let rope_theta_f32 = rope_theta as f32;
        if !rope_theta_f32.is_finite() || rope_theta_f32 <= 0.0 {
            return Err(Error::corrupt(
                "vision RoPE theta must be finite and positive",
            ));
        }

        Ok(Self {
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            patch_size,
            pooling_kernel_size,
            position_embedding_size,
            rms_norm_eps: rms_norm_eps_f32,
            rope_theta: rope_theta_f32,
        })
    }

    /// Loads the vision configuration from a file with an explicit size cap.
    pub fn from_file(path: &Path) -> Result<Self> {
        let bytes = read_file_bounded(path, MAX_VISION_CONFIG_BYTES)?;
        let text = std::str::from_utf8(&bytes)
            .map_err(|_| Error::corrupt(format!("{} is not valid UTF-8", path.display())))?;
        Self::from_json_str(text)
    }

    /// Vision hidden size.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Vision feed-forward intermediate size.
    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }

    /// Number of vision transformer layers.
    pub fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }

    /// Number of query heads.
    pub fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }

    /// Number of key/value heads before grouped-query expansion.
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads
    }

    /// Dimension of one attention head.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Patch edge length.
    pub fn patch_size(&self) -> usize {
        self.patch_size
    }

    /// Spatial pooling kernel size.
    pub fn pooling_kernel_size(&self) -> usize {
        self.pooling_kernel_size
    }

    /// Number of learned position-embedding entries per axis.
    pub fn position_embedding_size(&self) -> usize {
        self.position_embedding_size
    }

    /// RMSNorm epsilon.
    pub fn rms_norm_eps(&self) -> f32 {
        self.rms_norm_eps
    }

    /// Axial RoPE theta.
    pub fn rope_theta(&self) -> f32 {
        self.rope_theta
    }

    /// Query projection width.
    pub fn query_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// Key/value projection width.
    pub fn key_value_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// Expected tensor names and shapes for the vision weights.
    pub fn tensor_shapes(&self, text_hidden: usize) -> Vec<(String, Vec<usize>)> {
        let hidden = self.hidden_size;
        let intermediate = self.intermediate_size;
        let head_dim = self.head_dim;
        let query_dim = self.query_dim();
        let key_value_dim = self.key_value_dim();
        let mut shapes = Vec::with_capacity(5 + self.num_hidden_layers * 13);
        shapes.push((
            "vision_tower.patch_embedder.input_proj.weight".to_owned(),
            vec![hidden, PATCH_VALUES],
        ));
        shapes.push((
            "vision_tower.patch_embedder.position_embedding_table".to_owned(),
            vec![2, self.position_embedding_size, hidden],
        ));
        for index in 0..self.num_hidden_layers {
            let prefix = format!("vision_tower.encoder.layers.{index}");
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
            shapes.push((
                format!("{prefix}.self_attn.q_proj.linear.weight"),
                vec![query_dim, hidden],
            ));
            shapes.push((
                format!("{prefix}.self_attn.k_proj.linear.weight"),
                vec![key_value_dim, hidden],
            ));
            shapes.push((
                format!("{prefix}.self_attn.v_proj.linear.weight"),
                vec![key_value_dim, hidden],
            ));
            shapes.push((
                format!("{prefix}.self_attn.o_proj.linear.weight"),
                vec![hidden, query_dim],
            ));
            shapes.push((format!("{prefix}.self_attn.q_norm.weight"), vec![head_dim]));
            shapes.push((format!("{prefix}.self_attn.k_norm.weight"), vec![head_dim]));
            shapes.push((
                format!("{prefix}.mlp.gate_proj.linear.weight"),
                vec![intermediate, hidden],
            ));
            shapes.push((
                format!("{prefix}.mlp.up_proj.linear.weight"),
                vec![intermediate, hidden],
            ));
            shapes.push((
                format!("{prefix}.mlp.down_proj.linear.weight"),
                vec![hidden, intermediate],
            ));
        }
        shapes.push((
            "embed_vision.embedding_projection.weight".to_owned(),
            vec![text_hidden, hidden],
        ));
        shapes
    }
}

fn unsigned(object: &json::Object, key: &str) -> Result<usize> {
    let value = object
        .get(key)
        .ok_or_else(|| Error::corrupt(format!("vision configuration is missing {key}")))?;
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

/// Weights of one vision encoder layer.
#[derive(Debug)]
struct VisionLayerWeights {
    input_layernorm: Vec<f32>,
    post_attention_layernorm: Vec<f32>,
    pre_feedforward_layernorm: Vec<f32>,
    post_feedforward_layernorm: Vec<f32>,
    q_proj: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    o_proj: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    gate_proj: Vec<f32>,
    up_proj: Vec<f32>,
    down_proj: Vec<f32>,
}

/// FP32 weights of the vision tower and its text-space projection.
#[derive(Debug)]
pub struct VisionWeights {
    input_proj: Vec<f32>,
    position_embedding_table: Vec<f32>,
    layers: Vec<VisionLayerWeights>,
    embedding_projection: Vec<f32>,
    fingerprint: String,
    parameters: usize,
}

impl VisionWeights {
    /// Loads the vision weights from a Safetensors file.
    pub fn from_file(path: &Path, config: &VisionConfig, text_hidden: usize) -> Result<Self> {
        let bytes = read_file_bounded(path, MAX_VISION_FILE_BYTES)?;
        Self::from_bytes(&bytes, config, text_hidden)
    }

    /// Loads the vision weights from an in-memory Safetensors file.
    pub fn from_bytes(bytes: &[u8], config: &VisionConfig, text_hidden: usize) -> Result<Self> {
        let fingerprint = sha256_hex(bytes);
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

    /// Number of vision encoder layers.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    fn decode(
        file: &SafetensorsFile<'_>,
        config: &VisionConfig,
        text_hidden: usize,
        fingerprint: String,
    ) -> Result<Self> {
        let expected = config.tensor_shapes(text_hidden);
        let expected_names: std::collections::BTreeSet<&str> =
            expected.iter().map(|(name, _)| name.as_str()).collect();

        for entry in file.entries() {
            let name = entry.name();
            let is_vision = name.starts_with("vision_tower.") || name.starts_with("embed_vision.");
            if is_vision && !expected_names.contains(name) {
                return Err(Error::corrupt(format!(
                    "unexpected vision tensor {name} in the weight file"
                )));
            }
        }

        let mut decoded: std::collections::BTreeMap<String, Tensor> =
            std::collections::BTreeMap::new();
        let mut parameters: usize = 0;
        for (name, shape) in &expected {
            let entry = file.tensor(name).ok_or_else(|| {
                Error::corrupt(format!(
                    "vision weight {name} is missing from the Safetensors file"
                ))
            })?;
            if entry.shape().dims() != shape.as_slice() {
                return Err(Error::corrupt(format!(
                    "vision weight {name} has shape {}, expected {shape:?}",
                    entry.shape()
                )));
            }
            parameters = parameters
                .checked_add(entry.shape().numel())
                .ok_or_else(|| Error::limit_exceeded("vision weight element count overflow"))?;
            if parameters > MAX_VISION_WEIGHT_ELEMENTS {
                return Err(Error::limit_exceeded(format!(
                    "vision weights exceed the {MAX_VISION_WEIGHT_ELEMENTS}-element cap"
                )));
            }
            let tensor = entry.to_f32()?;
            tensor.validate_finite()?;
            decoded.insert(name.clone(), tensor);
        }

        let layers = build_layers(&mut decoded, config)?;
        Ok(Self {
            input_proj: take(
                &mut decoded,
                "vision_tower.patch_embedder.input_proj.weight",
            )?,
            position_embedding_table: take(
                &mut decoded,
                "vision_tower.patch_embedder.position_embedding_table",
            )?,
            layers,
            embedding_projection: take(&mut decoded, "embed_vision.embedding_projection.weight")?,
            fingerprint,
            parameters,
        })
    }
}

fn build_layers(
    decoded: &mut std::collections::BTreeMap<String, Tensor>,
    config: &VisionConfig,
) -> Result<Vec<VisionLayerWeights>> {
    let mut layers = Vec::with_capacity(config.num_hidden_layers());
    for index in 0..config.num_hidden_layers() {
        let prefix = format!("vision_tower.encoder.layers.{index}");
        layers.push(VisionLayerWeights {
            input_layernorm: take(decoded, &format!("{prefix}.input_layernorm.weight"))?,
            post_attention_layernorm: take(
                decoded,
                &format!("{prefix}.post_attention_layernorm.weight"),
            )?,
            pre_feedforward_layernorm: take(
                decoded,
                &format!("{prefix}.pre_feedforward_layernorm.weight"),
            )?,
            post_feedforward_layernorm: take(
                decoded,
                &format!("{prefix}.post_feedforward_layernorm.weight"),
            )?,
            q_proj: take(decoded, &format!("{prefix}.self_attn.q_proj.linear.weight"))?,
            k_proj: take(decoded, &format!("{prefix}.self_attn.k_proj.linear.weight"))?,
            v_proj: take(decoded, &format!("{prefix}.self_attn.v_proj.linear.weight"))?,
            o_proj: take(decoded, &format!("{prefix}.self_attn.o_proj.linear.weight"))?,
            q_norm: take(decoded, &format!("{prefix}.self_attn.q_norm.weight"))?,
            k_norm: take(decoded, &format!("{prefix}.self_attn.k_norm.weight"))?,
            gate_proj: take(decoded, &format!("{prefix}.mlp.gate_proj.linear.weight"))?,
            up_proj: take(decoded, &format!("{prefix}.mlp.up_proj.linear.weight"))?,
            down_proj: take(decoded, &format!("{prefix}.mlp.down_proj.linear.weight"))?,
        });
    }
    Ok(layers)
}

fn take(decoded: &mut std::collections::BTreeMap<String, Tensor>, name: &str) -> Result<Vec<f32>> {
    let tensor = decoded
        .remove(name)
        .ok_or_else(|| Error::internal(format!("decoded tensor {name} is missing")))?;
    Ok(tensor.into_parts().1)
}

/// Result of one vision tower pass.
#[derive(Debug)]
pub struct VisionOutput {
    pooled: Vec<f32>,
    hidden_states: Vec<Tensor>,
    soft_tokens: usize,
    text_hidden: usize,
}

impl VisionOutput {
    /// Projected soft tokens into the text hidden space, row-major.
    pub fn pooled(&self) -> &[f32] {
        &self.pooled
    }

    /// Captured hidden states: the patch-embedder output followed by every
    /// encoder layer. Empty unless capture was requested.
    pub fn hidden_states(&self) -> &[Tensor] {
        &self.hidden_states
    }

    /// Number of visual soft tokens.
    pub fn soft_tokens(&self) -> usize {
        self.soft_tokens
    }

    /// Width of each projected soft token.
    pub fn text_hidden(&self) -> usize {
        self.text_hidden
    }
}

/// FP32 vision tower and its projection into the text hidden space.
#[derive(Debug)]
pub struct VisionModel {
    config: VisionConfig,
    weights: VisionWeights,
    text_hidden: usize,
}

impl VisionModel {
    /// Builds a vision model from a validated configuration and weight set.
    pub fn new(config: VisionConfig, weights: VisionWeights, text_hidden: usize) -> Result<Self> {
        if weights.layer_count() != config.num_hidden_layers() {
            return Err(Error::internal(format!(
                "vision weight set has {} layers, configuration declares {}",
                weights.layer_count(),
                config.num_hidden_layers()
            )));
        }
        let expected_projection = config
            .hidden_size()
            .checked_mul(text_hidden)
            .ok_or_else(|| Error::limit_exceeded("vision projection size overflow"))?;
        if weights.embedding_projection.len() != expected_projection {
            return Err(Error::internal(format!(
                "vision projection has {} elements, expected {expected_projection}",
                weights.embedding_projection.len()
            )));
        }
        Ok(Self {
            config,
            weights,
            text_hidden,
        })
    }

    /// Vision configuration.
    pub fn config(&self) -> &VisionConfig {
        &self.config
    }

    /// SHA-256 fingerprint of the source weight file.
    pub fn fingerprint_hex(&self) -> &str {
        self.weights.fingerprint_hex()
    }

    /// Number of FP32 vision parameters loaded in memory.
    pub fn parameters(&self) -> usize {
        self.weights.parameters()
    }

    /// Runs the vision tower and returns the projected soft tokens.
    pub fn forward(&self, patches: &ImagePatches, capture_hidden: bool) -> Result<VisionOutput> {
        let valid = patches.valid_patches();
        if valid == 0 {
            return Err(Error::corrupt("vision tower received zero real patches"));
        }
        if valid > patches.max_patches() {
            return Err(Error::internal(
                "vision tower received more real patches than the padded tensor",
            ));
        }
        let hidden_size = self.config.hidden_size();
        let q_dim = self.config.query_dim();
        let kv_dim = self.config.key_value_dim();
        let head_dim = self.config.head_dim();
        let num_heads = self.config.num_attention_heads();
        let num_kv_heads = self.config.num_key_value_heads();
        let epsilon = self.config.rms_norm_eps();

        let patch_elements = checked_len(valid, PATCH_VALUES, "vision patch tensor")?;
        let source = patches.pixel_values();
        if source.len() < patch_elements {
            return Err(Error::internal(
                "vision patch tensor is shorter than the valid patch count",
            ));
        }
        let mut scaled = vec![0.0f32; patch_elements];
        for (destination, value) in scaled.iter_mut().zip(&source[..patch_elements]) {
            *destination = 2.0 * (*value - 0.5);
        }
        let mut hidden = kernels::linear(
            &scaled,
            &self.weights.input_proj,
            valid,
            PATCH_VALUES,
            hidden_size,
        )?;

        let position_size = self.config.position_embedding_size();
        let table = &self.weights.position_embedding_table;
        let expected_table = position_size
            .checked_mul(hidden_size)
            .and_then(|values| values.checked_mul(2))
            .ok_or_else(|| Error::limit_exceeded("vision position table size overflow"))?;
        if table.len() != expected_table {
            return Err(Error::internal(format!(
                "vision position table has {} values, expected {expected_table}",
                table.len()
            )));
        }
        for patch in 0..valid {
            let position = patches.positions()[patch];
            let x = usize::try_from(position[0])
                .map_err(|_| Error::internal("vision patch x position is negative"))?;
            let y = usize::try_from(position[1])
                .map_err(|_| Error::internal("vision patch y position is negative"))?;
            if x >= position_size || y >= position_size {
                return Err(Error::corrupt(format!(
                    "vision patch position ({x}, {y}) is outside the {position_size}-entry table"
                )));
            }
            let x_row = &table[x * hidden_size..(x + 1) * hidden_size];
            let y_row = &table[position_size * hidden_size + y * hidden_size
                ..position_size * hidden_size + (y + 1) * hidden_size];
            let destination = &mut hidden[patch * hidden_size..(patch + 1) * hidden_size];
            for ((target, x_value), y_value) in destination.iter_mut().zip(x_row).zip(y_row) {
                *target += x_value + y_value;
            }
        }

        let mut hidden_states = Vec::new();
        if capture_hidden {
            hidden_states.push(tensor_from(valid, hidden_size, &hidden)?);
        }

        let inv_freq = axial_inv_freq(head_dim, self.config.rope_theta())?;
        for layer in &self.weights.layers {
            let residual = hidden.clone();
            let normalized = kernels::rms_norm_rows(
                &hidden,
                &layer.input_layernorm,
                valid,
                hidden_size,
                epsilon,
            )?;
            let query = kernels::linear(&normalized, &layer.q_proj, valid, hidden_size, q_dim)?;
            let key = kernels::linear(&normalized, &layer.k_proj, valid, hidden_size, kv_dim)?;
            let value = kernels::linear(&normalized, &layer.v_proj, valid, hidden_size, kv_dim)?;

            let query = kernels::rms_norm_rows(
                &query,
                &layer.q_norm,
                checked_len(valid, num_heads, "vision query rows")?,
                head_dim,
                epsilon,
            )?;
            let key = kernels::rms_norm_rows(
                &key,
                &layer.k_norm,
                checked_len(valid, num_kv_heads, "vision key rows")?,
                head_dim,
                epsilon,
            )?;
            let value = kernels::rms_norm_rows_unscaled(
                &value,
                checked_len(valid, num_kv_heads, "vision value rows")?,
                head_dim,
                epsilon,
            )?;

            let mut query = query;
            let mut key = key;
            axial_rope(
                &mut query,
                valid,
                num_heads,
                head_dim,
                &patches.positions()[..valid],
                &inv_freq,
            )?;
            axial_rope(
                &mut key,
                valid,
                num_kv_heads,
                head_dim,
                &patches.positions()[..valid],
                &inv_freq,
            )?;

            let attention = kernels::bidirectional_attention(
                &query,
                &key,
                &value,
                kernels::AttentionShape {
                    seq_len: valid,
                    num_heads,
                    num_key_value_heads: num_kv_heads,
                    head_dim,
                    sliding_window: None,
                    scale: 1.0,
                },
            )?;
            let attention = kernels::linear(&attention, &layer.o_proj, valid, q_dim, hidden_size)?;
            let attention = kernels::rms_norm_rows(
                &attention,
                &layer.post_attention_layernorm,
                valid,
                hidden_size,
                epsilon,
            )?;
            hidden = kernels::add(&residual, &attention)?;

            let residual = hidden.clone();
            let normalized = kernels::rms_norm_rows(
                &hidden,
                &layer.pre_feedforward_layernorm,
                valid,
                hidden_size,
                epsilon,
            )?;
            let gate = kernels::linear(
                &normalized,
                &layer.gate_proj,
                valid,
                hidden_size,
                self.config.intermediate_size(),
            )?;
            let up = kernels::linear(
                &normalized,
                &layer.up_proj,
                valid,
                hidden_size,
                self.config.intermediate_size(),
            )?;
            let activated = kernels::geglu(&gate, &up)?;
            let down = kernels::linear(
                &activated,
                &layer.down_proj,
                valid,
                self.config.intermediate_size(),
                hidden_size,
            )?;
            let down = kernels::rms_norm_rows(
                &down,
                &layer.post_feedforward_layernorm,
                valid,
                hidden_size,
                epsilon,
            )?;
            hidden = kernels::add(&residual, &down)?;

            if capture_hidden {
                hidden_states.push(tensor_from(valid, hidden_size, &hidden)?);
            }
        }

        let pooled = pool_patches(&hidden, patches, hidden_size)?;
        let normalized =
            kernels::rms_norm_rows_unscaled(&pooled, patches.soft_tokens(), hidden_size, epsilon)?;
        let projected = kernels::linear(
            &normalized,
            &self.weights.embedding_projection,
            patches.soft_tokens(),
            hidden_size,
            self.text_hidden,
        )?;
        if projected.iter().any(|value| !value.is_finite()) {
            return Err(Error::internal(
                "vision projection produced a non-finite soft token",
            ));
        }

        Ok(VisionOutput {
            pooled: projected,
            hidden_states,
            soft_tokens: patches.soft_tokens(),
            text_hidden: self.text_hidden,
        })
    }
}

fn checked_len(rows: usize, width: usize, label: &str) -> Result<usize> {
    let length = rows
        .checked_mul(width)
        .ok_or_else(|| Error::limit_exceeded(format!("{label} size overflow")))?;
    if length > MAX_ACTIVATION_ELEMENTS {
        return Err(Error::limit_exceeded(format!(
            "{label} would allocate {length} elements, above the {MAX_ACTIVATION_ELEMENTS}-element cap"
        )));
    }
    Ok(length)
}

fn tensor_from(rows: usize, width: usize, data: &[f32]) -> Result<Tensor> {
    let shape = Shape::new(vec![rows, width])?;
    Tensor::from_slice(shape, data)
}

/// Averages each 3x3 patch block and scales by sqrt(hidden size).
fn pool_patches(hidden: &[f32], patches: &ImagePatches, hidden_size: usize) -> Result<Vec<f32>> {
    let (grid_width, grid_height) = patches.patch_grid();
    if grid_width % POOLING_KERNEL_SIZE != 0 || grid_height % POOLING_KERNEL_SIZE != 0 {
        return Err(Error::internal(
            "vision patch grid is not divisible by the pooling kernel",
        ));
    }
    let pooled_width = grid_width / POOLING_KERNEL_SIZE;
    let pooled_height = grid_height / POOLING_KERNEL_SIZE;
    let soft_tokens = pooled_width
        .checked_mul(pooled_height)
        .ok_or_else(|| Error::limit_exceeded("vision pooled grid overflow"))?;
    if soft_tokens != patches.soft_tokens() {
        return Err(Error::internal(format!(
            "vision pooler produces {soft_tokens} tokens, preprocessing declared {}",
            patches.soft_tokens()
        )));
    }
    let expected = checked_len(patches.valid_patches(), hidden_size, "vision hidden state")?;
    if hidden.len() != expected {
        return Err(Error::internal(format!(
            "vision hidden state has {} values, expected {expected}",
            hidden.len()
        )));
    }
    let mut pooled = vec![0.0f32; checked_len(soft_tokens, hidden_size, "vision pooled state")?];
    let divisor = (POOLING_KERNEL_SIZE * POOLING_KERNEL_SIZE) as f32;
    for patch in 0..patches.valid_patches() {
        let position = patches.positions()[patch];
        let x = usize::try_from(position[0])
            .map_err(|_| Error::internal("vision patch x position is negative"))?;
        let y = usize::try_from(position[1])
            .map_err(|_| Error::internal("vision patch y position is negative"))?;
        let bin = (x / POOLING_KERNEL_SIZE) + pooled_width * (y / POOLING_KERNEL_SIZE);
        if bin >= soft_tokens {
            return Err(Error::internal(
                "vision pooler computed an out-of-range bin",
            ));
        }
        let source = &hidden[patch * hidden_size..(patch + 1) * hidden_size];
        let destination = &mut pooled[bin * hidden_size..(bin + 1) * hidden_size];
        for (target, value) in destination.iter_mut().zip(source) {
            *target += *value;
        }
    }
    let scale = (hidden_size as f32).sqrt();
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Error::internal(
            "vision pooling scale is not finite and positive",
        ));
    }
    for value in &mut pooled {
        *value = (*value / divisor) * scale;
    }
    if pooled.iter().any(|value| !value.is_finite()) {
        return Err(Error::internal(
            "vision pooling produced a non-finite value",
        ));
    }
    Ok(pooled)
}

/// Computes the axial RoPE inverse frequencies.
///
/// The tower uses head_dim / 4 frequencies over half the head dimension:
/// the reference builds them with arange(0, head_dim / 2, 2) divided by
/// head_dim / 2.
fn axial_inv_freq(head_dim: usize, theta: f32) -> Result<Vec<f32>> {
    if head_dim == 0 || !head_dim.is_multiple_of(4) {
        return Err(Error::internal(
            "axial RoPE requires a head dimension divisible by four",
        ));
    }
    let spatial_dim = head_dim / 2;
    let count = spatial_dim / 2;
    let mut frequencies = Vec::with_capacity(count);
    for index in 0..count {
        let exponent = (2 * index) as f32 / spatial_dim as f32;
        let value = 1.0 / theta.powf(exponent);
        if !value.is_finite() {
            return Err(Error::internal("axial RoPE frequency is not finite"));
        }
        frequencies.push(value);
    }
    Ok(frequencies)
}

/// Applies the two-axis RoPE layout used by the Gemma 4 vision tower.
///
/// The first half of each head rotates with the x coordinate and the second
/// half with the y coordinate. Within each half, the first and second quarters
/// form the rotate-half pairs, matching the reference's H-H-W-W frequency
/// recomposition.
fn axial_rope(
    values: &mut [f32],
    sequence: usize,
    heads: usize,
    head_dim: usize,
    positions: &[[i32; 2]],
    inv_freq: &[f32],
) -> Result<()> {
    if !head_dim.is_multiple_of(4) {
        return Err(Error::internal(
            "axial RoPE requires a head dimension divisible by four",
        ));
    }
    if inv_freq.len() != head_dim / 4 {
        return Err(Error::internal(format!(
            "axial RoPE has {} frequencies, expected {}",
            inv_freq.len(),
            head_dim / 4
        )));
    }
    let expected = sequence
        .checked_mul(heads)
        .and_then(|rows| rows.checked_mul(head_dim))
        .ok_or_else(|| Error::limit_exceeded("axial RoPE buffer overflow"))?;
    if values.len() != expected || positions.len() < sequence {
        return Err(Error::internal(
            "axial RoPE input does not match the sequence layout",
        ));
    }
    let half = head_dim / 2;
    let quarter = head_dim / 4;
    let mut cos_h = vec![0.0f32; sequence * quarter];
    let mut sin_h = vec![0.0f32; sequence * quarter];
    let mut cos_w = vec![0.0f32; sequence * quarter];
    let mut sin_w = vec![0.0f32; sequence * quarter];
    for (patch, position) in positions.iter().take(sequence).enumerate() {
        if position[0] < 0 || position[1] < 0 {
            return Err(Error::internal("axial RoPE received a padding position"));
        }
        let x = position[0] as f32;
        let y = position[1] as f32;
        for (index, frequency) in inv_freq.iter().enumerate() {
            let angle_h = x * frequency;
            let angle_w = y * frequency;
            let offset = patch * quarter + index;
            cos_h[offset] = angle_h.cos();
            sin_h[offset] = angle_h.sin();
            cos_w[offset] = angle_w.cos();
            sin_w[offset] = angle_w.sin();
        }
    }

    for patch in 0..sequence {
        for head in 0..heads {
            let base = (patch * heads + head) * head_dim;
            let wide = base + half;
            for index in 0..quarter {
                let offset = patch * quarter + index;
                let (cos, sin) = (cos_h[offset], sin_h[offset]);
                let first = values[base + index];
                let second = values[base + index + quarter];
                values[base + index] = first * cos - second * sin;
                values[base + index + quarter] = second * cos + first * sin;
                let (cos, sin) = (cos_w[offset], sin_w[offset]);
                let first = values[wide + index];
                let second = values[wide + index + quarter];
                values[wide + index] = first * cos - second * sin;
                values[wide + index + quarter] = second * cos + first * sin;
            }
        }
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Error::internal("axial RoPE produced a non-finite value"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axial_frequencies_match_the_reference_layout() {
        let frequencies = axial_inv_freq(64, 100.0).expect("axial frequencies");
        assert_eq!(frequencies.len(), 16);
        assert!((frequencies[0] - 1.0).abs() <= 1e-6);
        let expected = 1.0 / 100.0f32.powf(30.0 / 32.0);
        assert!((frequencies[15] - expected).abs() <= 1e-6);
    }

    #[test]
    fn axial_rope_rotates_the_x_and_y_halves() {
        let frequencies = axial_inv_freq(4, 100.0).expect("axial frequencies");
        assert_eq!(frequencies, vec![1.0]);
        let mut values = vec![1.0, 0.0, 1.0, 0.0];
        axial_rope(&mut values, 1, 1, 4, &[[1, 0]], &frequencies).expect("axial rope");
        assert!((values[0] - 1.0f32.cos()).abs() <= 1e-6);
        assert!((values[1] - 1.0f32.sin()).abs() <= 1e-6);
        assert!((values[2] - 1.0).abs() <= 1e-6);
        assert!((values[3] - 0.0).abs() <= 1e-6);
    }

    #[test]
    fn axial_rope_rejects_a_non_divisible_head_dimension() {
        let error = axial_inv_freq(6, 100.0).expect_err("head dimension must be divisible by four");
        assert!(matches!(error, Error::Internal(_)));
        let error = axial_rope(&mut [0.0; 6], 1, 1, 6, &[[0, 0]], &[1.0])
            .expect_err("head dimension must be divisible by four");
        assert!(matches!(error, Error::Internal(_)));
    }
}
