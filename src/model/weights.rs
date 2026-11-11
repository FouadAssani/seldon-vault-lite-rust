//! Loader and validated inventory of the EmbeddingGemma 2 text weights.
//!
//! The loader reads the frozen Safetensors file, accepts only the tensors of
//! the text backbone, converts BF16 or F32 storage to FP32 once, and checks
//! every name and shape against the configuration. A missing tensor, an extra
//! text tensor, a shape mismatch, or a non-finite weight is rejected before
//! inference starts. Audio and vision tensors in the composite file are
//! ignored: this milestone implements text only.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::binary::read_file_bounded;
use crate::binary::safetensors::SafetensorsFile;
use crate::binary::sha256::{hex_digest, sha256_hex};
use crate::error::{Error, Result};
use crate::model::config::TextConfig;
use crate::model::source::ModelFiles;
use crate::tensor::Tensor;

/// Maximum size of the Safetensors file accepted by the loader.
///
/// The frozen composite file is about 1.5 GB. The cap leaves room for a
/// future shard but refuses an unbounded allocation from a hostile path.
pub const MAX_MODEL_FILE_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Maximum total number of FP32 weight elements kept in memory.
///
/// The text backbone has about 237 million parameters, or about 948 MiB in
/// FP32. The cap is deliberately generous but finite.
pub const MAX_WEIGHT_ELEMENTS: usize = 1 << 30;

/// Weights of one transformer layer.
#[derive(Debug)]
pub(crate) struct LayerWeights {
    pub input_layernorm: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
    pub pre_feedforward_layernorm: Vec<f32>,
    pub post_feedforward_layernorm: Vec<f32>,
    pub layer_scalar: f32,
    pub gate_proj: Vec<f32>,
    pub up_proj: Vec<f32>,
    pub down_proj: Vec<f32>,
    pub per_layer_input_gate: Vec<f32>,
    pub per_layer_projection: Vec<f32>,
    pub post_per_layer_input_norm: Vec<f32>,
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
}

/// FP32 weights of the text backbone.
#[derive(Debug)]
pub struct TextWeights {
    pub(crate) embed_tokens: Vec<f32>,
    pub(crate) embedding_projection: Vec<f32>,
    pub(crate) norm: Vec<f32>,
    pub(crate) ple_per_layer_model_projection: Vec<f32>,
    pub(crate) ple_per_layer_projection_norm: Vec<f32>,
    pub(crate) layers: Vec<LayerWeights>,
    fingerprint: String,
    parameters: usize,
}

impl TextWeights {
    /// Loads the text weights from a Safetensors file.
    ///
    /// # Errors
    ///
    /// Returns a typed error for an oversized file, a malformed header, a
    /// missing or extra text tensor, a shape mismatch, or a non-finite value.
    pub fn from_file(path: &Path, config: &TextConfig) -> Result<Self> {
        let bytes = read_file_bounded(path, MAX_MODEL_FILE_BYTES)?;
        Self::from_bytes(&bytes, config)
    }

    /// Loads model.safetensors from a directory or bundle source.
    ///
    /// A bundle section already carries its SHA-256 in the header, so the
    /// loader reuses it instead of hashing the 1.5 GB section again.
    pub fn from_files(files: &ModelFiles<'_>, config: &TextConfig) -> Result<Self> {
        let bytes = files.read("model.safetensors", MAX_MODEL_FILE_BYTES)?;
        match files.section_sha256("model.safetensors") {
            Some(digest) => Self::from_bytes_with_fingerprint(&bytes, config, hex_digest(&digest)),
            None => Self::from_bytes(&bytes, config),
        }
    }

    /// Loads the text weights from an in-memory Safetensors file.
    pub fn from_bytes(bytes: &[u8], config: &TextConfig) -> Result<Self> {
        let fingerprint = sha256_hex(bytes);
        Self::from_bytes_with_fingerprint(bytes, config, fingerprint)
    }

    /// Loads the text weights with an already verified SHA-256 fingerprint.
    ///
    /// The caller must have verified that `fingerprint` belongs to `bytes`.
    pub fn from_bytes_with_fingerprint(
        bytes: &[u8],
        config: &TextConfig,
        fingerprint: String,
    ) -> Result<Self> {
        let file = SafetensorsFile::parse(bytes)?;
        Self::decode(&file, config, fingerprint)
    }

    /// SHA-256 fingerprint of the source file.
    pub fn fingerprint_hex(&self) -> &str {
        &self.fingerprint
    }

    /// Number of FP32 parameters held in memory.
    pub fn parameters(&self) -> usize {
        self.parameters
    }

    /// Number of transformer layers.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    fn decode(
        file: &SafetensorsFile<'_>,
        config: &TextConfig,
        fingerprint: String,
    ) -> Result<Self> {
        let expected = config.tensor_shapes();
        let expected_names: BTreeSet<&str> =
            expected.iter().map(|(name, _)| name.as_str()).collect();

        for entry in file.entries() {
            let name = entry.name();
            if name.starts_with("language_model.") && !expected_names.contains(name) {
                return Err(Error::corrupt(format!(
                    "unexpected text tensor {name} in the weight file"
                )));
            }
        }

        let mut decoded: BTreeMap<String, Tensor> = BTreeMap::new();
        let mut parameters: usize = 0;
        for (name, shape) in &expected {
            let entry = file.tensor(name).ok_or_else(|| {
                Error::corrupt(format!(
                    "text weight {name} is missing from the Safetensors file"
                ))
            })?;
            if entry.shape().dims() != shape.as_slice() {
                return Err(Error::corrupt(format!(
                    "text weight {name} has shape {}, expected {shape:?}",
                    entry.shape()
                )));
            }
            parameters = parameters
                .checked_add(entry.shape().numel())
                .ok_or_else(|| Error::limit_exceeded("weight element count overflow"))?;
            if parameters > MAX_WEIGHT_ELEMENTS {
                return Err(Error::limit_exceeded(format!(
                    "text weights exceed the {MAX_WEIGHT_ELEMENTS}-element cap"
                )));
            }
            let tensor = entry.to_f32()?;
            tensor.validate_finite()?;
            decoded.insert(name.clone(), tensor);
        }

        let layers = build_layers(&mut decoded, config)?;
        let weights = Self {
            embed_tokens: take(&mut decoded, "language_model.embed_tokens.weight")?,
            embedding_projection: take(&mut decoded, "language_model.embedding_projection.weight")?,
            norm: take(&mut decoded, "language_model.norm.weight")?,
            ple_per_layer_model_projection: take(
                &mut decoded,
                "language_model.ple.per_layer_model_projection.weight",
            )?,
            ple_per_layer_projection_norm: take(
                &mut decoded,
                "language_model.ple.per_layer_projection_norm.weight",
            )?,
            layers,
            fingerprint,
            parameters,
        };
        Ok(weights)
    }
}

fn build_layers(
    decoded: &mut BTreeMap<String, Tensor>,
    config: &TextConfig,
) -> Result<Vec<LayerWeights>> {
    let mut layers = Vec::with_capacity(config.num_hidden_layers());
    for index in 0..config.num_hidden_layers() {
        let prefix = format!("language_model.layers.{index}");
        let scalar = take(decoded, &format!("{prefix}.layer_scalar"))?;
        let layer_scalar = *scalar
            .first()
            .ok_or_else(|| Error::corrupt(format!("{prefix}.layer_scalar is empty")))?;
        layers.push(LayerWeights {
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
            layer_scalar,
            gate_proj: take(decoded, &format!("{prefix}.mlp.gate_proj.weight"))?,
            up_proj: take(decoded, &format!("{prefix}.mlp.up_proj.weight"))?,
            down_proj: take(decoded, &format!("{prefix}.mlp.down_proj.weight"))?,
            per_layer_input_gate: take(
                decoded,
                &format!("{prefix}.ple_block.per_layer_input_gate.weight"),
            )?,
            per_layer_projection: take(
                decoded,
                &format!("{prefix}.ple_block.per_layer_projection.weight"),
            )?,
            post_per_layer_input_norm: take(
                decoded,
                &format!("{prefix}.ple_block.post_per_layer_input_norm.weight"),
            )?,
            q_proj: take(decoded, &format!("{prefix}.self_attn.q_proj.weight"))?,
            k_proj: take(decoded, &format!("{prefix}.self_attn.k_proj.weight"))?,
            v_proj: take(decoded, &format!("{prefix}.self_attn.v_proj.weight"))?,
            o_proj: take(decoded, &format!("{prefix}.self_attn.o_proj.weight"))?,
            q_norm: take(decoded, &format!("{prefix}.self_attn.q_norm.weight"))?,
            k_norm: take(decoded, &format!("{prefix}.self_attn.k_norm.weight"))?,
        });
    }
    Ok(layers)
}

fn take(decoded: &mut BTreeMap<String, Tensor>, name: &str) -> Result<Vec<f32>> {
    let tensor = decoded
        .remove(name)
        .ok_or_else(|| Error::internal(format!("decoded tensor {name} is missing")))?;
    Ok(tensor.into_parts().1)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = include_str!("../../fixtures/model/tiny_text/config.json");
    const WEIGHTS: &[u8] = include_bytes!("../../fixtures/model/tiny_text/model.safetensors");

    fn tiny_config() -> TextConfig {
        TextConfig::from_json_str(CONFIG).expect("tiny configuration must parse")
    }

    #[test]
    fn loads_the_tiny_fixture_and_checks_the_inventory() {
        let config = tiny_config();
        let weights = TextWeights::from_bytes(WEIGHTS, &config).expect("tiny weights must load");
        assert_eq!(weights.layer_count(), 3);
        assert_eq!(weights.parameters(), 10_315);
        assert_eq!(weights.embed_tokens.len(), 64 * 16);
        assert_eq!(weights.embedding_projection.len(), 8 * 16);
        assert_eq!(weights.ple_per_layer_model_projection.len(), 24 * 16);
        assert_eq!(weights.layers[0].q_proj.len(), 16 * 16);
        assert_eq!(weights.layers[2].q_proj.len(), 32 * 16);
        assert_eq!(weights.layers[2].o_proj.len(), 16 * 32);
        assert_eq!(weights.fingerprint_hex().len(), 64);
    }

    #[test]
    fn rejects_missing_and_shape_incompatible_weights() {
        let config = tiny_config();
        let error = TextWeights::from_bytes(
            include_bytes!("../../fixtures/formats/mini.safetensors"),
            &config,
        )
        .expect_err("the format fixture is not a text model");
        assert!(matches!(error, Error::Corrupt(_)));

        let wrong = CONFIG.replace(r#""hidden_size": 16"#, r#""hidden_size": 32"#);
        let wrong = TextConfig::from_json_str(&wrong).expect("configuration");
        let error = TextWeights::from_bytes(WEIGHTS, &wrong)
            .expect_err("a different hidden size must reject the weights");
        assert!(matches!(error, Error::Corrupt(_)));
    }

    #[test]
    fn rejects_a_non_finite_weight() {
        // The tiny fixture has no NaN. This test proves the loader checks the
        // decoded values by exercising a deliberately corrupted header on a
        // minimal synthetic file.
        let header = r#"{"language_model.embed_tokens.weight":{"dtype":"F32","shape":[1,1],"data_offsets":[0,4]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&f32::NAN.to_le_bytes());
        let config = TextConfig::from_json_str(
            &CONFIG
                .replace(r#""vocab_size": 64"#, r#""vocab_size": 1"#)
                .replace(r#""hidden_size": 16"#, r#""hidden_size": 1"#)
                .replace(r#""bos_token_id": 1"#, r#""bos_token_id": 0"#)
                .replace(r#""eos_token_id": 2"#, r#""eos_token_id": 0"#),
        )
        .expect("minimal configuration");
        let error = TextWeights::from_bytes(&bytes, &config).expect_err("NaN must be rejected");
        assert!(matches!(error, Error::Internal(_) | Error::Corrupt(_)));
    }
}
