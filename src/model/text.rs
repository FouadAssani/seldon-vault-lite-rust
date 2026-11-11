//! EmbeddingGemma 2 text inference.
//!
//! This module implements the frozen text backbone in FP32 with scalar
//! kernels: scaled token embeddings, projection-only per-layer embeddings
//! (PLE), 24 bidirectional encoder layers with sliding or full attention,
//! grouped-query attention, per-head query/key/value RMSNorm, layer-specific
//! RoPE, gated GELU feed-forward blocks, the final RMSNorm, the 512 to 768
//! projection, mean pooling, L2 normalization, and Matryoshka truncation with
//! renormalization.
//!
//! The implementation is deliberately strict: an unknown activation, an
//! inconsistent layer configuration, an out-of-vocabulary token, or a
//! sequence longer than the documented 8,192-token context is rejected. No
//! fallback produces a plausible-looking embedding from a file name or a
//! textual marker.

use std::path::Path;

use crate::error::{Error, Result};
use crate::model::config::{LayerType, MAX_TEXT_TOKENS, TextConfig};
use crate::model::prompt::{self, Prompt};
use crate::model::tokenizer::Tokenizer;
use crate::model::weights::TextWeights;
use crate::tensor::kernels;
use crate::tensor::{Shape, Tensor};

/// Maximum number of FP32 elements in one activation buffer.
///
/// The largest documented sequence is 8,192 tokens. At hidden size 512, the
/// residual stream is 4.2 million elements; the attention and feed-forward
/// buffers are larger but still bounded. The cap leaves room for those
/// buffers while refusing an unbounded allocation.
pub const MAX_ACTIVATION_ELEMENTS: usize = 1 << 28;

/// Result of one forward pass.
#[derive(Debug)]
pub struct ForwardOutput {
    projected: Tensor,
    pooled: Vec<f32>,
    normalized: Vec<f32>,
    hidden_states: Vec<Tensor>,
}

impl ForwardOutput {
    /// Per-token projected hidden states, shape [sequence, embedding_dim].
    pub fn projected(&self) -> &Tensor {
        &self.projected
    }

    /// Mean-pooled projected vector before L2 normalization.
    pub fn pooled(&self) -> &[f32] {
        &self.pooled
    }

    /// L2-normalized embedding in the full output dimension.
    pub fn normalized(&self) -> &[f32] {
        &self.normalized
    }

    /// Captured hidden states: the scaled embedding output followed by the
    /// output of every layer. The vector is empty unless capture was requested.
    /// The projected embedding is available through projected().
    pub fn hidden_states(&self) -> &[Tensor] {
        &self.hidden_states
    }

    /// Returns the normalized embedding truncated to a Matryoshka dimension.
    ///
    /// The leading dimensions are kept and the result is L2-normalized again,
    /// as required by the model card.
    pub fn embedding(&self, dimension: usize) -> Result<Vec<f32>> {
        kernels::truncate_and_normalize(&self.normalized, dimension)
    }
}

/// FP32 text backbone: configuration plus validated weights.
#[derive(Debug)]
pub struct TextModel {
    config: TextConfig,
    weights: TextWeights,
}

impl TextModel {
    /// Loads a text model from a directory containing config.json and
    /// model.safetensors.
    ///
    /// # Errors
    ///
    /// Returns a typed error for a missing or invalid configuration, an
    /// oversized weight file, or a weight inventory that does not match the
    /// configuration.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config = TextConfig::from_file(&model_dir.join("config.json"))?;
        let weights = TextWeights::from_file(&model_dir.join("model.safetensors"), &config)?;
        Self::new(config, weights)
    }

    /// Builds a text model from a validated configuration and weight set.
    ///
    /// # Errors
    ///
    /// Returns an internal error if the weight inventory does not match the
    /// configuration. The public loader already rejects such files; this check
    /// protects direct construction from an inconsistent pair.
    pub fn new(config: TextConfig, weights: TextWeights) -> Result<Self> {
        if weights.layer_count() != config.num_hidden_layers() {
            return Err(Error::internal(format!(
                "weight set has {} layers, configuration declares {}",
                weights.layer_count(),
                config.num_hidden_layers()
            )));
        }
        Ok(Self { config, weights })
    }

    /// Model configuration.
    pub fn config(&self) -> &TextConfig {
        &self.config
    }

    /// Weight fingerprint recorded from the Safetensors file.
    pub fn fingerprint_hex(&self) -> &str {
        self.weights.fingerprint_hex()
    }

    /// Number of FP32 parameters loaded in memory.
    pub fn parameters(&self) -> usize {
        self.weights.parameters()
    }

    /// Runs one forward pass and returns the pooled embedding.
    ///
    /// # Errors
    ///
    /// Returns a corruption error for an empty or out-of-vocabulary sequence,
    /// a limit error for a sequence above the documented context, and an
    /// internal error for a non-finite numerical result.
    pub fn forward(&self, input_ids: &[u32], capture_hidden: bool) -> Result<ForwardOutput> {
        let sequence = input_ids.len();
        if sequence == 0 {
            return Err(Error::corrupt("cannot embed an empty token sequence"));
        }
        if sequence > MAX_TEXT_TOKENS {
            return Err(Error::limit_exceeded(format!(
                "sequence has {sequence} tokens, above the {MAX_TEXT_TOKENS}-token context"
            )));
        }
        if sequence > self.config.max_position_embeddings() {
            return Err(Error::limit_exceeded(format!(
                "sequence has {sequence} tokens, above the {} declared positions",
                self.config.max_position_embeddings()
            )));
        }

        let hidden_size = self.config.hidden_size();
        let ple_dim = self.config.hidden_size_per_layer_input();
        let layer_count = self.config.num_hidden_layers();
        let embedding_dim = self.config.embedding_dim();
        let epsilon = self.config.rms_norm_eps();
        let mut hidden = vec![0.0f32; checked_len(sequence, hidden_size, "embedding buffer")?];

        for (row, token) in input_ids.iter().enumerate() {
            let index = *token as usize;
            if index >= self.config.vocab_size() {
                return Err(Error::corrupt(format!(
                    "token id {token} is outside the vocabulary of {} entries",
                    self.config.vocab_size()
                )));
            }
            let source = &self.weights.embed_tokens[index * hidden_size..(index + 1) * hidden_size];
            let destination = &mut hidden[row * hidden_size..(row + 1) * hidden_size];
            for (target, value) in destination.iter_mut().zip(source) {
                *target = *value * self.config.embedding_scale();
            }
        }

        let mut hidden_states = Vec::new();
        if capture_hidden {
            hidden_states.push(tensor_from(sequence, hidden_size, &hidden)?);
        }

        // Projection-only per-layer embeddings. The projection is applied to
        // the scaled embeddings, then reshaped to [sequence, layers, ple_dim]
        // and normalized over the last dimension.
        let ple_width = checked_len(layer_count, ple_dim, "per-layer embedding width")?;
        checked_len(sequence, ple_width, "per-layer projection buffer")?;
        let mut per_layer = kernels::linear(
            &hidden,
            &self.weights.ple_per_layer_model_projection,
            sequence,
            hidden_size,
            ple_width,
        )?;
        let scale = self.config.ple_scale();
        for value in &mut per_layer {
            *value *= scale;
        }
        let per_layer = kernels::rms_norm_rows(
            &per_layer,
            &self.weights.ple_per_layer_projection_norm,
            sequence
                .checked_mul(layer_count)
                .ok_or_else(|| Error::limit_exceeded("per-layer row count overflow"))?,
            ple_dim,
            epsilon,
        )?;

        checked_len(
            sequence,
            self.config.intermediate_size(),
            "feed-forward buffer",
        )?;

        let has_full = self
            .config
            .layers()
            .iter()
            .any(|layer| layer.layer_type == LayerType::FullAttention);
        let has_sliding = self
            .config
            .layers()
            .iter()
            .any(|layer| layer.layer_type == LayerType::SlidingAttention);
        let full_inv_freq = if has_full {
            Some(self.config.rope_inv_freq(LayerType::FullAttention)?)
        } else {
            None
        };
        let sliding_inv_freq = if has_sliding {
            Some(self.config.rope_inv_freq(LayerType::SlidingAttention)?)
        } else {
            None
        };

        for (layer_index, layer) in self.weights.layers.iter().enumerate() {
            let layer_config = &self.config.layers()[layer_index];
            let head_dim = layer_config.head_dim;
            let num_heads = self.config.num_attention_heads();
            let num_kv_heads = layer_config.num_key_value_heads;
            let query_dim = checked_len(num_heads, head_dim, "query projection width")?;
            let key_value_dim = checked_len(num_kv_heads, head_dim, "key/value projection width")?;
            checked_len(sequence, query_dim, "query buffer")?;
            checked_len(sequence, key_value_dim, "key/value buffer")?;

            let residual = hidden.clone();
            let normalized = kernels::rms_norm_rows(
                &hidden,
                &layer.input_layernorm,
                sequence,
                hidden_size,
                epsilon,
            )?;
            let query =
                kernels::linear(&normalized, &layer.q_proj, sequence, hidden_size, query_dim)?;
            let key = kernels::linear(
                &normalized,
                &layer.k_proj,
                sequence,
                hidden_size,
                key_value_dim,
            )?;
            let value = kernels::linear(
                &normalized,
                &layer.v_proj,
                sequence,
                hidden_size,
                key_value_dim,
            )?;

            let query = kernels::rms_norm_rows(
                &query,
                &layer.q_norm,
                sequence
                    .checked_mul(num_heads)
                    .ok_or_else(|| Error::limit_exceeded("query row count overflow"))?,
                head_dim,
                epsilon,
            )?;
            let key = kernels::rms_norm_rows(
                &key,
                &layer.k_norm,
                sequence
                    .checked_mul(num_kv_heads)
                    .ok_or_else(|| Error::limit_exceeded("key row count overflow"))?,
                head_dim,
                epsilon,
            )?;
            let value = kernels::rms_norm_rows_unscaled(
                &value,
                sequence
                    .checked_mul(num_kv_heads)
                    .ok_or_else(|| Error::limit_exceeded("value row count overflow"))?,
                head_dim,
                epsilon,
            )?;

            let inv_freq = match layer_config.layer_type {
                LayerType::FullAttention => full_inv_freq.as_deref(),
                LayerType::SlidingAttention => sliding_inv_freq.as_deref(),
            }
            .ok_or_else(|| Error::internal("missing RoPE frequencies for a layer type"))?;
            let mut query = query;
            let mut key = key;
            kernels::rope_with_inv_freq(&mut query, sequence, num_heads, head_dim, inv_freq, 0)?;
            kernels::rope_with_inv_freq(&mut key, sequence, num_kv_heads, head_dim, inv_freq, 0)?;

            let sliding_window = match layer_config.layer_type {
                LayerType::FullAttention => None,
                LayerType::SlidingAttention => Some(self.config.sliding_window()),
            };
            let attention = kernels::bidirectional_attention(
                &query,
                &key,
                &value,
                kernels::AttentionShape {
                    seq_len: sequence,
                    num_heads,
                    num_key_value_heads: num_kv_heads,
                    head_dim,
                    sliding_window,
                    scale: 1.0,
                },
            )?;
            let attention =
                kernels::linear(&attention, &layer.o_proj, sequence, query_dim, hidden_size)?;
            let attention = kernels::rms_norm_rows(
                &attention,
                &layer.post_attention_layernorm,
                sequence,
                hidden_size,
                epsilon,
            )?;
            hidden = kernels::add(&residual, &attention)?;

            let residual = hidden.clone();
            let normalized = kernels::rms_norm_rows(
                &hidden,
                &layer.pre_feedforward_layernorm,
                sequence,
                hidden_size,
                epsilon,
            )?;
            let gate = kernels::linear(
                &normalized,
                &layer.gate_proj,
                sequence,
                hidden_size,
                self.config.intermediate_size(),
            )?;
            let up = kernels::linear(
                &normalized,
                &layer.up_proj,
                sequence,
                hidden_size,
                self.config.intermediate_size(),
            )?;
            let activated = kernels::geglu(&gate, &up)?;
            let down = kernels::linear(
                &activated,
                &layer.down_proj,
                sequence,
                self.config.intermediate_size(),
                hidden_size,
            )?;
            let down = kernels::rms_norm_rows(
                &down,
                &layer.post_feedforward_layernorm,
                sequence,
                hidden_size,
                epsilon,
            )?;
            hidden = kernels::add(&residual, &down)?;

            let residual = hidden.clone();
            let gate = kernels::linear(
                &hidden,
                &layer.per_layer_input_gate,
                sequence,
                hidden_size,
                ple_dim,
            )?;
            let gate = kernels::gelu_tanh(&gate);
            let mut gated = vec![0.0f32; checked_len(sequence, ple_dim, "PLE gate buffer")?];
            for row in 0..sequence {
                let gate_row = &gate[row * ple_dim..(row + 1) * ple_dim];
                let input_row = &per_layer[row * ple_width + layer_index * ple_dim
                    ..row * ple_width + (layer_index + 1) * ple_dim];
                let gated_row = &mut gated[row * ple_dim..(row + 1) * ple_dim];
                for ((destination, gate_value), input_value) in
                    gated_row.iter_mut().zip(gate_row).zip(input_row)
                {
                    *destination = gate_value * input_value;
                }
            }
            let projected = kernels::linear(
                &gated,
                &layer.per_layer_projection,
                sequence,
                ple_dim,
                hidden_size,
            )?;
            let projected = kernels::rms_norm_rows(
                &projected,
                &layer.post_per_layer_input_norm,
                sequence,
                hidden_size,
                epsilon,
            )?;
            hidden = kernels::add(&residual, &projected)?;
            let layer_scalar = layer.layer_scalar;
            for value in &mut hidden {
                *value *= layer_scalar;
            }

            if capture_hidden {
                hidden_states.push(tensor_from(sequence, hidden_size, &hidden)?);
            }
        }

        let hidden =
            kernels::rms_norm_rows(&hidden, &self.weights.norm, sequence, hidden_size, epsilon)?;
        let projected = kernels::linear(
            &hidden,
            &self.weights.embedding_projection,
            sequence,
            hidden_size,
            embedding_dim,
        )?;
        let projected = Tensor::new(Shape::new(vec![sequence, embedding_dim])?, projected)?;
        let pooled = kernels::mean_pool(projected.data(), sequence, embedding_dim)?;
        let normalized = kernels::l2_normalize(&pooled)?;
        Ok(ForwardOutput {
            projected,
            pooled,
            normalized,
            hidden_states,
        })
    }

    /// Embeds token ids and returns a Matryoshka-dimension vector.
    ///
    /// # Errors
    ///
    /// Returns an internal error when the requested dimension is zero or
    /// larger than the model output, before running the forward pass.
    pub fn embed_ids(&self, input_ids: &[u32], dimension: usize) -> Result<Vec<f32>> {
        if dimension == 0 || dimension > self.config.embedding_dim() {
            return Err(Error::internal(format!(
                "embedding dimension {dimension} is outside 1..={}",
                self.config.embedding_dim()
            )));
        }
        self.forward(input_ids, false)?.embedding(dimension)
    }
}

/// Text embedder: a text model plus the frozen tokenizer.
#[derive(Debug)]
pub struct TextEmbedder {
    model: TextModel,
    tokenizer: Tokenizer,
}

impl TextEmbedder {
    /// Loads a text embedder from a model directory containing config.json,
    /// model.safetensors, and tokenizer.json.
    ///
    /// # Errors
    ///
    /// Returns a typed error for an invalid configuration, a missing or
    /// inconsistent weight tensor, an oversized file, or a tokenizer whose
    /// vocabulary does not match the model.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let model = TextModel::load(model_dir)?;
        let tokenizer = Tokenizer::load(&model_dir.join("tokenizer.json"))?;
        if tokenizer.vocab_size() != model.config().vocab_size() {
            return Err(Error::corrupt(format!(
                "tokenizer vocabulary has {} entries, model declares {}",
                tokenizer.vocab_size(),
                model.config().vocab_size()
            )));
        }
        Ok(Self { model, tokenizer })
    }

    /// Builds an embedder from an already loaded model and tokenizer.
    pub fn from_parts(model: TextModel, tokenizer: Tokenizer) -> Self {
        Self { model, tokenizer }
    }

    /// Text model.
    pub fn model(&self) -> &TextModel {
        &self.model
    }

    /// Frozen tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Embeds raw text with the official prompt prefix.
    ///
    /// The prefix is applied before tokenization, so a query and a document
    /// never share an accidental embedding.
    pub fn embed(&self, text: &str, prompt: Prompt, dimension: usize) -> Result<Vec<f32>> {
        let full = prompt::apply(prompt, text);
        let ids = self.tokenizer.encode(&full, true)?;
        if ids.len() > MAX_TEXT_TOKENS {
            return Err(Error::limit_exceeded(format!(
                "tokenized input has {} tokens, above the {MAX_TEXT_TOKENS}-token context",
                ids.len()
            )));
        }
        self.model.embed_ids(&ids, dimension)
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
