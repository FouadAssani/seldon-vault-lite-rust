# ADR 0004 — Compliant text embeddings for EmbeddingGemma 2

- **Status**: accepted
- **Date**: 2026-11-11
- **Milestone**: M2
- **Decision maker**: project maintainer
- **Supersedes**: none
- **Superseded by**: none

## Context

M1 froze `google/embeddinggemma-2` at revision
`f9972ac22e696a105c482202646979e33e4b5a1e` and proved exact tokenization.
M2 must turn token ids into text embeddings that match the official
implementation, not merely a plausible vector. The model is an encoder with
several non-standard elements that must be reproduced exactly:

- 24 bidirectional layers, hidden size 512, output dimension 768;
- projection-only per-layer embeddings (PLE) mixed into every layer;
- sliding attention with a radius of 512 on 23 layers and full attention on
  layers 5, 11, 17, and 23;
- 4 query heads with 2 key/value heads for sliding layers, and 4 query heads
  with 1 key/value head and head dimension 512 for full layers;
- per-head query, key, and value RMSNorm; the value norm has no learned scale;
- layer-specific RoPE (theta 10,000 for sliding, 1,000,000 for full);
- attention scale 1.0, not 1/sqrt(head_dim);
- gated GELU feed-forward blocks and a per-layer scalar;
- final RMSNorm, a 512 to 768 projection, mean pooling, L2 normalization, and
  Matryoshka truncation with renormalization.

The official model card forbids FP16 for activations: the activation range
exceeds FP16's dynamic range. The engine therefore keeps every activation in
FP32 and converts BF16 storage weights to FP32 once at load time.

## Decisions

### D1 — Exact architecture

The text encoder is implemented in `model::text` with scalar FP32 kernels.
The forward pass follows the official Transformers implementation:

1. token embedding lookup, multiplied by `sqrt(hidden_size)`;
2. projection-only PLE: a linear projection of the embeddings, scaled by
   `1/sqrt(hidden_size)`, reshaped to [sequence, layers, 512], and normalized
   over the last dimension;
3. for each layer: pre-attention RMSNorm, query/key/value projections,
   per-head RMSNorm, layer-specific RoPE, bidirectional grouped-query
   attention, output projection, post-attention RMSNorm, residual;
4. pre-feed-forward RMSNorm, gate and up projections, GELU(tanh) gate times
   up, down projection, post-feed-forward RMSNorm, residual;
5. PLE block: gate, GELU, elementwise multiply by the layer's PLE slice,
   projection, RMSNorm, residual, then multiply by the layer scalar;
6. final RMSNorm, 512 to 768 projection, mean pooling over every non-padding
   token (the prompt is included), and L2 normalization;
7. Matryoshka truncation to 768, 512, 256, or 128 dimensions, followed by a
   second L2 normalization.

The attention mask is symmetric: a sliding layer at position i attends to
every position j with `|i - j| <= sliding_window`. This is the inclusive
radius documented by the configuration, not the causal one-sided window of
the generative Gemma models.

### D2 — Strict weight loading

The loader reads the frozen Safetensors file, ignores audio and vision
tensors, and accepts only the 413 text tensors under the `language_model.`
prefix. It verifies every expected name and shape before decoding, rejects
extra text tensors, missing tensors, shape mismatches, and non-finite values,
and converts BF16 or F32 storage to FP32. The source file's SHA-256 is
retained so tests can compare it with the manifest.

Limits:

- model file: 4 GiB;
- FP32 weight elements: 2^30;
- tensor element count: 2^30, inherited from the M1 tensor core.

### D3 — Tokenization and prompts

Text is prefixed with the official prompt before tokenization. A query and a
document are distinct roles; using the wrong prefix silently changes the
embedding space. The tokenizer's post-processor adds `<bos>` and `<eos>`.
The engine rejects an empty token sequence, an out-of-vocabulary token, and
any sequence above the documented 8,192-token context. Token ids remain exact;
any divergence from the golden corpus blocks embedding parity.

### D4 — Pooling, projection, and Matryoshka dimensions

The reference SentenceTransformers pipeline uses mean pooling with
`include_prompt = true`, then L2 normalization. Projecting per token and
pooling afterwards is equivalent to pooling the final hidden states before
the projection. The engine follows the official per-token projection and then
pools, which keeps the layer probe comparable with the reference.

Truncation keeps the leading dimensions and renormalizes. A slice of a unit
vector is not unit length; skipping the second normalization degrades cosine
ranking silently.

### D5 — Numerical policy and tolerances

The FP32 scalar path is the reference. There is no SIMD or `unsafe` in M2.
Tests compare:

- hidden states and projected states with absolute tolerance `2e-5` and
  relative tolerance `1e-4`;
- final normalized embeddings with absolute tolerance `1e-4` and cosine
  similarity `>= 0.9999`;
- token ids and dimensions exactly.

The tolerances allow different accumulation orders in scalar code while
remaining tight enough to catch a wrong normalization, mask, RoPE base, or
projection. The initial M1 target of `atol <= 1e-5` was adjusted after
measurement: the scalar path matches the official layer outputs within a few
units in the last place, but a strict `1e-5` absolute bound is brittle for
values of magnitude above 10. The documented values are therefore `2e-5`
absolute and `1e-4` relative for intermediate tensors. The projected
diagnostic state uses `2e-4` absolute because the 512-product projection
amplifies accumulation-order differences; the product embedding keeps the
`1e-4` absolute bound and the `0.9999` cosine floor.

### D6 — Determinism and concurrency

The model is immutable after loading: `TextModel` and `TextEmbedder` take
`&self` for inference and are `Send + Sync`. The CLI performs one request at
a time. A bounded request pool and the HTTP job queue belong to M6; M2
provides a thread-safe embedding API and a multithreaded benchmark to measure
the latency/throughput trade-off without claiming a service that does not
exist.

### D7 — Verification

The committed fixtures are:

- `fixtures/model/text_golden.json`: eight text cases (English query and
  document, French, code, empty, Unicode, classification, clustering), each
  with token ids and normalized embeddings at 768, 512, 256, and 128
  dimensions; plus a layer probe with the embedding output, all 24 layer
  outputs, and the final projected state for a short text.
- `fixtures/model/tiny_text/`: a complete 3-layer model with 56 tensors,
  generated with the official implementation, plus its golden layer outputs
  and embedding. It exercises sliding and full attention, GQA, PLE, two RoPE
  bases, and Matryoshka truncation without the 1.5 GB weights.

The tiny fixture runs on every `cargo test`. The full-model parity test runs
when `.model-cache/<revision>/model.safetensors` is present; it verifies the
weights' SHA-256 against the manifest and replays the golden corpus. Setting
`SELDON_SKIP_MODEL_PARITY=1` skips the heavy test during a quick check.

## Consequences

### Positive

- Text embeddings are produced by the real architecture, not a surrogate.
- The strict inventory rejects a mislabeled or truncated weight file before
  any computation.
- The tiny fixture makes architecture regressions visible without model-scale
  data.
- Matryoshka truncation cannot silently skip renormalization.

### Costs and risks

- The scalar forward pass is slow in debug builds; the test profile is
  optimized to keep the parity suite practical.
- The full model test requires the 1.5 GB development asset, which is not in
  Git.
- No SIMD, quantization, or batching optimization is claimed at M2.
- The model is specialized to this revision; a weight or processor change
  requires a new manifest and new golden fixtures.

### Follow-up

- **M3**: persistent storage, BM25/KNN index, and hybrid RRF search.
- **M4/M5**: PNG/JPEG and WAV decoders with multimodal encoders.
- **M6**: HTTP API, bounded job queue, and hardening.
- **M7**: NEON kernels compared with the scalar oracle before any timing
  claim.
