# ADR 0010 — Native WAV decoding, official audio preprocessing, and audio embeddings

- **Status:** Accepted for milestone M5.
- **Date:** 2026-11-11.
- **Context:** The plan requires a native audio path that produces real
  EmbeddingGemma 2 audio embeddings comparable with text and images. No
  third-party crate, external decoder, inference engine, or transcription tool
  may be used. The frozen model revision is
  `f9972ac22e696a105c482202646979e33e4b5a1e`.
- **Decision owners:** project maintainers.

## Context

Milestone M5 adds the audio modality. The frozen model ships a Gemma 4 audio
tower in `config.json` (`audio_config`), a `Gemma4AudioFeatureExtractor`
configuration in `preprocessor_config.json`, and audio token ids in the
composite configuration. The official implementation is available only as a
laboratory reference; the product must reproduce it in Rust `std`.

The audio path has three independent correctness surfaces:

1. **Container decoding.** WAV/RIFF files can contain PCM, IEEE float, or
   compressed codecs; the model processor expects 16 kHz mono samples.
2. **Feature extraction.** The official processor computes a 128-bin log-mel
   spectrogram with specific framing, windowing, FFT size, mel scale, floor,
   padding, and mask semantics.
3. **Encoder inference.** The Gemma 4 audio tower uses two stride-2
   convolution/normalization blocks, sinusoidal relative positions, chunked
   local attention with a relative key projection, clipped linear layers, a
   causal depthwise light convolution, two feed-forward branches, and a
   1536-dimensional output projection into the text hidden space.

A fake vector computed from a file name, duration, or textual marker would
violate the project contract. The audio path therefore fails closed: a decode,
preprocess, or inference failure is returned as a typed error.

## Decisions

### WAV/RIFF decoding

- Accept RIFF/WAVE with `WAVE_FORMAT_PCM` (format tag 1) at 8, 16, 24, or
  32 bits per sample, and `WAVE_FORMAT_IEEE_FLOAT` (format tag 3) at 32 or
  64 bits per sample.
- Accept `WAVE_FORMAT_EXTENSIBLE` (0xFFFE) only when its sub-format is PCM or
  IEEE float.
- Reject every other codec as unsupported. MP3, ADPCM, A-law, and mu-law are
  not guessed or approximated.
- Validate the RIFF size, every chunk boundary and padding byte, the format
  fields, block alignment, byte rate, data size, frame count, channel count,
  and sample rate before reading samples.
- Caps: 64 MiB file, 64 MiB data chunk, eight channels, 8 kHz to 192 kHz, and
  30 seconds at 192 kHz (5,760,000 source frames).
- Downmix channels by averaging. Convert PCM samples to FP32 in [-1, 1].
  Reject non-finite IEEE-float samples.
- Return mono FP32 at the source rate. The model-rate conversion is a separate,
  testable step.

### Conversion and resampling

- Resample to 16 kHz with a normalized windowed-sinc kernel: 32 taps, a Hann
  window, and a cutoff of `min(1, target/source)` to suppress aliasing when
  downsampling.
- Truncate to 480,000 samples (30 seconds at 16 kHz), matching the official
  feature extractor defaults `max_length=480000` and `truncation=True`.
- The conversion is deterministic and uses only `std`; it does not call the
  platform audio stack.

### Official log-mel preprocessing

- 16 kHz mono input.
- Frame length 320 samples (20 ms), hop 160 samples (10 ms), FFT length 512.
- Periodic Hann window `0.5 - 0.5 cos(2πn/320)`.
- Semicausal padding: prepend 160 zeros so the first frame is centered at
  `t=0`; the analysis window is 321 samples and the last sample is dropped by
  the no-preemphasis branch.
- 257 magnitude bins; 128 HTK mel filters from 0 Hz to 8000 Hz, no area
  normalization, floor `0.001`, natural logarithm.
- Right-pad the waveform to a multiple of 128 samples.
- A mel frame is valid only when the last sample of its 321-sample analysis
  window is real audio. Invalid frames are exactly zero, matching the official
  `speech * mask[..., None]` step.
- FFT and mel projection are computed in f64 and rounded to f32 once. The
  committed golden fixtures compare the first values and the final embeddings;
  the mel path is not claimed bit-exact because a different radix-2 FFT
  implementation can differ in the last f32 bit. The observed maximum absolute
  difference against the official features for a 0.1 s clip is
  `4.77e-7` and the mean absolute difference is `1.65e-8`.

### Audio tower

- Parse `audio_config` strictly: `gemma4_audio`, hidden size, 12 layers,
  eight heads, head dimension 128, convolution kernel 5, subsampling channels
  `[128, 32]`, chunk size 12, left context 13, right context 0, logit cap 50,
  RMSNorm epsilon 1e-6, output projection 1536, residual weight 0.5, gradient
  clipping 1e10, SiLU, and clipped linears. Float16 is forbidden.
- Load every audio tensor, including the scalar input/output clipping bounds,
  from the same frozen Safetensors file. Reject missing, unexpected, or
  mis-shaped tensors.
- Subsampling: two 3x3 convolution layers with stride 2 and padding 1, no
  bias, followed by channel LayerNorm (weight, no bias) and ReLU. The time mask
  is strided by two after each layer. The flattened feature layout is
  `[time, frequency, channel]`.
- Relative positions: sinusoidal `[sin..., cos...]` embeddings for positions
  `context_size/2 ... 0`, projected by a per-layer `relative_k_proj`.
- Chunked local attention: blocks of 12 queries; each query attends to valid
  keys with distance 0..11 (no future keys). Per-head scores are
  `q·k + q·relative_key`, scaled by `1/sqrt(head_dim)/ln 2` and
  `ln(1+e)/ln 2`, soft-capped with `tanh(score/50)*50`, masked, and
  softmaxed per head. The query scaling also includes
  `softplus(per_dim_scale)`.
- Light convolution: RMSNorm, clipped linear to 2h, GLU, causal depthwise
  convolution (kernel 5), RMSNorm, SiLU, clipped linear to h, residual.
- Feed-forward: RMSNorm, clipped linear to 4h, SiLU, clipped linear to h,
  RMSNorm, scale by 0.5, residual.
- Layer order: `feed_forward1`, pre-attention norm, attention, post-attention
  norm plus residual, light convolution, `feed_forward2`, output norm.
- Output: linear to 1536 with bias, unscaled RMSNorm, linear to the text hidden
  size (512). Only valid tokens are kept.
- Padding queries are skipped; the tower is causal along time, so trailing
  padding cannot influence a valid query. Captured layer probes select valid
  rows only.

### Embedding layout

- Audio-only prompt: `<bos><|audio><|audio|>*soft_tokens<audio|><eos>`.
- Replace the audio placeholder embeddings with the projected soft tokens;
  build placeholders from the padding token embedding, as the official model
  does.
- Mean-pool every token, including the begin/end markers and text special
  tokens, then L2-normalize.
- Matryoshka dimensions 768, 512, 256, and 128, renormalized after truncation.
- No transcription and no textual fallback.

### Storage, CLI, and index

- Accept `Modality::Audio` in the persistent store. The journal tag was
  already reserved as code 2.
- CLI `embed --audio <PATH>` prints a JSON vector.
- JSONL `index` documents accept exactly one of `text`, `image`, or
  `audio`. Audio documents store empty chunk text, so they participate in
  vector search but not BM25. The audio path becomes the default source.
- CLI `search --audio <PATH> --mode vector` embeds the query clip and ranks
  the stored audio documents in the shared space.
- Mixed batches load the image and/or audio embedder as needed. Text
  documents in a multimodal batch use the loaded embedder's text path.

### Validation

The audio golden fixture `fixtures/model/audio_golden.json` was generated
outside the product with Transformers 5.22.2 and PyTorch 2.15.0 CPU in
float32. It records six clips: silence, a short tone, a long tone, French
speech, a music chord progression, and rain-like noise. For each clip it
records the mel-frame count, valid-frame count, audio token count, input ids,
mel-feature fingerprint and first values, SSCP output fingerprint and first
values, all 12 encoder-layer probes, the projected audio features, and the
final 768-dimensional embedding.

Tolerances used by the Rust tests:

- mel first values: absolute `1e-5`;
- SSCP, layer, and projected-feature first values: absolute `2e-4` and
  relative `2e-4`;
- final embedding: cosine `>= 0.999` and maximum absolute difference
  `<= 2e-3`;
- exact equality for input ids, frame counts, token counts, and dimensions;
- unit norm for the full and truncated embeddings.

A tiny official fixture, `fixtures/model/tiny_audio/`, combines the committed
tiny text model with a deterministic two-layer audio tower (hidden 16, two
heads of dimension 8, subsampling channels [128, 8], chunk size 2, left context
3, output projection 16). Its golden records a deterministic 12-frame sine
input, the official log-mel features and mask, the SSCP output, both layer
outputs, the projected audio features, and the final normalized 8-dimensional
embedding. `tests/audio_tiny.rs` validates the audio configuration, weight
loader, SSCP, chunked attention, light convolution, feed-forward branches,
projection, and text connection without the frozen model cache.

The full six-case parity test and the text-to-audio ranking test are marked
`#[ignore]` because they load the frozen model; the default suite covers WAV
decoding, preprocessing, configuration, the tiny audio fixture, and the
short-clip parity test.

Observed ranking on the six-clip corpus (text query with the SearchQuery
prompt):

| Query | Top clip | Cosine |
|---|---|---|
| `rain falling` | `rain_noise_2s` | 0.7056 |
| `music` | `music_chords_2s` | 0.7181 |
| `a test of audio search` | `french_speech` | 0.7707 |

## Consequences

- The audio path is real and reference-validated; it is not a placeholder.
- The product can index and search WAV audio with text or audio queries in the
  shared embedding space.
- MP3, audio transcription, streaming audio, and broader audio ranking quality
  remain future work. The CLI and documentation must not claim "all audio".
- The scalar audio tower is CPU-heavy for long clips. M7 profiling will use
  this path as the correctness baseline before any SIMD optimization.
- The golden fixtures are laboratory-generated; the product never runs Python,
  Transformers, PyTorch, or a platform audio tool.

## Alternatives considered

- **Use an external audio decoder or resampler.** Rejected by the zero-crate and
  no-FFI contract.
- **Support every WAV codec.** Rejected: the official processor expects PCM or
  float samples, and guessing compressed formats would produce fake audio.
- **Skip resampling and reject non-16 kHz input.** Rejected: the plan requires
  conversions and resampling in the product.
- **Use a linear interpolation resampler.** Rejected: it aliases badly when
  converting 44.1/48 kHz to 16 kHz and would degrade embeddings. The
  windowed-sinc kernel is still small and deterministic.
- **Use only the first 30 seconds silently.** Accepted, because it matches the
  official feature extractor defaults. The cap is documented.
- **Compute attention for padding queries.** Rejected as wasted work; causal
  structure makes it irrelevant for valid tokens.
- **Declare audio supported from a file name or duration.** Rejected by the
  golden-rule contract.
