# Text embedding benchmark — milestone M2 baseline

## Purpose

This is the scalar FP32 baseline for the M2 text encoder. It is not a
comparison with another engine and it is not an optimized result. M7 will use
it as the reference before introducing NEON or any other kernel.

## Environment

| Item | Value |
|---|---|
| Date | 2026-11-11 |
| CPU | Apple M2 Pro, 10 cores |
| Memory | 16 GiB |
| OS | macOS 26.5.1 (build 25F80) |
| Target | `aarch64-apple-darwin` |
| Rust | rustc 1.99.1, cargo 1.99.1 |
| Profile | release: `opt-level = 3`, thin LTO, one codegen unit |
| Model | `google/embeddinggemma-2` at `f9972ac22e696a105c482202646979e33e4b5a1e` |
| Weights | BF16 Safetensors, 1,488,915,288 bytes |
| Input | `What causes the northern lights?` |
| Prompt | `SearchQuery` |
| Output | 768 dimensions, L2-normalized |

## Sequential latency

Ten calls after one warm-up, measured with `std::time::Instant`:

| Metric | Value |
|---|---|
| p50 | 1.485 s |
| p95 | 1.504 s |
| min | 1.469 s |
| max | 1.504 s |

## Concurrent throughput

Four threads, five requests each, sharing one immutable `TextEmbedder`:

| Metric | Value |
|---|---|
| Requests | 20 |
| Threads | 4 |
| Wall time | 8.244 s |
| Throughput | 2.43 requests/s |

The concurrent rate is below the ideal 4x sequential rate (about 2.7
requests/s). The scalar forward pass is compute-bound and shares memory
bandwidth; this is the expected starting point, not an optimization claim.

## Cold start and resident memory

Measured with `/usr/bin/time -l` around one CLI `embed` invocation:

| Metric | Value |
|---|---|
| Wall time | 10.36 s |
| User time | 8.42 s |
| System time | 1.09 s |
| Maximum resident set size | 2,779,676,672 bytes (~2.59 GiB) |
| Peak memory footprint | 2,804,418,912 bytes (~2.61 GiB) |

The peak includes the 1.4 GiB Safetensors file bytes and the FP32 text weight
set (about 948 MiB), plus activations and the process. Loading and BF16 to
FP32 conversion dominate cold start; one forward pass is about 1.5 s after the
model is resident.

## Reproduction

```bash
cargo test --release --test text_bench -- --ignored --nocapture
cargo build --release --target aarch64-apple-darwin
/usr/bin/time -l target/aarch64-apple-darwin/release/seldon-vault-lite \
  embed --prompt SearchQuery "What causes the northern lights?"
```

## Interpretation

- These numbers are a baseline for the scalar reference, not a product
  performance claim.
- The 4-thread run shows that the embedding API is thread-safe, but no
  batching or intra-request parallelism is implemented at M2.
- Memory can be reduced later by streaming shards, by keeping weights in BF16
  with FP32 accumulation at the kernel boundary, or by quantization after
  parity validation. None of those changes is accepted before the M2 golden
  tests are replayed.
- M7 must compare every optimization with this baseline and with the golden
  embeddings before reporting a gain.

## Image path baseline — milestone M4

The image parity test measures one end-to-end image embedding with the frozen
model in the release profile. It is a correctness baseline, not an optimized
result.

| Item | Value |
|---|---|
| Date | 2026-11-11 |
| CPU | Apple M2 Pro, 10 cores |
| Memory | 16 GiB |
| OS | macOS 26.5.1 (build 25F80) |
| Target | aarch64-apple-darwin |
| Rust | rustc 1.99.1, cargo 1.99.1 |
| Profile | release: opt-level = 3, thin LTO, one codegen unit |
| Model | google/embeddinggemma-2 at f9972ac22e696a105c482202646979e33e4b5a1e |
| Input | fixtures/media/quadrants_16x16.png |
| Visual budget | 70 soft tokens |
| Resized size | 384x384 |
| Real patches | 576 |
| Soft tokens | 64 |
| Output | 768 dimensions, L2-normalized |
| Wall time | 90.36 s |

The wall time includes model loading, preprocessing, the 16-layer vision tower,
the text encoder over 68 tokens, and the parity assertions. The scalar vision
tower dominates: 16 layers over 576 patches with 12 heads of dimension 64 and
a 768-to-3,072 feed-forward. The default 280-token budget produces 2,304 real
patches, so the same scalar path is about 16 times larger and is not practical
yet.

The end-to-end CLI path (index one image at 70 tokens, then retrieve it with a
text vector query) takes about 100 s in the test profile. That measurement
includes model loading, preprocessing, the vision tower, one image embedding,
the text query embedding, and the journal write. The two-image ranking test
(index two images, run a text vector query, then run an image vector query)
takes about 271 s. The rotation/aspect vision parity test embeds a 2:1
aspect-ratio image and a rotated image in about 156 s.

No intra-request parallelism, batching, or SIMD is implemented at M4. M7 must
profile the image path before optimizing, compare every change with the image
goldens and the preprocessing fingerprints, and record CPU, compiler, model
revision, memory, and parameters with the new numbers.

## Audio path baseline — milestone M5

The audio parity path measures one end-to-end audio embedding with the frozen
model in the release profile. It is a correctness baseline, not an optimized
result.

| Item | Value |
|---|---|
| Date | 2026-11-11 |
| CPU | Apple M2 Pro, 10 cores |
| Memory | 16 GiB |
| OS | macOS 26.5.1 (build 25F80) |
| Target | aarch64-apple-darwin |
| Rust | rustc 1.99.1, cargo 1.99.1 |
| Profile | release: opt-level = 3, thin LTO, one codegen unit |
| Model | google/embeddinggemma-2 at f9972ac22e696a105c482202646979e33e4b5a1e |
| Output | 768 dimensions, L2-normalized |

| Input | Duration | Audio tokens | Wall time | Max RSS |
|---|---|---|---|---|
| `fixtures/media/audio/short_tone_100ms.wav` | 0.1 s | 3 | 16.25 s | 4,028,203,008 bytes (~3.75 GiB) |
| `fixtures/media/audio/rain_noise_2s.wav` | 2.0 s | 50 | 31.05 s | 4,037,902,336 bytes (~3.76 GiB) |

Both runs include loading the 1.49 GB Safetensors file, converting the text and
audio weights to FP32, the WAV decode, mono conversion and resampling, the
log-mel preprocessing, the 12-layer audio tower, the text encoder over the
audio prompt, and the final projection. The audio tower dominates: the 0.1 s
clip already produces three soft tokens, and the 2 s clip produces 50 tokens
through 12 layers of chunked attention and a 4,096-wide feed-forward. The
scalar path is not practical for 30 s clips yet; M7 profiling will use these
runs as the correctness baseline.

Reproduction:

```bash
cargo build --release --target aarch64-apple-darwin
/usr/bin/time -l target/aarch64-apple-darwin/release/seldon-vault-lite \
  embed --audio fixtures/media/audio/short_tone_100ms.wav --dim 768
/usr/bin/time -l target/aarch64-apple-darwin/release/seldon-vault-lite \
  embed --audio fixtures/media/audio/rain_noise_2s.wav --dim 768
```

Interpretation:

- The short clip's first vector values match the official golden exactly at the
  displayed precision; the full parity test checks cosine and absolute
  tolerance.
- Audio preprocessing is a small fraction of the time; the tower and weight
  loading dominate.
- Resident memory is about 3.76 GiB because the text and audio weight sets are
  held in FP32 simultaneously. The image path has a similar peak with the
  vision tower.
- No batching, SIMD, or quantization is applied at M5. Any optimization must
  replay the six-case audio parity tests and the text-to-audio ranking test
  before reporting a gain.

## NEON kernels — milestone M7

This section records the first M7 optimization step. The scalar path remains
the correctness oracle; the NEON path is selected at runtime and falls back to
scalar when the feature is unavailable. The frozen-model parity suites were
replayed after the change.

### Environment

| Item | Value |
|---|---|
| Date | 2026-11-11 |
| CPU | Apple M2 Pro, 10 cores |
| Memory | 16 GiB |
| OS | macOS 26.5.1 (build 25F80) |
| Target | `aarch64-apple-darwin` |
| Rust | rustc 1.99.1, cargo 1.99.1 |
| Profile | release: `opt-level = 3`, thin LTO, one codegen unit |
| Model | `google/embeddinggemma-2` at `f9972ac22e696a105c482202646979e33e4b5a1e` |

### Kernel microbenchmark

Best of three runs, release profile, measured with `std::time::Instant`:

| Operation | Scalar | NEON | Speedup |
|---|---:|---:|---:|
| dot, length 128 | 12.541 ms | 1.980 ms | 6.33x |
| dot, length 768 | 12.192 ms | 0.875 ms | 13.93x |
| dot, length 3072 | 12.696 ms | 0.827 ms | 15.35x |
| linear 1x768x768 | 84.252 ms | 8.205 ms | 10.27x |
| linear 16x768x3072 | 134.228 ms | 13.403 ms | 10.01x |
| linear 32x768x3072 | 160.838 ms | 15.893 ms | 10.12x |
| rms_norm_rows 32x768 | 32.449 ms | 13.651 ms | 2.38x |

The dot and linear cases are dominated by FMA throughput. The row-wise RMSNorm
case is limited by the per-row reduction and the weighted scale pass; it still
improves because the sum of squares and the scale are vectorized.

### End-to-end text embedding

Ten calls after one warm-up, same input and prompt as the M2 baseline:

| Metric | M2 scalar | M7 NEON | Speedup |
|---|---:|---:|---:|
| p50 | 1.485 s | 0.159 s | 9.3x |
| p95 | 1.504 s | 0.159 s | 9.5x |
| Concurrent throughput (4 threads, 20 requests) | 2.43 req/s | 24.47 req/s | 10.1x |

The concurrent run completed in 817 ms. The scalar path was compute-bound and
shared memory bandwidth; the NEON path removes most of the arithmetic cost, so
the remaining time is dominated by memory traffic and the non-vectorized
softmax, GELU, RoPE, and pooling tails.

### Parity replay

| Suite | Result | Wall time |
|---|---|---:|
| Text parity (frozen model) | pass | 12.98 s |
| Image parity (frozen model, 70 visual tokens) | pass | 22.02 s |
| Audio parity (short tone, frozen model) | pass | 15.03 s |

The text parity test checks the official golden vectors at 1e-4 absolute and
relative tolerance and 0.9999 cosine. The image and audio tests keep their
existing thresholds. No threshold was relaxed for M7.

### Reproduction

```bash
cargo test --release --lib neon_microbenchmark -- --ignored --nocapture
cargo test --release --test text_bench -- --ignored --nocapture
cargo test --release --test text_embeddings real_model_parity -- --nocapture
cargo test --release --test image_embeddings image_embedding_matches_the_official_reference -- --nocapture
cargo test --release --test audio_embeddings audio_embedding_matches_the_official_reference -- --nocapture
```

### Interpretation

- These are measured gains on the reference machine, not a claim that the
  engine is optimized for every Apple Silicon model.
- The NEON path is deterministic: the four-lane reduction order is fixed.
- The scalar path is still the oracle and is still tested.
- The container format and packer are implemented. Bundle-backed loading from
  a directory or a `.svlm` file is implemented and produces identical
  embeddings. Embedded `include_bytes!` loading and Mach-O size/startup
  measurements are the next M7 steps.

## Bundle-backed model loading — milestone M7

The frozen model was loaded once from the development directory and once from
the verified `.svlm` bundle. Both runs used the same release binary:

```bash
REV=f9972ac22e696a105c482202646979e33e4b5a1e
BIN=target/aarch64-apple-darwin/release/seldon-vault-lite
/usr/bin/time -l "$BIN" embed --model-dir ".model-cache/$REV" \
  --ids 2,5,6,7,1 --dim 8
/usr/bin/time -l "$BIN" embed --model-dir ".model-cache/$REV/model.svlm" \
  --ids 2,5,6,7,1 --dim 8
```

| Source | Wall time | User time | Maximum RSS | Output |
|---|---:|---:|---:|---|
| Model directory | 6.96 s | 6.38 s | 2,580,709,376 B (~2.40 GiB) | reference |
| `.svlm` bundle file | 7.40 s | 6.50 s | 2,617,540,608 B (~2.44 GiB) | identical bytes |

The bundle path pays for the header CRC-32, catalog digest, and section SHA-256
verification. The weights loader reuses the section digest from the bundle
header, so the 1.5 GB Safetensors section is not hashed a second time. Before
that reuse the bundle path took 13.49 s; the change removed the redundant pass
without changing the output.

The frozen bundle was also loaded through the image and audio embedders in
ignored release tests: 7.33 s for the image bundle and 7.36 s for the audio
bundle, both matching the frozen Safetensors fingerprint and the expected tower
layer counts.

These are single-process cold-start measurements on the reference Apple M2 Pro.
They include model loading, weight decoding, tokenizer loading, and the forward
pass for five token ids. The embedded `include_bytes!` path will be measured
separately when the final packaging step is implemented.

## Embedded tiny bundle — milestone M7

The `embedded-model` feature was tested with the 48,590-byte tiny text bundle:

```bash
SELDON_MODEL_BUNDLE="$PWD/.model-cache/tiny_text.svlm" \
  CARGO_TARGET_DIR=target/embedded \
  cargo build --release --features embedded-model --target aarch64-apple-darwin
target/embedded/aarch64-apple-darwin/release/seldon-vault-lite \
  embed --model-dir /nonexistent --ids 2,5,6,7,1 --dim 8
```

| Binary | Size | Model source | Output |
|---|---:|---|---|
| Default release | 1.4 MB | `fixtures/model/tiny_text` directory | reference |
| Embedded release | 1.5 MB | 48,590-byte embedded bundle | identical bytes |

The embedded binary ignored the nonexistent `--model-dir` and produced the same
8-dimensional vector.

### Full embedded build

The frozen 1,525,784,026-byte bundle was embedded into the release binary:

```bash
SELDON_MODEL_BUNDLE="$PWD/.model-cache/f9972ac22e696a105c482202646979e33e4b5a1e/model.svlm" \
  CARGO_TARGET_DIR=target/embedded-full \
  cargo build --release --features embedded-model --target aarch64-apple-darwin
```

| Item | Value |
|---|---|
| Build time | 2 m 24 s |
| Build directory | 10 GiB |
| Binary size | 1.4 GiB (1,525,784,026-byte bundle plus code) |
| Linked libraries | `/usr/lib/libSystem.B.dylib` only |
| Deployment target | `minos 11.0` |
| Code signature | ad-hoc, linker-signed |
| Gatekeeper | `spctl` rejects because the binary is not notarized |

| Run | Wall time | User time | System time | Max RSS | Peak footprint |
|---|---:|---:|---:|---:|---:|
| Cold (first run) | 18.68 s | 6.54 s | 1.31 s | 2,617,475,072 B (~2.44 GiB) | 1,091,750,048 B (~1.02 GiB) |
| Warm (second run) | 6.75 s | 6.52 s | 0.21 s | 2,617,458,688 B (~2.44 GiB) | 1,091,733,664 B (~1.02 GiB) |

The cold run pages the 1.4 GiB binary in from disk; the warm run is comparable
to the directory path (6.96 s) and the bundle-file path (7.40 s). The maximum
RSS includes the file-backed embedded bundle pages; the peak memory footprint
is about 1.02 GiB, dominated by the FP32 text weights and activations.

The embedded binary produced byte-identical 768-dimensional vectors to the
directory-loaded binary for:

- text token ids `2,5,6,7,1`;
- `fixtures/media/quadrants_16x16.png` at the 70-token budget;
- `fixtures/media/audio/short_tone_100ms.wav`.

The `spctl` rejection is expected for an ad-hoc linker-signed binary. Local
execution works without a developer account. Distribution to other Macs still
requires a Developer ID signature and notarization; that is a distribution
concern for M8, not a runtime dependency of the product.

## Clean-environment release validation — milestone M8

The full embedded artifact was exercised with `scripts/release_e2e.sh` on
2026-11-11 (report timestamp 2026-11-11T21:03:06Z) against the development
machine. The harness staged only the binary
and the committed media fixtures, ran with `env -i`, a temporary HOME and
TMPDIR, and a system-only PATH, and completed the text/image/audio embedding,
ingestion, lexical/vector/hybrid search, verify, compact, restart, and loopback
HTTP lifecycle.

| Item | Value |
|---|---|
| Platform | arm64 |
| macOS | 26.5.1 |
| Artifact | embedded full bundle |
| Binary SHA-256 | `b2e697b78602d8eaf82fc170b8d92056ff0088a38bcb77bbfd6f648cf52902e8` |
| Binary size | 1,539,215,600 bytes |
| Version | `seldon-vault-lite 0.1.0 (aarch64-apple-darwin)` |
| Signing | ad-hoc |
| Gatekeeper | rejected (expected for ad-hoc) |
| Network isolation | disabled in this run; `sandbox-exec` was unavailable in the execution sandbox |
| Documents after restart | 3 |

This is a development-machine run, not the physical clean-Mac gate. It records
the lifecycle and artifact properties; it is not a new performance claim. A
clean-Mac run must record the network-isolation mode as enabled and archive the
JSON report with the release candidate. The image and audio towers remain scalar
correctness baselines; their latency is dominated by the scalar path and is not
an optimized result.
