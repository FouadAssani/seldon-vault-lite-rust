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
(index two images and run a text vector query) takes about 176 s.

No intra-request parallelism, batching, or SIMD is implemented at M4. M7 must
profile the image path before optimizing, compare every change with the image
goldens and the preprocessing fingerprints, and record CPU, compiler, model
revision, memory, and parameters with the new numbers.
