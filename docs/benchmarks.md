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
