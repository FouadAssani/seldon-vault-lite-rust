# Seldon Vault Lite — 100% Rust Implementation Plan, No Third-Party Dependencies

**Version:** 1.0 — November 11, 2026
**Goal:** build, from base formats and algorithms, a fully local, CPU-only multimodal embedding, indexing, and search engine specialized for `google/embeddinggemma-2`, distributed as a standalone executable per platform. **First target: macOS Apple Silicon only (ARM64); Linux x86-64 and ARM64 next. No Intel macOS support.**

> This document is a development roadmap, not a claim that numerical compatibility, performance, or binary self-sufficiency has already been achieved.

## 0. The challenge contract

### What "without dependencies" means here

1. **100% Rust product code**: no C, C++, Go, Python, external assembly, FFI, unjustified `unsafe`, or dynamic application library.
2. **Zero third-party crates**: the production `Cargo.toml` declares no dependencies, including `dev-dependencies` and `build-dependencies`. `std` and `core::arch` intrinsics are allowed.
3. **Zero external inference engine**: no llama.cpp, Candle, ONNX Runtime, PyTorch, or inference server.
4. **Zero external database**: no SQLite or DB server; storage, journal, vector index, and text search are developed in Rust.
5. **Zero service or network required at runtime**: the product works offline, without a key or download.
6. **One CPU executable per OS/architecture pair** containing the engine, processing resources, and model weights. Indexed data remains persisted in a working directory: "a single binary" does not mean "no data files".
7. **No universal binary portability promise**: deliver `aarch64-apple-darwin` first (Apple Silicon Mac only), then `x86_64-unknown-linux-musl` and `aarch64-unknown-linux-musl`. The `x86_64-apple-darwin` target is explicitly excluded. A macOS executable cannot be entirely detached from the system libraries provided by macOS; this is not an application dependency to install.
8. **Allowed development tools**: Rust/Cargo compiler, Git, debugging tools, comparison scripts, and reference implementation *laboratory-only*. None is part of the delivered product.

### Final functional scope

- Text and code: query and document embeddings.
- Images: PNG and JPEG support, with native preprocessing and visual encoder.
- Audio: WAV PCM support first, then MP3 if the final format matrix requires it; **native** audio encoding, without transcription conversion.
- Comparable cross-modal embeddings: text ↔ image, text ↔ audio, and media ↔ media.
- Dimensions: 768, 512, 256, and 128, with renormalization after truncation.
- Documents, metadata, filters, persistent indexing, and hybrid search.
- Local HTTP API, resource control, diagnostics, clean shutdown, and recovery after interruption.
- No generative LLM, no transcription, no GPU, and no external orchestration.

### Output rules

A step is complete only if its tests pass, its results are reproducible, and no regression in the already-validated scope is observed. Never declare a modality "supported" if it only returns a vector computed from a file name or a textual marker.

## 1. Prepare the reference repository and quality criteria

1. Create the `seldon-vault-lite` repository, the development branch, and the Rust tree.
2. Create a `Cargo.toml` **without** populated `[dependencies]`, `[dev-dependencies]`, and `[build-dependencies]`.
3. Define **macOS Apple Silicon as the single initial platform** with `aarch64-apple-darwin`. Exclude any Intel macOS compilation or validation; defer `x86_64-unknown-linux-musl` then `aarch64-unknown-linux-musl` to the Linux port phase.
4. Define the minimum macOS version for Apple Silicon (macOS 11 or later depending on the capabilities used), then verify the compiled binary's actual `MACOSX_DEPLOYMENT_TARGET`.
5. Write an ADR for the choices: `std` only, formats, persistent data, `unsafe` policy, model versions, and compatibility.
6. Set the invariants: no crash on hostile input, explicit memory bounds, error handling without `unwrap()` on external input, deterministic scalar computations.
7. Add reproducible verification commands: `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test`, `cargo build --release`, and inspection of the Cargo dependency closure.

**Validation:** minimal compilable CLI application, standard unit tests, empty Cargo graph, and no linkage to an additional application library.

## 2. Freeze the model and produce a ground-truth reference

1. Freeze the official `google/embeddinggemma-2` repository at a precise revision and record the SHA-256 of **all** files used.
2. Study the exact parameters (`config.json`, `tokenizer.json`/tokenization model, processor configuration, Safetensors weights, prompts, and multimodal rules).
3. Write an inventory of layers, dimensions, numeric types, expected tensors, and weight names. Do not assume a generic "Gemma 4" model is compatible.
4. Build *golden tests* outside the product with the official implementation: tokens, intermediate layer outputs, and final vectors, for French/English texts, code, empty/Unicode strings, varied images, and WAV excerpts.
5. Record the expected inputs and outputs as small versioned fixtures, with provenance and fingerprints; do not commit giant weights to Git.
6. Define distinct tolerances: exact comparison for tokenization and dimensions, absolute/relative tolerance for FP32 tensors, cosine and search rank for quantized embeddings.

**Validation:** frozen model manifest, documented reference bench, and a "golden" corpus covering each modality. This step precedes optimization.

## 3. Develop formats and binary primitives in Rust

1. Create a bounded byte reader with systematic checks of sizes, offsets, overflows, and endianness.
2. Implement a strict JSON parser limited to the necessary features but compliant for objects, arrays, numbers, strings, escapes, and Unicode characters; no naive string-splitting parser.
3. Implement a Safetensors reader: JSON header, metadata, data views, names, shapes, strides, and overlap/offset verification.
4. Add BF16 ↔ FP32 conversions and reading of F32/BF16 elements; **do not use FP16 as activation precision**.
5. Provide a multi-file/shard loader and fingerprint verification before use.
6. Define the versioned internal format for embedded weights, but do not impose it on inference until parity with the original weights is demonstrated.

**Validation:** corruption, truncation, extreme-size, invalid-character, field-order, and out-of-bounds memory access tests. No code execution from metadata.

## 4. Build the CPU compute core

1. Define `Tensor`, `Shape`, `DType`, `TensorView`, and bounded allocations, without hidden implicit copies.
2. Implement a reference FP32 scalar path: addition, multiplication, matmul, dot products, stable softmax, RMSNorm, GELU/GeGLU, RoPE, attention, and L2 normalization.
3. Add shape validation and prevent `NaN`, division by zero, unstable accumulations, and index overflows.
4. Design a reusable activation buffer so as not to allocate at every layer.
5. Add a CPU scheduler in Rust with `std::thread`, bounded work queue, cooperative shutdown, and configurable pool size.
6. Compare operation results against the reference model fixtures; optimize only after agreement.

**Validation:** every operation has numerical tests, pathological inputs, and benchmarks; scalar mode is the correctness reference.

## 5. Reproduce tokenization and instructions exactly

1. Identify the algorithm actually declared in the model files and implement **that** variant, with its normalization and whitespace rules.
2. Implement UTF-8, special tokens, padding, masks, and context limits per the official configurations.
3. Reproduce the task prefixes `SearchQuery`, `Document`, similarity, classification, and other retained uses.
4. Preserve document titles in the intended prompt; explicitly distinguish a query from an indexed document.
5. Compare token IDs against the reference, including accented French, emoji, code, and long inputs.

**Validation:** exact token ID equality on the reference corpus. Any divergence blocks the integration of inference layers.

## 6. Implement EmbeddingGemma 2 text

1. Load the text model weights and verify names, shapes, and types of all tensors.
2. Implement the exact architecture: embeddings, per-layer inputs, local/global attention mechanisms, RoPE, normalizations, gates, and projections.
3. Respect the 24 layers, sliding-window behavior, and details specific to the Gemma 4/EmbeddingGemma 2 architecture, including non-standard elements noted in official sources.
4. Perform pooling and the 512 → 768 projection in accordance with the model.
5. Add L2 normalization and Matryoshka dimensions 768/512/256/128, **with renormalization after reduction**.
6. Support concurrent requests in a bounded way and measure the latency/throughput/memory trade-off.

**Validation:** match layer outputs and reference embeddings, then consistent ranking of a text/code corpus. No unvalidated approximate result is accepted as "compatible".

## 7. Implement the image path entirely in Rust

1. Develop PNG (with native DEFLATE/zlib) and JPEG decoders limited first to the announced variants, then expanded according to fixtures.
2. Impose caps on dimensions, pixel count, decompressed memory, and processing time to avoid decompression bombs.
3. Reproduce the official preprocessing: color conversion, resizing, patch placement, positions, and visual token budgets.
4. Implement the image encoder, projections, and its connection to the common embedding space.
5. Test several different images, rotations, and aspect ratios, then **text → image** search.

**Validation:** expected differences between distinct images, compatibility with reference vectors, and cross-modal results. No use of an external program to decode images.

## 8. Implement the audio path entirely in Rust

1. Implement WAV/RIFF PCM (explicitly announced formats and rates). Provide conversions and resampling in the product.
2. Implement exactly the official preprocessing (windowing, spectral transform, frequency representation, channels, masks, and segmentation), **after inspecting the official processor**.
3. Implement the audio encoder and its projection into the same space as text and image.
4. Test silence, short sounds, long sounds, French speech, music, and environmental sounds; compare against the reference.
5. If MP3 is a required final format, add a native MP3 decoder after WAV stabilization and apply the same limits/fixtures. Never announce "all audio".

**Validation:** audio vectors compliant with golden tests and functional text → audio segment search, without Whisper or transcription.

## 9. Optimize the engine without sacrificing correctness

1. Profile the CPU on real texts, images, and sounds: tokenization, preprocessing, matmul, attention, loading, and concurrent request times.
2. Prioritize macOS Apple Silicon AArch64 kernels via NEON with `core::arch`, while keeping a scalar path. AVX2/FMA kernels will only be added in the Linux x86-64 port. Do not depend on the Accelerate framework or Metal, since the challenge requires CPU Rust without FFI.
3. Add CPU capability detection and runtime kernel selection, without illegal instructions on older processors.
4. Experiment with blocking/cache locality, parallelism, and buffer reuse.
5. Design, if necessary, clean weight quantization (Q8 first), with **safe accumulations/activations** and embedding drift control. Quantization is not a given.
6. At each optimization, verify golden tests and search ranking before comparing times against scalar mode and an external reference.

**Validation:** measured gain on target machines, zero numerical regression beyond threshold; benchmark results archived with CPU, compiler, model revision, memory, and parameters.

## 10. Develop native persistent storage

1. Define a model for documents, chunks, vectors, modalities, sources, ACLs/filters, and metadata; stable IDs and explicit schema versions.
2. Define a portable storage format with framed records, bounded lengths, checksums, and version.
3. Develop an append-only journal with atomic commit, `fsync` at critical points, deterministic replay, and recovery after a brutal stop.
4. Add snapshots, safe compaction, writer locking, and concurrent reads.
5. Add an inverted lexical index with BM25, normalization, filters, and positions if necessary.
6. Start with exact vector search (cosine/normalized dot product), which will serve as the correctness oracle; introduce native HNSW only after volume/latency measurements.
7. Implement hybrid RRF fusion (lexical + vector), pagination, and consistent deletions.

**Validation:** crash-injection tests at every write step, identical recovery, no duplicates/losses, deterministic filters and scores; HNSW comparison with the exact oracle when it exists.

## 11. Develop the HTTP API and CLI in Rust `std`

1. Create a `std::net::TcpListener` HTTP/1.1 server with strict parsing, bounded connections, timeouts, validated `Content-Length`, and maximum body size.
2. Implement the API contract JSON with the internal parser; documented HTTP codes and error responses.
3. Expose:
   - `GET /health` — process alive;
   - `GET /ready` — weights loaded, storage ready;
   - `GET /v1/models` — version/fingerprint, modalities, and dimensions;
   - `POST /v1/embeddings` — text and media via documented schema;
   - `POST /v1/documents` — ingestion and metadata;
   - `DELETE /v1/documents/{id}` — complete deletion;
   - `POST /v1/search` — hybrid search and filters;
   - `GET /v1/jobs/{id}` — ingestion status.
4. Implement a bounded ingestion queue, concurrency limiting, cancellations, graceful shutdown, and local observability.
5. Bind by default to `127.0.0.1`; for LAN, require explicit configuration, an access token, and strict request limits. Without external TLS, do not expose directly to the Internet.
6. Add a native CLI (`serve`, `embed`, `index`, `search`, `verify`, `version`); do not launch an auxiliary process.

**Validation:** integration tests for all routes, malformed inputs, Unicode JSON, slow connections, large requests, and concurrency; no panic on network error.

## 12. Build the standalone single executable

1. During development, keep model files outside the binary to allow fast iterations.
2. Add a packer **also written in Rust `std`** that freezes weights, tokenizer, processors, and manifest into a versioned and verified internal container.
3. Embed this container at final compile time (`include_bytes!` or equivalent Rust mechanism), with no download at first startup.
4. Test the real limits of the Mach-O linker, startup, size, and RAM on **Apple Silicon Mac only**. BF16 weights alone are on the order of 1.5 GB; activations, indexes, and buffers increase memory consumption.
5. Produce and test **first** the macOS `aarch64-apple-darwin` binary. Verify its linked libraries and that no third-party application library is required. Also check signing/Gatekeeper restrictions in distribution scenarios, without requiring a developer account for local use.
6. Only then, port to static Linux x86-64 and ARM64 MUSL and verify the absence of unauthorized dynamic libraries.
7. Keep **only** user data, logs, and snapshots as external persistent files.

**Validation:** copy the single executable to a clean, offline machine, start the service, index text/an image/a WAV, search, stop and restart without data loss or external model files.

## 13. Full test campaign before release

### Mandatory tests

- **Correctness:** token ID parity, intermediate computations, and final vectors against golden tests.
- **Multimodal:** text query returning a relevant image; text query returning an audio excerpt; image↔image search.
- **Dimensions:** 128/256/512/768, truncation followed by normalization; same space for queries and documents.
- **Persistence:** ingestion, modification, deletion, crash, replay, snapshot, restart, and restore.
- **Robustness:** corrupted images/audio, arbitrary data, size limits, invalid JSON, cancellation, and out of memory.
- **Security:** internal fuzzing from a home-made Rust corpus generator; analysis of parsers and `unsafe` SIMD regions; LAN exposure tests.
- **Performance:** p50/p95/p99, resident RAM, cold start, throughput, vector scalability, and saturated CPU.
- **Distribution:** macOS Apple Silicon only in the first phase, no third-party crates, absence of unauthorized dynamic application library, model included, no network at runtime. Test on a clean Apple Silicon Mac without Homebrew, Python, or an installed inference tool. Then test each Linux architecture separately.

### Definition of "done"

A single binary per architecture, **validated first only on macOS Apple Silicon**, usable offline; **real** text/image/audio embeddings compliant with references; persistent hybrid search; documented and secure API for local use; recovery after incident; measured performance; reproducible tests; readable source and respected licenses. The Linux x86-64/ARM64 port comes next, without an Intel macOS target.

## 14. Delivery order — milestones without shortcuts

| Milestone | Observable result | Unlocks |
|---|---|---|
| M0 | Rust `std`-only repository + ADR + reference datasets + macOS ARM64 build | Verifiable development on Apple Silicon |
| M1 | Safetensors/JSON/tokenization + FP32 tensors | First inference |
| M2 | Compliant text embeddings | Semantic text search |
| M3 | Persistent index/storage + hybrid search | Useful text product |
| M4 | PNG/JPEG images and compliant visual embeddings | Multimodal image search |
| M5 | WAV and compliant audio embeddings | Multimodal audio search |
| M6 | HTTP API, jobs, and hardening | Complete local service |
| M7 | NEON SIMD and verified macOS ARM64 packaging | macOS Apple Silicon distribution |
| M8 | End-to-end tests on a clean Apple Silicon Mac and final documentation | macOS 1.0 release |
| M9 | Linux x86-64/ARM64 MUSL port, tests, and documentation | Linux release |

**Indicative estimate:** for one person, several months at minimum; 9 to 18 months or more are plausible for a robust, optimized, and verified multimodal engine, with decoders and storage written entirely. The riskiest phases are image/audio encoder parity, quantization, media parsers, and fault tolerance.

## 15. Suggested final tree

```text
seldon-vault-lite/
├── Cargo.toml                 # zero dependencies
├── Cargo.lock
├── README.md
├── docs/
│   ├── adr/
│   ├── architecture.md
│   ├── model-parity.md
│   ├── api.md
│   └── formats.md
├── src/
│   ├── main.rs
│   ├── api/                  # native HTTP/1.1 + JSON
│   ├── cli/
│   ├── binary/               # Safetensors, headers, checksums
│   ├── tensor/               # scalar and SIMD kernels
│   ├── model/                # tokenizer, Gemma 4/EmbeddingGemma 2
│   ├── media/                # PNG, JPEG, WAV, optional MP3
│   ├── index/                # BM25, KNN, optional HNSW, RRF
│   ├── storage/              # journal, snapshots, transactions
│   ├── jobs/
│   └── error.rs
├── tests/                    # Rust std tests only
├── fixtures/                 # compact golden tests
└── tools/
    └── model-packer/         # Rust std only, used at build time
```

## 16. Risks to keep visible

1. **Zero crate** greatly increases maintenance load and bug surface in HTTP, JSON, PNG/JPEG, audio, index, and persistence: a serious security review becomes imperative.
2. **Specialized model** means a major change in weight structure or processor requires an engine update; freezing the model is a feature, not a hidden weakness.
3. **FP16 forbidden** for activations per the official documentation; start with FP32, then study BF16 and weight quantization with validation.
4. **A single executable file ≠ zero disk writes**: indexed documents and the WAL must survive restarts.
5. **No application dependency ≠ OS or CPU independence**: network, files, threads, and memory necessarily go through platform services.
6. **Performance to prove**: Rust is not automatically faster than llama.cpp; the metric must be performance at comparable quality.

## Starting technical references

- [Google — official EmbeddingGemma 2 model card](https://github.com/google-gemma/docs/blob/main/docs/models/embeddinggemma/model-card-v2.md): architecture, formats, MRL dimensions, precision rules, and prefixes.
- [Google — original EmbeddingGemma 2 weights](https://huggingface.co/google/embeddinggemma-2): configuration, processors, and checkpoints.
- [Dalailalama — EmbeddingGemma 2 in Rust](https://github.com/Dalailalama/embeddinggemma2-rust): independent implementation to study, without making it a product dependency.
- [Rust — official `std` documentation](https://doc.rust-lang.org/std/): allowed primitives.
- [Rust — official macOS targets](https://doc.rust-lang.org/rustc/platform-support/apple-darwin.html): Apple Silicon and minimum versions.

**Guiding principle:** start by proving text correctness, then extend to media, then only optimize and embed the weights into the executable. Every step must be a reproducible demonstration, not just a successful compilation.
