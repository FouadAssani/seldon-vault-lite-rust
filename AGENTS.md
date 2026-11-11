# AGENTS.md — Seldon Vault Lite

**Project:** local multimodal embedding, indexing, and search engine, 100% Rust, with no third-party dependencies, specialized for `google/embeddinggemma-2`.
**Reference document:** [`seldon-vault-lite-plan.md`](seldon-vault-lite-plan.md). In case of doubt or conflict, the plan is authoritative. This file is the operational summary intended for coding agents and contributors.
**Language:** English only. All project artifacts — documentation, ADRs, code comments, commit messages, CLI help, API messages, and test names — are written in English. Vendored third-party skills under `.dsh/skills/` are exempt.

## 0. Golden rule

A step is complete only if **its tests pass, its results are reproducible, and no regression is observed in the scope already validated**. A successful compilation is not proof. Never declare a modality "supported" if it only returns a vector computed from a file name or a textual marker.

State at the time of writing (2026-11-11): milestone **M0** is complete; milestone **M1** is complete on `main` with the bounded reader, SHA-256, BF16 conversion, strict JSON, Safetensors reader, FP32 tensors, scalar kernels, frozen model manifest, exact BPE tokenizer, and official prompt prefixes, all tested. Milestone **M2** is complete on `main`: strict text configuration and weight loading, the full 24-layer EmbeddingGemma 2 text encoder, per-layer embeddings, bidirectional sliding/full grouped-query attention, layer-specific RoPE, pooling, the 512 to 768 projection, Matryoshka dimensions, the CLI `embed` command, and a tiny official model fixture plus full-model parity tests. Text embeddings are validated against official goldens. Milestone **M3** is implemented on branch `feat/m3-persistent-storage`: an append-only journal with framed CRC-32 records, atomic commit and fsync, deterministic replay, torn-tail recovery, corruption detection, an exclusive writer lock, a persistent document/chunk store, versioned snapshots, crash-safe compaction, lock-free read-only views, BM25 lexical search, exact cosine vector search, RRF hybrid fusion, and the CLI `index`, `search`, `verify`, and `compact` commands. Crash-injection tests cover every journal byte, every snapshot temporary-file byte, every compaction journal temporary-file byte, and truncated snapshot files. Milestone **M4** is in progress: a native zlib/DEFLATE decoder, a strict PNG parser for non-interlaced 8-bit grayscale, RGB, and RGBA images, and a strict baseline JPEG decoder for 8-bit grayscale and three-component images are implemented on branch `feat/m4-jpeg-decoder`. Both decoders share one validated decoded-image type. PNG and JPEG enforce explicit file, chunk or segment, dimension, pixel, MCU, block, and decoded-memory caps, with committed fixtures and decoder reference comparisons. Official image preprocessing and the 16-layer vision tower are implemented on branch feat/m4-vision-encoder, and image-only embeddings are validated against official goldens. The CLI can embed an image with embed --image, index image documents, and retrieve them with a text query through exact vector search; preprocessing goldens cover rotations and non-square aspect ratios. A two-image text-to-image ranking test passes; broader ranking quality, image-to-image ranking, HNSW, the HTTP API, and audio are not implemented yet; no fake result is returned. The crate minimum is Rust 1.89 because storage uses std file locking.

## 1. Mission and scope

- Text and code: query and document embeddings.
- Images: PNG and JPEG, native preprocessing and visual encoder.
- Audio: WAV PCM first, then MP3 only if the final format matrix requires it; native audio encoding, **no transcription**.
- Comparable cross-modal embeddings: text ↔ image, text ↔ audio, media ↔ media.
- Dimensions 768, 512, 256, and 128, with **renormalization after truncation**.
- Documents, metadata, filters, persistent indexing, hybrid search.
- Local HTTP API, resource control, diagnostics, clean shutdown, recovery after interruption.
- **Out of scope:** generative LLM, transcription, GPU, external orchestration.

## 2. Technical contract — non-negotiable

| Rule | Practical consequence | Verification |
|---|---|---|
| **Zero third-party crates** | No entries in `[dependencies]`, `[dev-dependencies]`, or `[build-dependencies]`. Only `std` and `core::arch` intrinsics are allowed. | `cargo tree --edges normal,build,dev` must show only the root crate. |
| **100% Rust product code** | No C, C++, Go, Python, external assembly, FFI, or dynamic application library. Comparison/reference tools remain **laboratory-only**. | Code review and `otool -L` on the final binary. |
| **Zero external inference engine** | No llama.cpp, Candle, ONNX Runtime, PyTorch, or inference server. | No auxiliary process launched by the product. |
| **Zero external database** | No SQLite or DB server: storage, journal, vector index, and text search are written in Rust. | No storage dependency in the Cargo graph. |
| **Zero network required at runtime** | Works offline, without a key or download. | Test on a clean, offline machine. |
| **`unsafe` justified only** | Forbidden in parsers and business code. Allowed for `core::arch` SIMD kernels, with a `// SAFETY:` comment, a safe wrapper, and an equivalent tested scalar path. | Line-by-line review of every `unsafe` block. |
| **Precision** | Activations in **FP32**; BF16 weights accepted; **FP16 forbidden for activations**. | Types and parity tests. |
| **No panic on hostile input** | No `unwrap()`/`expect()` on external input; typed errors propagated with `?`. `expect()` reserved for proven invariants; `unwrap()` tolerated in tests. | `cargo clippy`, review, corruption tests. |
| **Explicit memory bounds** | Every allocation derived from external input is capped (dimensions, pixels, decompressed size, HTTP body, ingestion queue, thread pool). | Extreme-size and decompression-bomb tests. |
| **Determinism** | The FP32 scalar path is the correctness reference; any optimization must stay within tolerances. | Golden tests replayed after each optimization. |
| **No Intel macOS target** | Phase 1: `aarch64-apple-darwin` only. No compilation or validation for `x86_64-apple-darwin`. | `rustup target list`, build scripts. |

## 3. Build targets

1. **Phase 1 (single initial target):** `aarch64-apple-darwin`, macOS 11 or later depending on the capabilities actually used. Verify the binary's effective `MACOSX_DEPLOYMENT_TARGET`.
2. **Phase 2 (Linux port):** `x86_64-unknown-linux-musl`, then `aarch64-unknown-linux-musl`, statically linked.
3. **Excluded:** `x86_64-apple-darwin` (Intel Mac). No universal binary promise.

## 4. Target tree

```text
seldon-vault-lite/
├── Cargo.toml                 # zero dependencies
├── Cargo.lock
├── README.md
├── AGENTS.md
├── docs/
│   ├── adr/                   # decisions: std only, formats, persistence, unsafe, model
│   ├── architecture.md
│   ├── model-parity.md
│   ├── api.md
│   └── formats.md
├── src/
│   ├── main.rs
│   ├── api/                   # native HTTP/1.1 + JSON
│   ├── cli/                   # serve, embed, index, search, verify, compact, version
│   ├── binary/                # Safetensors, headers, checksums
│   ├── tensor/                # scalar and SIMD kernels
│   ├── model/                 # tokenizer, EmbeddingGemma 2
│   ├── media/                 # PNG, JPEG, WAV, optional MP3
│   ├── index/                 # BM25, exact KNN, optional HNSW, RRF
│   ├── storage/               # journal, snapshots, transactions
│   ├── jobs/
│   └── error.rs
├── tests/                     # Rust std tests only
├── fixtures/                  # compact golden tests
└── tools/
    └── model-packer/          # Rust std only, used at build time
```

Create modules as milestones progress, not ahead of time and empty. Every public module is documented with `///` or `//!`.

## 5. Work loop

1. Read the plan, the relevant ADR, and the module documentation before changing code.
2. Identify the current milestone (M0–M9) and the smallest verifiable step to perform.
3. Write or complete the **golden tests** and fixtures before implementation.
4. Implement the scalar reference path first, then optimize only after agreement.
5. Run the full verification below.
6. Update the ADR/documentation and record benchmark results if the step affects performance.
7. Commit with a message of the form `type(scope): summary` and mention the milestone.

Never skip the test step to "save time", and never merge an unvalidated optimization.

### Starting point — M0

- [x] Create `Cargo.toml` with no dependencies and a compilable `src/main.rs` with a minimal CLI (`version`).
- [x] Write the founding ADR: `std` only, formats, persistent data, `unsafe` policy, model version, and compatibility.
- [x] Record the invariants: no crash on hostile input, explicit memory bounds, no `unwrap()` on external input, deterministic scalar computations.
- [x] Create the `docs/`, `src/`, `tests/`, `fixtures/`, and `tools/model-packer/` tree without unnecessary empty modules.
- [x] Verify the empty Cargo graph, build for `aarch64-apple-darwin`, and check linked libraries.

## 6. Verification commands

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build --release --target aarch64-apple-darwin

# "Zero crate" contract: the output must contain only the root crate.
cargo tree --edges normal,build,dev

# Linked libraries: macOS system libraries only.
otool -L target/aarch64-apple-darwin/release/seldon-vault-lite

# Deployment version actually recorded in the Mach-O.
vtool -show-build target/aarch64-apple-darwin/release/seldon-vault-lite
```

`rustup target add aarch64-apple-darwin` is an allowed development tool. None of these tools is part of the delivered product.

## 7. Rust code rules

- **Formatting and lint:** `rustfmt` and `clippy` with `-D warnings`. No ignored warning without written justification.
- **Naming:** `snake_case` for functions/variables, `CamelCase` for types, `SCREAMING_CASE` for constants. No `get_` prefix; `as_`/`to_`/`into_` depending on conversion cost.
- **Types:** newtypes for domain identifiers and units (`DocumentId`, `Dimension`, `TokenId`) rather than bare `String`/`usize`. Make invalid states unrepresentable when reasonable.
- **Allocations:** `Vec::with_capacity` when the size is known; activation buffers reused per layer; no hidden implicit copies in hot paths; no reflex `clone()`.
- **Errors:** central error type in `src/error.rs`, manual implementation of `std::error::Error`, propagation with `?`. Distinguish user error, corrupted input, limit exceeded, and internal error.
- **Invariants:** validate shapes, indices, offsets, lengths, and overflows before any access. Use `checked_*`/`saturating_*` when arithmetic can overflow.
- **Concurrency:** `std::thread` only; bounded pool, cooperative shutdown, documented lock order, no lock held during a long operation or blocking I/O.
- **SIMD:** runtime CPU capability detection; NEON for macOS ARM64; AVX2/FMA only in the Linux x86-64 port; mandatory scalar fallback; scalar vs SIMD differential tests.
- **Documentation:** explain the *why* (format, constraint, tolerance), not just the *what*. Every `unsafe` carries a verifiable `// SAFETY:` comment.

> Generic Rust advice that assumes crates (`thiserror`, `anyhow`, `tokio`, `crossbeam`, `parking_lot`, `serde`, `clap`, `axum`, etc.) **does not apply here**: everything must be implemented with `std`. The Rust skills in `.dsh/skills/` can help with ownership, errors, concurrency, or `unsafe`, but their dependency recommendations must be ignored.

## 8. Model and numerical parity

- **Freeze** `google/embeddinggemma-2` at a precise revision and record the SHA-256 of **all** files used in a versioned manifest.
- **Inventory** layers, dimensions, numeric types, expected tensors, and weight names. Do not assume a generic "Gemma 4" is compatible: note non-standard elements in the official sources.
- **Golden tests** built outside the product with the official implementation: token IDs, intermediate layer outputs, and final vectors for French/English, code, empty/Unicode strings, varied images, and WAV excerpts.
- **Fixtures** compact, versioned, with provenance and fingerprints. **Never commit giant weights** to Git.
- **Distinct tolerances:** exact equality for tokenization and dimensions; absolute/relative tolerance for FP32 tensors; cosine and search rank for quantized embeddings.
- **Tokenization:** reproduce exactly the variant declared by the model (normalization, spaces, UTF-8, special tokens, padding, masks, context limits). Any token ID divergence **blocks** the integration of inference layers.
- **Prompts:** `SearchQuery`, `Document`, similarity, and classification prefixes per the official configurations; preserve document titles; explicitly distinguish query from indexed document.
- **Text:** 24 layers, sliding window, compliant pooling and 512 → 768 projection, L2 normalization, Matryoshka dimensions 768/512/256/128 with **renormalization after reduction**.
- **Images:** PNG (native DEFLATE/zlib) and JPEG decoders written in Rust; strict caps on dimensions, pixels, decompressed memory, and time; official preprocessing reproduced (color, resizing, patches, positions, token budget).
- **Audio:** WAV/RIFF PCM with native conversions and resampling; official preprocessing reproduced after inspecting the official processor; native MP3 only if required, with the same limits and fixtures. **No Whisper, no transcription.**
- **Cross-modality:** text, image, and audio must share the same embedding space and be comparable.

## 9. Storage, index, and API

### Storage

- Explicit model: documents, chunks, vectors, modalities, sources, ACLs/filters, metadata; stable IDs and schema versions.
- Portable format: framed records, bounded lengths, checksums, version.
- Append-only journal: atomic commit, `fsync` at critical points, deterministic replay, recovery after a brutal stop.
- Snapshots, safe compaction, writer locking, concurrent reads.
- Crash-injection tests at **every** write step; identical recovery; no duplicates and no losses.

### Index

- Inverted lexical index with BM25, normalization, and filters.
- **Exact** vector search (cosine/normalized dot product) as the correctness oracle.
- Native HNSW only after volume and latency measurements, compared with the exact oracle.
- Hybrid RRF fusion (lexical + vector), pagination, consistent deletions, and deterministic scores.

### HTTP API and CLI

- `std::net::TcpListener` HTTP/1.1 server: strict parsing, bounded connections, timeouts, validated `Content-Length`, maximum body size.
- JSON produced by the internal parser; documented HTTP codes and errors.
- Routes: `GET /health`, `GET /ready`, `GET /v1/models`, `POST /v1/embeddings`, `POST /v1/documents`, `DELETE /v1/documents/{id}`, `POST /v1/search`, `GET /v1/jobs/{id}`.
- Bounded ingestion queue, concurrency limiting, cancellations, graceful shutdown, local observability.
- Default binding to `127.0.0.1`. For LAN: explicit configuration, access token, strict limits. Without external TLS, **do not expose directly to the Internet**.
- Native CLI: `serve`, `embed`, `index`, `search`, `verify`, `compact`, `version`. No auxiliary process.

## 10. Tests and definition of "done"

Mandatory tests before release:

- **Correctness:** token ID parity, intermediate computations, and final vectors against golden tests.
- **Multimodal:** text → relevant image; text → audio excerpt; image ↔ image.
- **Dimensions:** 128/256/512/768, truncation followed by normalization; same space for queries and documents.
- **Persistence:** ingestion, modification, deletion, crash, replay, snapshot, restart, restore.
- **Robustness:** corrupted images/audio, arbitrary data, limit sizes, invalid JSON, cancellation, out of memory.
- **Security:** internal fuzzing with a home-made Rust corpus generator; review of parsers and `unsafe` SIMD regions; LAN exposure tests.
- **Performance:** p50/p95/p99, resident RAM, cold start, throughput, vector scalability, saturated CPU.
- **Distribution:** macOS Apple Silicon only in phase 1; no third-party crates; no unauthorized dynamic application library; model included; no network at runtime; test on a clean Mac without Homebrew, Python, or an inference tool.

**Done** means: a single binary per architecture, validated first on macOS Apple Silicon, usable offline; real text/image/audio embeddings compliant with references; persistent hybrid search; documented and secure API for local use; recovery after incident; measured performance; reproducible tests; readable source and respected licenses. The Linux port comes next, without an Intel macOS target.

## 11. Performance and packaging

- **Profile before optimizing**: tokenization, preprocessing, matmul, attention, loading, concurrent requests.
- Prioritize NEON via `core::arch` on macOS ARM64, keeping a scalar path. AVX2/FMA only on Linux x86-64. **No Accelerate or Metal** (FFI forbidden).
- CPU capability detection and runtime kernel selection, without illegal instructions on older processors.
- Blocking/cache locality, bounded parallelism, buffer reuse.
- Possible Q8 quantization with safe accumulations/activations and embedding drift control; this is not a given.
- At each optimization: verify golden tests and search ranking **before** comparing times. Archive benchmarks with CPU, compiler, model revision, memory, and parameters.
- During development, keep model files **outside the binary**. The final packer is written in Rust `std`, freezes weights/tokenizer/processors/manifest into a versioned and verified container, then is embedded via `include_bytes!`.
- Test the real limits of the Mach-O linker, startup, size, and RAM on Apple Silicon Mac. BF16 weights alone are on the order of 1.5 GB; activations and buffers increase consumption.
- Verify `otool -L` and Gatekeeper scenarios, without requiring a developer account for local use.
- Only **user data, logs, and snapshots** are external persistent files. A single binary does not mean zero disk writes.

## 12. Milestones

| Milestone | Observable result | Unlocks |
|---|---|---|
| **M0** | Rust `std`-only repository + ADR + reference datasets + macOS ARM64 build | Verifiable development on Apple Silicon |
| **M1** | Safetensors/JSON/tokenization + FP32 tensors | First inference |
| **M2** | Compliant text embeddings | Semantic text search |
| **M3** | Persistent index/storage + hybrid search | Useful text product |
| **M4** | PNG/JPEG images and compliant visual embeddings | Multimodal image search |
| **M5** | WAV and compliant audio embeddings | Multimodal audio search |
| **M6** | HTTP API, jobs, and hardening | Complete local service |
| **M7** | NEON SIMD and verified macOS ARM64 packaging | macOS Apple Silicon distribution |
| **M8** | End-to-end tests on a clean Apple Silicon Mac and final documentation | macOS 1.0 release |
| **M9** | Linux x86-64/ARM64 MUSL port, tests, and documentation | Linux release |

Indicative estimate: several months at minimum for one person; 9 to 18 months or more are plausible. The riskiest phases: image/audio encoder parity, quantization, media parsers, and fault tolerance.

## 13. Risks to keep visible

1. **Zero crate** greatly increases maintenance load and bug surface (HTTP, JSON, PNG/JPEG, audio, index, persistence): a serious security review is imperative.
2. **Specialized model**: a major change in weights or processor requires an engine update. Freezing the model is an accepted feature.
3. **FP16 forbidden** for activations: start in FP32, then study BF16 and quantization with validation.
4. **Single binary ≠ zero disk**: indexed documents and the WAL must survive restarts.
5. **No application dependency ≠ OS/CPU independence**: network, files, threads, and memory go through the platform.
6. **Performance to prove**: Rust is not automatically faster than llama.cpp; the metric is performance at comparable quality.

## 14. Pre-commit checklist

- [ ] No dependency added (check `cargo tree`).
- [ ] `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test`, and the `aarch64-apple-darwin` build pass.
- [ ] Golden tests and fixtures up to date; no parity regression.
- [ ] No `unsafe` without a `// SAFETY:` comment and without a scalar path test.
- [ ] No heavy weights, secret, or model file committed.
- [ ] Documentation/ADR updated if behavior or a decision changes.
- [ ] Commit message in `type(scope): summary` format, with the relevant milestone.

## 15. References

- [Google — official EmbeddingGemma 2 model card](https://github.com/google-gemma/docs/blob/main/docs/models/embeddinggemma/model-card-v2.md)
- [Google — original EmbeddingGemma 2 weights](https://huggingface.co/google/embeddinggemma-2)
- [Dalailalama — EmbeddingGemma 2 in Rust](https://github.com/Dalailalama/embeddinggemma2-rust) (to study, never as a dependency)
- [Rust — official `std` documentation](https://doc.rust-lang.org/std/)
- [Rust — official macOS targets](https://doc.rust-lang.org/rustc/platform-support/apple-darwin.html)
