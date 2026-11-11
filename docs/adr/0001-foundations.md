# ADR 0001 — Foundations of the Seldon Vault Lite engine

- **Status**: accepted
- **Date**: 2026-11-11
- **Milestone**: M0
- **Decision maker**: project maintainer
- **Supersedes**: none
- **Superseded by**: none

## Context

Seldon Vault Lite must become a local multimodal embedding, indexing, and search engine, distributed as a standalone executable per platform. The project imposes an unusual contract: **zero third-party crates**, **100% Rust product code**, no external inference engine, no external database, and no network required at runtime. The first target is macOS Apple Silicon (`aarch64-apple-darwin`); Intel macOS is excluded.

Milestone M0 must make development verifiable: a Rust repository that compiles for the initial target, a minimal CLI, a central error type, standard tests, a founding ADR, and documented reference criteria.

## Decisions

### D1 — Rust `std` only, no third-party crates

The `Cargo.toml` declares **no** `[dependencies]`, `[dev-dependencies]`, or `[build-dependencies]` section. Only `std` and `core::arch` intrinsics are allowed. Comparison or reference implementation tools remain **laboratory-only** and are not part of the product.

Consequences: HTTP, JSON, PNG, JPEG, WAV, Safetensors, index, storage, and parsers must be written in Rust. Any proposal adding a crate is rejected, including for tests.

Verification: `cargo tree --edges normal,build,dev` must show only the root crate.

### D2 — macOS Apple Silicon target and macOS 11 floor

Phase 1 only compiles and validates `aarch64-apple-darwin`. The `x86_64-apple-darwin` target is excluded: no compilation, no universal binary, no promise of Intel support. Linux targets (`x86_64-unknown-linux-musl` then `aarch64-unknown-linux-musl`) are deferred to milestone M9.

The deployment floor is **macOS 11.0**, a value enforced by `.cargo/config.toml` via `MACOSX_DEPLOYMENT_TARGET` with `force = true`. The `MACOSX_DEPLOYMENT_TARGET` actually recorded in the Mach-O is verified with `vtool -show-build`; a value other than 11.0 is a validation failure.

The `rust-toolchain.toml` file selects the stable toolchain with `clippy`, `rustfmt`, and the `aarch64-apple-darwin` target. The version validated for M0 is **Rust 1.99.1**; it is recorded here and in the verification results to make measurements reproducible.

### D3 — Formats and trust boundaries

Formats are treated as **hostile inputs** until fully validated:

- **JSON**: strict parser written in Rust, compliant for objects, arrays, numbers, strings, escapes, and Unicode. Depth, string length, and input size capped. No naive string splitting.
- **Safetensors**: read-only, bounded JSON header, verified offsets and lengths, rejected overlaps, no metadata execution.
- **Internal weight container**: magic, version, framed sections, bounded lengths, checksums, and manifest. Format defined at milestone M1 and versioned; it does not replace the original weights until parity is demonstrated.
- **Media**: PNG (native DEFLATE/zlib), JPEG, and WAV/RIFF PCM decoded in Rust, with caps on dimensions, pixels, decompressed memory, and duration. No external process, no dynamic application library.
- **Text**: strict UTF-8. No silent lossy conversion on model inputs.

Each parser must reject truncations, extreme sizes, invalid offsets, and arithmetic overflows before any access.

### D4 — Persistent data outside the binary

The binary contains the engine and, eventually, the model weights. **User data, logs, and snapshots** are the only external persistent files. They live in an explicit working directory.

Storage will be an append-only journal with framed records, bounded lengths, checksums, and a schema version. Commits will be atomic and `fsync` will be used at critical points. Replay must be deterministic and recovery after a brutal stop must produce the same state, without duplicates or losses. Snapshots and safe compaction will be added at milestone M3; no external database is allowed.

### D5 — `unsafe` policy

`unsafe` is **forbidden by default**, particularly in parsers, storage, business code, and the CLI.

It is allowed only for `core::arch` SIMD kernels (NEON on macOS ARM64, then AVX2/FMA on Linux x86-64), with:

1. a `// SAFETY:` comment justifying each invariant;
2. a safe wrapper preventing any invalid call;
3. an equivalent scalar path retained;
4. scalar vs SIMD differential tests.

No FFI, no `asm!`, no external assembler, no dynamic application library. Every `unsafe` block is reviewed line by line.

### D6 — Frozen model and numerical parity

The target model is **`google/embeddinggemma-2`**, frozen at a precise revision. A versioned manifest records the SHA-256 of **all** files used (configuration, tokenizer, processors, weights, prompts). Giant weights are never committed to Git.

Numerical constraints:

- BF16 weights accepted, activations in **FP32**;
- **FP16 forbidden for activations**;
- Matryoshka dimensions 768, 512, 256, and 128, with **L2 renormalization after truncation**;
- `SearchQuery` and `Document` prompts per the official configurations;
- no assumption of compatibility with a generic "Gemma": non-standard elements are noted in the official sources.

Milestone M0 loads no weights and claims no modality. The first real golden tests are produced at milestone M1/M2 with the official implementation, outside the product.

### D7 — Robustness and determinism invariants

These invariants apply to all product code:

- **No panic on hostile input**: no `unwrap()` or `expect()` on external input. Errors are typed and propagated with `?`. `expect()` is reserved for proven invariants; `unwrap()` is tolerated in tests.
- **Explicit memory bounds**: every allocation derived from external input is capped (dimensions, pixels, decompressed size, HTTP body, ingestion queue, thread pool).
- **Checked arithmetic**: use of `checked_*`/`saturating_*` when overflows are possible; validation of shapes, indices, offsets, and lengths before any access.
- **Determinism**: the FP32 scalar path is the correctness reference. Any optimization must stay within tolerances and golden tests are replayed after each optimization.
- **No implicit lossy conversion**: model inputs are validated before conversion.

The error taxonomy will progressively distinguish: usage error, corrupted input, limit exceeded, and internal error.

### D8 — CLI and exit codes

The M0 CLI exposes `version` and `help`, with the aliases `--version`/`-V` and `--help`/`-h`. The `serve`, `embed`, `index`, `search`, and `verify` commands will be added in later milestones; no stub simulates nonexistent behavior.

Observable contract:

- data goes to `stdout`, diagnostics to `stderr`;
- exit code `0`: success;
- exit code `1`: execution error (I/O);
- exit code `2`: usage error (unknown command, extra argument, non-UTF-8 argument);
- a closed pipe (`EPIPE`) is treated as success, in accordance with Unix tool conventions;
- no panic on a non-UTF-8 argument: the error is typed.

### D9 — Reproducible verification

The following commands constitute the repository's quality gate:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build --release --target aarch64-apple-darwin
cargo tree --edges normal,build,dev
otool -L target/aarch64-apple-darwin/release/seldon-vault-lite
vtool -show-build target/aarch64-apple-darwin/release/seldon-vault-lite
```

The Cargo graph must contain only the root crate. `otool -L` must show only macOS system libraries. `vtool -show-build` must show a deployment target of 11.0. None of these checks may be skipped before a commit. The `scripts/verify.sh` script runs this quality gate and automatically checks the Cargo graph and `minos 11.0`.

## Consequences

### Positive

- The "zero crate" contract is verifiable from the first commit.
- The target and macOS floor are explicit and controllable.
- The CLI and errors are testable without launching a process.
- Robustness invariants are established before the parsers arrive.
- Parity and fixture criteria are documented before inference.

### Costs and risks

- Everything must be reimplemented: HTTP, JSON, media, storage, index. The bug and security surface increases sharply; a dedicated review is necessary.
- The model is frozen: a major change in weights or processor will require an engine update.
- Performance is not a given: it will have to be measured and compared at comparable quality, not merely assumed.
- A single binary does not imply the absence of disk writes: the WAL, snapshots, and user data persist.

### Follow-up

- **ADR 0002 (M1)**: internal weight and versioned container format.
- **ADR 0003 (M1/M3)**: journal format, snapshots, and transactions.
- **ADR 0004 (M6)**: HTTP contract, LAN authentication, and limits.
- The source code license will be decided before any distribution; weights remain subject to the model's official license.
