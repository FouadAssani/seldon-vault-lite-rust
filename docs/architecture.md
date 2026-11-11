# Architecture — milestone M0 state

## Purpose

This document describes the repository organization at milestone M0 and the boundaries that later milestones must respect. It does not replace ADR 0001, which is authoritative for foundational decisions.

## Current state

The repository contains:

```text
Cargo.toml              # zero dependencies
Cargo.lock              # reproducible locking
rust-toolchain.toml     # stable (validated 1.99.1), clippy, rustfmt, ARM64
.cargo/config.toml      # aarch64-apple-darwin target, macOS 11.0
src/
├── main.rs             # translates the CLI result into an exit code
├── lib.rs              # version/target constants and module root
├── error.rs            # central error type
└── cli/mod.rs          # parsing and execution of `version`/`help`
tests/
└── cli.rs              # binary integration tests
docs/
├── adr/0001-foundations.md
├── architecture.md
└── model-parity.md
fixtures/
└── README.md           # golden test policy (no real fixture)
tools/
└── model-packer/
    └── README.md       # role planned at milestone M12
scripts/
└── verify.sh           # full milestone quality gate
```

No `api`, `binary`, `tensor`, `model`, `media`, `index`, `storage`, or `jobs` module is created empty. They will appear with their first testable feature, at the corresponding milestone.

## Layers and dependencies

The `seldon_vault_lite` library is the testable root; the `main.rs` binary only calls the CLI and converts errors into exit codes. Future modules will follow downward dependencies:

```text
cli / api  →  model / index / storage  →  tensor / binary / media  →  error
```

Rules:

- `error` depends on no other internal module.
- Parsers do not depend on inference layers.
- Storage does not depend on the HTTP API.
- No module depends on a third-party crate.

## CLI contract

| Command | Effect | Output | Code |
|---|---|---|---|
| `version` | Prints `seldon-vault-lite <version> (<target>)` | stdout | 0 |
| `help` | Prints help, commands, and codes | stdout | 0 |
| `--version`, `-V` | Alias of `version` | stdout | 0 |
| `--help`, `-h` | Alias of `help` | stdout | 0 |
| no command | Usage error | stderr | 2 |
| unknown command | Usage error | stderr | 2 |
| extra argument | Usage error | stderr | 2 |
| non-UTF-8 argument | Usage error | stderr | 2 |
| write failure | Execution error | stderr (best-effort) | 1 |
| closed pipe | Normal end | — | 0 |

The `serve`, `embed`, `index`, `search`, and `verify` commands are planned but **not yet accepted**. Explicitly refusing them is better than returning a fake result.

## Error model

`src/error.rs` defines `Error` and `Result<T>`. The current variants are `Usage(String)` and `Io(io::Error)`. The enum is `#[non_exhaustive]` to allow adding `Corrupt`, `LimitExceeded`, and `Internal` variants without breaking callers.

Rules:

- `?` propagates errors; no `unwrap()` on external input.
- `Display` produces a user-facing message; `source()` exposes the technical cause.
- `exit_code()` centralizes translation into an exit code.
- The HTTP API will reuse the same taxonomy to choose statuses.

## Trust boundaries

Everything coming from outside is untrusted: CLI arguments, files, HTTP bodies, JSON, Safetensors headers, images, audio, and metadata. Each boundary must:

1. validate types, lengths, offsets, and versions before allocation;
2. cap dimensions and memory;
3. use checked arithmetic;
4. return a typed error, never a panic.

The FP32 scalar path remains the correctness reference for all numerical computations. SIMD optimizations will be accepted only after agreement with this path and replay of golden tests.

## Build configuration

- Target: `aarch64-apple-darwin` (enforced by `.cargo/config.toml`).
- Deployment: `MACOSX_DEPLOYMENT_TARGET=11.0` (forced).
- `rust-toolchain.toml`: stable toolchain (validated with Rust 1.99.1), `clippy`, `rustfmt`.
- `Cargo.toml`: no dependencies; release profile `opt-level = 3`, `lto = "thin"`, `codegen-units = 1`.

## Verification

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build --release --target aarch64-apple-darwin
cargo tree --edges normal,build,dev
otool -L target/aarch64-apple-darwin/release/seldon-vault-lite
vtool -show-build target/aarch64-apple-darwin/release/seldon-vault-lite
```

The `scripts/verify.sh` script groups these commands and fails if the Cargo graph contains a dependency or if `minos` is not 11.0.

The `cargo tree` output must be reduced to `seldon-vault-lite v0.1.0`. `otool -L` must list only `/usr/lib/libSystem.B.dylib` (and possibly system libraries expected by the Rust runtime). `vtool` must confirm `minos 11.0`.

## Next steps

1. **M1**: bounded byte reader, strict JSON, Safetensors, BF16 ↔ FP32, tensors, and FP32 scalar kernels.
2. **M2**: exact tokenization and compliant text embeddings.
3. **M3**: persistent storage, BM25/KNN index, and RRF fusion.
4. **M4/M5**: PNG/JPEG/WAV decoders and multimodal embeddings.
5. **M6**: HTTP API, jobs, and hardening.
6. **M7**: NEON and macOS packaging.
7. **M8/M9**: final macOS validation, then Linux port.

None of these steps may be declared complete without reproducible tests and without replaying the golden tests.
