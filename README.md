# Seldon Vault Lite

Local multimodal embedding, indexing, and search engine, written in Rust `std` only (no third-party crates), specialized for `google/embeddinggemma-2`.

**Status: milestone M0 — foundations.** The CLI exposes `version` and `help`. The `serve`, `embed`, `index`, `search`, and `verify` commands will be added in later milestones. No modality (text, image, audio) is declared supported yet: no fake result is returned.

## Requirements

- macOS 11 or later on Apple Silicon (`aarch64-apple-darwin`);
- stable Rust toolchain (validated with 1.99.1), selected by `rust-toolchain.toml` with `clippy`, `rustfmt`, and the `aarch64-apple-darwin` target;
- no other tool is required to build.

The `x86_64-apple-darwin` target (Intel Mac) is explicitly excluded.

## Build and test

```bash
cargo build
cargo run -- version
cargo run -- --help
```

## Full verification

The `scripts/verify.sh` script runs the entire quality gate. The equivalent commands are:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build --release --target aarch64-apple-darwin

# "Zero crate" contract: only the root crate must appear.
cargo tree --edges normal,build,dev

# Linked libraries and Mach-O deployment target.
otool -L target/aarch64-apple-darwin/release/seldon-vault-lite
vtool -show-build target/aarch64-apple-darwin/release/seldon-vault-lite
```

Or simply:

```bash
./scripts/verify.sh
```

## Technical contract

- **Zero third-party crates**: `Cargo.toml` contains no dependency section; only `std` and `core::arch` are allowed.
- **100% Rust product code**: no C, C++, FFI, `asm!`, or dynamic application library.
- **Initial target**: `aarch64-apple-darwin`, macOS 11.0 minimum.
- **Persistent data**: user data, logs, and snapshots will live outside the binary; weights are not committed.
- **Robustness**: no panic on hostile input, explicit memory bounds, checked arithmetic, deterministic FP32 scalar path.
- **CLI**: data on `stdout`, errors on `stderr`; exit codes `0` (success), `1` (execution), `2` (usage).

## Documentation

- [Implementation plan](seldon-vault-lite-plan.md)
- [Agent and contributor guidelines](AGENTS.md)
- [ADR 0001 — Foundations](docs/adr/0001-foundations.md)
- [Architecture (M0 state)](docs/architecture.md)
- [Numerical parity and reference datasets](docs/model-parity.md)
- [Fixtures policy](fixtures/README.md)

All project documentation and code are written in English.

## Rust skills (DSH)

The skills from the [actionbook/rust-skills](https://github.com/actionbook/rust-skills) repository are installed in `.dsh/skills/` (version 2.0.9, commit `5c40d3a`).

Update:

```bash
git clone --depth 1 https://github.com/actionbook/rust-skills.git /tmp/rust-skills
cp -R /tmp/rust-skills/skills/. .dsh/skills/
```
