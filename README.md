# Seldon Vault Lite

Local multimodal embedding, indexing, and search engine, written in Rust
`std` only (no third-party crates), specialized for
`google/embeddinggemma-2`.

**Status: milestone M2 — compliant text embeddings.**
The M1 formats, FP32 tensor core, frozen model manifest, exact BPE tokenizer,
and official prompt prefixes are complete. M2 adds the EmbeddingGemma 2 text
encoder: strict configuration and weight loading, projection-only per-layer
embeddings, bidirectional sliding/full attention, grouped-query attention,
per-head query/key/value RMSNorm, layer-specific RoPE, gated GELU blocks, the
512 to 768 projection, mean pooling, L2 normalization, and Matryoshka
dimensions 768/512/256/128 with renormalization. The CLI exposes `version`,
`help`, and `embed`. Image and audio are not implemented and are not declared
supported.

## Requirements

- macOS 11 or later on Apple Silicon (`aarch64-apple-darwin`);
- stable Rust toolchain (validated with 1.99.1), selected by
  `rust-toolchain.toml` with `clippy`,
  `rustfmt`, and the `aarch64-apple-darwin` target;
- no other tool is required to build.

The `x86_64-apple-darwin` target (Intel Mac) is explicitly excluded.

## Build and test

```bash
cargo build
cargo run -- version
cargo run -- --help
cargo test

# Text embedding with the frozen model cache.
cargo run --release -- embed --prompt SearchQuery "What causes the northern lights?"

# Token ids without the tokenizer, useful for diagnostics.
cargo run -- embed --model-dir fixtures/model/tiny_text --ids 1,5,7,9,2 --dim 8
```

The full-model parity test runs when `.model-cache/<revision>/model.safetensors`
is present. Set `SELDON_SKIP_MODEL_PARITY=1` to skip it during a quick check;
the tiny official fixture still runs.

## What milestone M2 provides

| Area | Status |
|---|---|
| Bounded byte reader, checked offsets, file-size cap | implemented |
| Streaming SHA-256 and file fingerprints | implemented |
| BF16 to FP32 decoding and FP32 to BF16 encoding | implemented |
| Strict JSON parser with caps and Unicode escapes | implemented |
| Read-only Safetensors parser with offset/overlap validation | implemented |
| FP32 tensors, shapes, views, and element caps | implemented |
| Scalar add/mul/matmul/dot/softmax/RMSNorm/GELU/GeGLU/RoPE/attention/L2 | implemented |
| Deterministic Safetensors fixture | implemented |
| Frozen model manifest with SHA-256 for every file | implemented |
| Exact BPE tokenizer and byte fallback | implemented |
| Golden token IDs and reduced tokenizer fixture | implemented |
| Official prompt prefixes with query/document roles | implemented |
| Strict text configuration and text weight inventory | implemented |
| Projection-only per-layer embeddings (PLE) | implemented |
| 24 bidirectional sliding/full layers and grouped-query attention | implemented |
| Per-head Q/K/V RMSNorm and layer-specific RoPE | implemented |
| 512 to 768 projection, mean pooling, L2 normalization | implemented |
| Matryoshka 768/512/256/128 with renormalization | implemented |
| Tiny official model fixture and full-model text parity test | implemented |
| CLI `embed` for text and token ids | implemented |
| Image/audio embeddings | pending |

The format and tensor layers are documented in
[docs/formats.md](docs/formats.md) and decided in
[docs/adr/0002-formats-and-tensors.md](docs/adr/0002-formats-and-tensors.md).
The frozen model and tokenizer are documented in
[docs/tokenizer.md](docs/tokenizer.md) and decided in
[docs/adr/0003-model-freeze-and-tokenizer.md](docs/adr/0003-model-freeze-and-tokenizer.md).

Model files are development assets outside Git, in `.model-cache/<revision>/`.
The committed reduced tokenizer reproduces the full tokenizer on the golden
corpus. When the full `tokenizer.json` is present, `cargo test` also verifies
its SHA-256 and every golden case against it.

## Full verification

The `scripts/verify.sh` script runs the entire quality gate. The
equivalent commands are:

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

- **Zero third-party crates**: `Cargo.toml` contains no dependency
  section; only `std` and `core::arch` are allowed.
- **100% Rust product code**: no C, C++, FFI, `asm!`, or dynamic
  application library.
- **Initial target**: `aarch64-apple-darwin`, macOS 11.0 minimum.
- **Persistent data**: user data, logs, and snapshots will live outside the
  binary; weights are not committed.
- **Robustness**: no panic on hostile input, explicit memory bounds, checked
  arithmetic, deterministic FP32 scalar path.
- **CLI**: data on `stdout`, errors on `stderr`; exit codes
  `0` (success), `1` (execution), `2` (usage).

## Documentation

- [Implementation plan](seldon-vault-lite-plan.md)
- [Agent and contributor guidelines](AGENTS.md)
- [ADR 0001 — Foundations](docs/adr/0001-foundations.md)
- [ADR 0002 — Formats and tensors](docs/adr/0002-formats-and-tensors.md)
- [ADR 0003 — Model freeze and tokenizer](docs/adr/0003-model-freeze-and-tokenizer.md)
- [ADR 0004 — Text embeddings](docs/adr/0004-text-embeddings.md)
- [Architecture (M2 state)](docs/architecture.md)
- [Formats and binary primitives](docs/formats.md)
- [Tokenizer and prompts](docs/tokenizer.md)
- [Numerical parity and reference datasets](docs/model-parity.md)
- [Text embedding benchmark baseline](docs/benchmarks.md)
- [Frozen model manifest](fixtures/model/manifest.json)
- [Fixtures policy](fixtures/README.md)

All project documentation and code are written in English.

## Rust skills (DSH)

The skills from the [actionbook/rust-skills](https://github.com/actionbook/rust-skills)
repository are installed in `.dsh/skills/` (version 2.0.9, commit
`5c40d3a`).

Update:

```bash
git clone --depth 1 https://github.com/actionbook/rust-skills.git /tmp/rust-skills
cp -R /tmp/rust-skills/skills/. .dsh/skills/
```
