# Architecture — milestone M2 state

## Purpose

This document describes the repository organization at milestone M2 and the
boundaries that later milestones must respect. It does not replace ADR 0001,
ADR 0002, ADR 0003, or ADR 0004, which are authoritative for foundational,
format, model, and text-embedding decisions.

## Current state

The repository contains:

```text
Cargo.toml              # zero dependencies
Cargo.lock              # reproducible locking
rust-toolchain.toml     # stable, clippy, rustfmt, ARM64
.cargo/config.toml      # aarch64-apple-darwin target, macOS 11.0
src/
├── main.rs             # translates the CLI result into an exit code
├── lib.rs              # version, target, frozen revision constants
├── error.rs            # central error type
├── cli/mod.rs          # version, help, and embed
├── binary/
│   ├── mod.rs
│   ├── reader.rs       # bounded byte reader and file cap
│   ├── sha256.rs       # streaming SHA-256
│   ├── bf16.rs         # BF16 <-> FP32 conversion
│   └── safetensors.rs  # strict read-only Safetensors reader
├── json/mod.rs         # strict bounded JSON parser
├── model/
│   ├── mod.rs
│   ├── config.rs       # strict text configuration and tensor inventory
│   ├── weights.rs      # validated FP32 text weight loader
│   ├── text.rs         # EmbeddingGemma 2 text forward pass
│   ├── prompt.rs       # official prompt prefixes
│   └── tokenizer.rs    # exact BPE tokenizer for the frozen model
└── tensor/
    ├── mod.rs
    ├── dtype.rs        # F32/BF16 dtype policy
    ├── shape.rs        # validated shapes and strides
    ├── owned.rs        # owned FP32 tensors and views
    └── kernels.rs      # scalar reference kernels and model operations
tests/
├── cli.rs              # binary integration tests, including embed
├── formats.rs          # public API format/tensor integration tests
├── text_embeddings.rs  # tiny and full-model text parity tests
└── tokenizer.rs        # golden tokenizer and manifest parity tests
docs/
├── adr/0001-foundations.md
├── adr/0002-formats-and-tensors.md
├── adr/0003-model-freeze-and-tokenizer.md
├── adr/0004-text-embeddings.md
├── architecture.md
├── formats.md
├── model-parity.md
└── tokenizer.md
fixtures/
├── README.md
├── formats/
│   └── mini.safetensors # deterministic format fixture
├── model/
│   ├── manifest.json    # frozen revision, sizes, SHA-256, prompts
│   ├── text_golden.json # eight text cases and a layer probe
│   └── tiny_text/       # complete 3-layer official model and golden outputs
└── tokenizer/
    ├── golden.json      # official token ids for the golden corpus
    └── mini_tokenizer.json # reduced tokenizer for reproducible tests
tools/
└── model-packer/
    └── README.md       # role planned at the packaging milestone
scripts/
└── verify.sh           # full milestone quality gate
```

No `api`, `media`, `index`, `storage`, or `jobs` module is created
empty. They will appear with their first testable feature, at the corresponding
milestone. Image and audio inference are not implemented yet.

## Layers and dependencies

The `seldon_vault_lite` library is the testable root; the `main.rs` binary
only calls the CLI and converts errors into exit codes. Dependencies point
downward:

```text
cli / api  ->  model / index / storage  ->  binary / tensor / media  ->  error
```

Rules:

- `error` depends on no other internal module.
- `json` depends only on `error`.
- `binary` depends on `error`, `json`, and `tensor`.
- `tensor` depends only on `error`.
- `model` depends on `binary`, `json`, `tensor`, and `error`; it does
  not depend on storage or the HTTP API.
- Parsers do not depend on inference layers.
- Storage does not depend on the HTTP API.
- No module depends on a third-party crate.

## CLI contract

| Command | Effect | Output | Code |
|---|---|---|---|
| `version` | prints name, version, target | stdout | 0 |
| `help` | prints help, commands, and codes | stdout | 0 |
| `embed` | loads the text model and prints one JSON vector | stdout | 0 |
| `--version`, `-V` | alias of version | stdout | 0 |
| `--help`, `-h` | alias of help | stdout | 0 |
| no command | usage error | stderr | 2 |
| unknown command | usage error | stderr | 2 |
| extra argument | usage error | stderr | 2 |
| non-UTF-8 argument | usage error | stderr | 2 |
| invalid embed option | usage error | stderr | 2 |
| dimension above the model output | usage error | stderr | 2 |
| out-of-vocabulary token id | corruption error | stderr | 1 |
| model or weight loading failure | execution error | stderr | 1 |
| write failure | execution error | stderr (best-effort) | 1 |
| closed pipe | normal end | — | 0 |

`embed` accepts `--model-dir`, `--prompt`, `--dim`, `--text`, `--ids`,
and positional text. With no text or ids, it reads standard input with a 1 MiB
cap. Text is prefixed with the official prompt before tokenization; `--ids`
bypasses the tokenizer and the prefix for diagnostics.

`serve`, `index`, `search`, and `verify` remain explicitly refused until
implemented. No stub returns a fake result.

## Error model

`src/error.rs` defines `Error` and `Result<T>`. The variants are:

- `Usage` (exit code 2);
- `Io` (exit code 1);
- `Corrupt` (exit code 1);
- `LimitExceeded` (exit code 1);
- `Unsupported` (exit code 1);
- `Internal` (exit code 1).

`Display` produces a user-facing message; `source()` exposes the technical
cause for I/O errors. The HTTP API will reuse the same taxonomy to choose
statuses.

## Format boundaries

### Binary reader

`ByteReader` returns borrowed slices, uses checked arithmetic for every range,
and allocates only through `read_vec` with an explicit cap.
`read_file_bounded` checks metadata and reads through `Read::take`.

### JSON

`src/json` is a strict RFC 8259 subset with configurable caps on input size,
depth, string size, container entries, and total values. It rejects duplicate
keys, trailing commas, non-finite numbers, lone surrogates, and trailing
content. Objects are ordered with a `BTreeMap`.

### Safetensors

`SafetensorsFile` borrows the file bytes, validates the header and every
tensor range, rejects overlaps, and exposes FP32 decoding and a SHA-256
fingerprint. FP16 is rejected because activations must stay FP32.

### Tensors and kernels

`Shape` validates rank, dimensions, and element count at construction.
`Tensor` is owned FP32 data capped at `2^30` elements. `TensorView` borrows
without copying. `src/tensor/kernels.rs` is the deterministic scalar
reference. M2 adds:

- linear projection against an [out, in] weight matrix;
- per-row RMSNorm with and without a learned scale;
- mean pooling;
- truncation followed by renormalization;
- RoPE with precomputed inverse frequencies;
- bidirectional grouped-query attention with an optional sliding window.

SIMD will be compared against this path before any optimization is accepted.

## Model boundary

The model is frozen at revision
`f9972ac22e696a105c482202646979e33e4b5a1e`. The manifest is
`fixtures/model/manifest.json`.

### Configuration

`model::config::TextConfig` parses the composite `config.json` (using its
`text_config` member) or a standalone text configuration. It rejects an
unknown model type, an unsupported dtype, FP16, a non-GELU activation,
attention bias, non-zero attention dropout, an unknown layer type, an unknown
RoPE type, inconsistent layer counts, a non-full last layer, and mismatched
head layouts. It exposes the expected tensor inventory so the loader can reject
a wrong weight file before decoding.

Limits include 16,384 hidden size, 128 layers, 128 heads, 1,024 head
dimension, 65,536 intermediate size, 1,000,000 vocabulary entries, 16,384
embedding dimension, and 1,048,576 positions.

### Weight loading

`model::weights::TextWeights` reads the frozen Safetensors file with a 4 GiB
cap, ignores audio and vision tensors, accepts only the 413
`language_model.` text tensors, validates every name and shape, rejects extra
or missing text tensors and non-finite values, and converts BF16 or F32
storage to FP32 once. The source SHA-256 is retained for manifest comparison.
The FP32 weight set is capped at `2^30` elements.

### Tokenizer

`model::tokenizer::Tokenizer` loads the frozen tokenizer.json with bounded
JSON limits and validates the BPE pipeline before use:

- model type BPE, byte fallback, fuse_unk, no ignore_merges;
- normalizer replaces a space with `▁`;
- pre-tokenizer splits on a space with `MergedWithPrevious`;
- post-processor prepends `<bos>` and appends `<eos>`;
- decoder applies Replace, ByteFallback, and Fuse.

Encoding matches added tokens in raw text, normalizes spaces, starts from
single characters or byte tokens, and applies merges by rank. The input is
capped at 1 MiB and the output at 262,144 tokens. The full tokenizer.json is a
development asset outside Git; the reduced fixture reproduces it on the golden
corpus.

### Prompts

`model::prompt` exposes the official prefixes and a `Prompt` enum.
`Prompt::from_name` parses the official names. A query and a document are
distinct roles; a test compares every constant with the frozen manifest.

### Text inference

`model::text::TextModel` owns the configuration and FP32 weights.
`TextEmbedder` adds the tokenizer. The forward pass follows ADR 0004:

1. scaled token embedding;
2. projection-only PLE, scaled, reshaped to [sequence, layers, 512], and
   normalized;
3. per-layer bidirectional attention with per-head Q/K/V RMSNorm,
   layer-specific RoPE, grouped-query attention, output projection, and
   residual;
4. gated GELU feed-forward with residuals;
5. PLE residual block and layer scalar;
6. final RMSNorm, 512 to 768 projection, mean pooling, and L2 normalization;
7. Matryoshka truncation with renormalization.

The model rejects empty sequences, ids outside the vocabulary, sequences above
the 8,192-token context, and dimensions above the model output. Activation
buffers are capped at `2^28` elements. The model is immutable and
`Send + Sync`; repeated calls are deterministic.

### Model files

Model files live in `.model-cache/<revision>/`, which is ignored by Git.
Weights are never committed. The manifest records the weights' SHA-256 from the
Hugging Face LFS object id. The tiny official model fixture is committed
because it is 47.5 KiB, not model-scale data.

## Trust boundaries

Everything coming from outside is untrusted: CLI arguments, files, HTTP bodies,
JSON, Safetensors headers, tokenizer documents, images, audio, and metadata.
Each boundary must:

1. validate types, lengths, offsets, and versions before allocation;
2. cap dimensions and memory;
3. use checked arithmetic;
4. return a typed error, never a panic.

The FP32 scalar path remains the correctness reference for all numerical
computations.

## Build configuration

- Target: `aarch64-apple-darwin` (enforced by `.cargo/config.toml`).
- Deployment: `MACOSX_DEPLOYMENT_TARGET=11.0` (forced).
- `rust-toolchain.toml`: stable toolchain, `clippy`, `rustfmt`.
- `Cargo.toml`: no dependencies; release profile `opt-level = 3`,
  `lto = "thin"`, `codegen-units = 1`; test profile `opt-level = 2` so
  the full-model parity suite is practical.

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

The full-model parity test is:

```bash
cargo test --release --test text_embeddings real_model_parity -- --nocapture
```

It runs automatically when the model cache is present. Set
`SELDON_SKIP_MODEL_PARITY=1` to skip it during a quick check.

`scripts/verify.sh` groups these commands and fails if the Cargo graph
contains a dependency or if `minos` is not 11.0.

## Next steps

1. **M2**: complete for text embeddings, layer-output parity, pooling, the
   512 to 768 projection, and Matryoshka dimensions. Image and audio are not
   claimed.
2. **M3**: persistent storage, BM25/KNN index, and RRF fusion.
3. **M4/M5**: PNG/JPEG/WAV decoders and multimodal embeddings.
4. **M6**: HTTP API, bounded job queue, and hardening.
5. **M7**: NEON and macOS packaging.
6. **M8/M9**: final macOS validation, then Linux port.

None of these steps may be declared complete without reproducible tests and
without replaying the golden tests.
