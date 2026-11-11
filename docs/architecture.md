# Architecture — milestone M1 state

## Purpose

This document describes the repository organization at milestone M1 and the
boundaries that later milestones must respect. It does not replace ADR 0001,
ADR 0002, or ADR 0003, which are authoritative for foundational, format, and
model decisions.

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
├── cli/mod.rs          # parsing and execution of version/help
├── binary/
│   ├── mod.rs
│   ├── reader.rs       # bounded byte reader and file cap
│   ├── sha256.rs       # streaming SHA-256
│   ├── bf16.rs         # BF16 <-> FP32 conversion
│   └── safetensors.rs  # strict read-only Safetensors reader
├── json/mod.rs         # strict bounded JSON parser
├── model/
│   ├── mod.rs
│   ├── prompt.rs       # official prompt prefixes
│   └── tokenizer.rs    # exact BPE tokenizer for the frozen model
└── tensor/
    ├── mod.rs
    ├── dtype.rs        # F32/BF16 dtype policy
    ├── shape.rs        # validated shapes and strides
    ├── owned.rs        # owned FP32 tensors and views
    └── kernels.rs      # scalar reference kernels
tests/
├── cli.rs              # binary integration tests
├── formats.rs          # public API format/tensor integration tests
└── tokenizer.rs        # golden tokenizer and manifest parity tests
docs/
├── adr/0001-foundations.md
├── adr/0002-formats-and-tensors.md
├── adr/0003-model-freeze-and-tokenizer.md
├── architecture.md
├── formats.md
├── model-parity.md
└── tokenizer.md
fixtures/
├── README.md
├── formats/
│   └── mini.safetensors # deterministic format fixture
├── model/
│   └── manifest.json    # frozen model revision, sizes, SHA-256, prompts
└── tokenizer/
    ├── golden.json      # official token ids for the golden corpus
    └── mini_tokenizer.json # reduced tokenizer for reproducible tests
tools/
└── model-packer/
    └── README.md       # role planned at milestone M12
scripts/
└── verify.sh           # full milestone quality gate
```

No `api`, `media`, `index`, `storage`,
or `jobs` module is created empty. They will appear with their first
testable feature, at the corresponding milestone. Inference is not implemented
yet.

## Layers and dependencies

The `seldon_vault_lite` library is the testable root; the
`main.rs` binary only calls the CLI and converts errors into exit
codes. Dependencies point downward:

```text
cli / api  ->  model / index / storage  ->  binary / tensor / media  ->  error
```

Rules:

- `error` depends on no other internal module.
- `json` depends only on `error`.
- `binary` depends on `error`, `json`, and
  `tensor`.
- `tensor` depends only on `error`.
- `model` depends on `binary`, `json`, and
  `error`; it does not depend on inference or storage.
- Parsers do not depend on inference layers.
- Storage does not depend on the HTTP API.
- No module depends on a third-party crate.

## CLI contract

The CLI is unchanged at M1: `version` and `help`.

| Command | Effect | Output | Code |
|---|---|---|---|
| `version` | prints name, version, target | stdout | 0 |
| `help` | prints help, commands, and codes | stdout | 0 |
| `--version`, `-V` | alias of version | stdout | 0 |
| `--help`, `-h` | alias of help | stdout | 0 |
| no command | usage error | stderr | 2 |
| unknown command | usage error | stderr | 2 |
| extra argument | usage error | stderr | 2 |
| non-UTF-8 argument | usage error | stderr | 2 |
| write failure | execution error | stderr (best-effort) | 1 |
| closed pipe | normal end | — | 0 |

`serve`, `embed`, `index`, `search`,
and `verify` remain explicitly refused until implemented. No stub
returns a fake result.

## Error model

`src/error.rs` defines `Error` and `Result<T>`.
The variants are:

- `Usage` (exit code 2);
- `Io` (exit code 1);
- `Corrupt` (exit code 1);
- `LimitExceeded` (exit code 1);
- `Unsupported` (exit code 1);
- `Internal` (exit code 1).

`Display` produces a user-facing message; `source()` exposes
the technical cause for I/O errors. The HTTP API will reuse the same taxonomy
to choose statuses.

## Format boundaries

### Binary reader

`ByteReader` returns borrowed slices, uses checked arithmetic for
every range, and allocates only through `read_vec` with an explicit
cap. `read_file_bounded` checks metadata and reads through
`Read::take`.

### JSON

`src/json` is a strict RFC 8259 subset with configurable caps on
input size, depth, string size, container entries, and total values. It rejects
duplicate keys, trailing commas, non-finite numbers, lone surrogates, and
trailing content. Objects are ordered with a `BTreeMap`.

### Safetensors

`SafetensorsFile` borrows the file bytes, validates the header and
every tensor range, rejects overlaps, and exposes FP32 decoding and a SHA-256
fingerprint. FP16 is rejected because activations must stay FP32.

### Tensors

`Shape` validates rank, dimensions, and element count at
construction. `Tensor` is owned FP32 data capped at
`2^30` elements. `TensorView` borrows without copying.
`src/tensor/kernels.rs` is the deterministic scalar reference; SIMD
will be compared against it before any optimization is accepted.

## Model boundary

The model is frozen at revision
`f9972ac22e696a105c482202646979e33e4b5a1e`. The manifest is
`fixtures/model/manifest.json`.

### Tokenizer

`model::tokenizer::Tokenizer` loads the frozen tokenizer.json with
bounded JSON limits and validates the BPE pipeline before use:

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

`model::prompt` exposes the official prefixes and a `Prompt`
enum. A query and a document are distinct roles; a test compares every
constant with the frozen manifest.

### Model files

Model files live in `.model-cache/<revision>/`, which is ignored by
Git. Weights are never committed. The manifest records the weights' SHA-256
from the Hugging Face LFS object id.

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

- Target: `aarch64-apple-darwin` (enforced by
  `.cargo/config.toml`).
- Deployment: `MACOSX_DEPLOYMENT_TARGET=11.0` (forced).
- `rust-toolchain.toml`: stable toolchain (validated with Rust
  1.99.1), `clippy`, `rustfmt`.
- `Cargo.toml`: no dependencies; release profile
  `opt-level = 3`, `lto = "thin"`,
  `codegen-units = 1`.

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

`scripts/verify.sh` groups these commands and fails if the Cargo
graph contains a dependency or if `minos` is not 11.0.

## Next steps

1. **M1**: complete for the formats, tensor, model-freeze, and tokenizer scope.
   The model weights are not loaded and no inference is claimed.
2. **M2**: text embeddings, layer-output parity, pooling, the 512 to 768
   projection, and Matryoshka dimensions.
3. **M3**: persistent storage, BM25/KNN index, and RRF fusion.
4. **M4/M5**: PNG/JPEG/WAV decoders and multimodal embeddings.
5. **M6**: HTTP API, jobs, and hardening.
6. **M7**: NEON and macOS packaging.
7. **M8/M9**: final macOS validation, then Linux port.

None of these steps may be declared complete without reproducible tests and
without replaying the golden tests.
