# ADR 0002 — Binary formats, strict JSON, Safetensors, and FP32 tensors

- **Status**: accepted
- **Date**: 2026-11-11
- **Milestone**: M1
- **Decision maker**: project maintainer
- **Supersedes**: none
- **Superseded by**: none

## Context

Milestone M1 must turn external files into trustworthy FP32 inputs: a bounded
byte reader, a strict JSON parser, a Safetensors reader, BF16 to FP32
conversion, validated tensors, and a deterministic scalar compute core. These
are the foundations of the first inference and of every later parser (PNG,
JPEG, WAV, HTTP JSON, and storage).

The model contract is unusual: no third-party crate, no external inference
engine, no FFI, and no unsafe code outside justified SIMD kernels. Parsers are
therefore product code and must be written and reviewed as such. M1 also
includes the frozen model reference and exact tokenization; those steps depend
on the official model files and are gated explicitly below.

## Decisions

### D1 — Bounded byte reader

All binary parsing goes through `ByteReader`:

- it borrows a byte slice and returns borrowed slices; it never allocates from
  an untrusted length;
- every offset addition uses `checked_add`;
- every read checks the end offset against the slice length before indexing;
- truncation and range overflow return `Error::Corrupt`, not a panic;
- the only allocating helper, `read_vec`, requires a caller-provided
  cap and returns `Error::LimitExceeded` before reading;
- `read_file_bounded` checks the regular-file metadata, rejects
  non-regular files, and reads through `Read::take` so a file that
  grows during the read cannot bypass the cap.

Endianness is explicit and can be changed mid-stream. Little-endian is the
default because Safetensors and RIFF use it.

### D2 — Strict JSON

The parser implements the RFC 8259 subset needed by configuration, Safetensors
headers, and the future HTTP API. It is not a naive string splitter.

Accepted: objects, arrays, booleans, null, numbers, strings, escapes, and
Unicode scalar values. Rejected:

- duplicate object keys;
- trailing commas and trailing content;
- leading zeros, a leading plus sign, a bare decimal point, and an empty
  exponent;
- NaN, Infinity, and numbers that overflow to a non-finite f64;
- unescaped control characters;
- lone surrogates and invalid escape sequences;
- invalid UTF-8 when parsing bytes.

Objects use a `BTreeMap` so iteration order is deterministic and does
not depend on a random seed.

Limits are explicit and configurable:

| Limit | Default |
|---|---|
| Input size | 16 MiB |
| Nesting depth | 128 |
| String size after unescaping | 1 MiB |
| Entries per array or object | 100,000 |
| Total values in the document | 1,000,000 |

Integers that fit in `u64` or `i64` are preserved exactly.
Integers outside those ranges, and numbers with a fraction or exponent, become
`f64`. Callers that need an exact integer use
`Number::as_u64` or `Number::as_i64`, which return
`None` for floating-point tokens.

### D3 — Safetensors reader

The reader is read-only and treats the header as hostile:

- the 8-byte header length must fit `usize` and stay under 16 MiB;
- the header must be valid UTF-8 and a JSON object;
- at most 100,000 tensors and 10,000 metadata entries;
- only `F32` and `BF16` are accepted; `F16` is
  rejected with `Error::Unsupported` because FP16 is forbidden for
  activations;
- a tensor descriptor must contain exactly `dtype`, `shape`,
  and `data_offsets`; unknown fields are rejected;
- shape entries must be non-negative integer tokens, the rank is capped at 8,
  and the element count is checked for overflow;
- `begin <= end`, `end` is inside the data region, the byte
  span equals `numel * dtype_size`, and non-empty tensors are aligned
  to the element size;
- zero-sized tensors occupy no byte and cannot overlap;
- non-empty tensors may not overlap;
- metadata values must be strings; metadata is never executed;
- a SHA-256 fingerprint of the complete file is exposed for manifests.

The parser borrows the file bytes and exposes `TensorEntry::to_f32`,
which widens BF16 exactly or reads little-endian F32. The resulting tensor
obeys the element cap of D5.

### D4 — BF16 to FP32 conversion

BF16 is the upper 16 bits of an FP32 value, so decoding is exact. Encoding uses
round-to-nearest, ties-to-even. NaN is quieted while preserving the sign bit,
so a hostile weight file cannot inject a signalling NaN. Values above the
largest finite BF16 value round to infinity, as IEEE-754 requires.

Internal tensors are always FP32. BF16 is a storage dtype only; it is never an
activation precision. FP16 is rejected at dtype parsing.

### D5 — FP32 tensors and shapes

- `Shape` validates rank (at most 8), each dimension (at most
  `u32::MAX`), and the element count with checked multiplication.
- Once constructed, `Shape::numel` is exact and cannot overflow.
- `Tensor` is an owned FP32 buffer with a validated shape. The cap is
  `2^30` elements, or 4 GiB of FP32 data. A hostile header cannot
  request an unbounded allocation.
- `TensorView` borrows a shape and data without copying.
- Row-major layout and strides are fixed.
- `validate_finite` detects NaN and infinities before they enter the
  embedding space.

### D6 — Reference scalar kernels

The scalar FP32 path is the correctness reference. It is single-threaded,
deterministic, and contains no unsafe code. M1 implements:

- element-wise addition, subtraction, multiplication, and scaling;
- dot product and in-place addition;
- row-major matmul with a reusable output buffer;
- numerically stable softmax;
- RMSNorm with a non-negative epsilon;
- GELU with the tanh approximation used by Gemma-style models, and GeGLU;
- rotary position embeddings with the rotate-half convention;
- scaled dot-product attention with optional causal masking;
- L2 normalization.

Validation rules:

- lengths and shapes are checked before any access;
- allocation sizes are capped;
- non-finite results are rejected where a silent NaN would corrupt the
  embedding space;
- softmax rejects all-negative-infinity rows and handles positive infinity
  explicitly;
- L2 normalization rejects zero vectors;
- no SIMD or unsafe is introduced in M1.

### D7 — Tokenization and model freeze are gated

The plan requires exact token ID parity. That cannot be validated without the
official tokenizer files and a frozen model revision. M1 therefore does not
fabricate tokenizer fixtures and does not claim tokenization support.

The next M1 steps are:

1. freeze `google/embeddinggemma-2` at a precise revision;
2. record the SHA-256 of every used file in a versioned manifest;
3. inventory layers, dimensions, dtypes, tensor names, prompts, and processor
   rules;
4. generate golden token IDs and tensor excerpts with the official
   implementation, outside the product;
5. implement the declared tokenizer variant and prove exact token ID equality.

Any token ID divergence blocks inference integration.

### D8 — Internal weight container

The packed container will be a versioned Rust-defined format with:

- a magic number and a schema version;
- framed sections with bounded lengths;
- a manifest mapping names to dtype, shape, offsets, and SHA-256;
- a checksum over the container;
- no executable metadata.

The implementation is deferred to the packaging milestone (M12). The
Safetensors reader and the SHA-256 implementation are the building blocks; no
third-party container format is introduced.

### D9 — Verification

The quality gate remains:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build --release --target aarch64-apple-darwin
cargo tree --edges normal,build,dev
otool -L target/aarch64-apple-darwin/release/seldon-vault-lite
vtool -show-build target/aarch64-apple-darwin/release/seldon-vault-lite
```

M1 adds:

- unit tests for the reader, SHA-256, BF16, JSON, Safetensors, shapes,
  tensors, and every kernel;
- corruption, truncation, overflow, and extreme-size tests;
- a deterministic Safetensors fixture at
  `fixtures/formats/mini.safetensors` with its SHA-256 recorded;
- integration tests through the public API.

## Consequences

### Positive

- External bytes cannot trigger a panic, an unbounded allocation, or an
  unchecked index.
- JSON and Safetensors are strict enough to reject malformed headers before
  any tensor is decoded.
- BF16 weights are accepted while activations stay FP32.
- The scalar kernels give a deterministic oracle for future NEON work.
- The fixture makes format regressions visible in continuous verification.

### Costs and risks

- The parsers are product code and must be security-reviewed; strictness
  increases maintenance.
- The 4 GiB tensor cap is generous for activations but still explicit. Model
  shards must be loaded tensor by tensor.
- Tokenization remains unproven until the model is frozen. No modality is
  declared supported at M1.
- The RoPE convention and the GELU approximation must be checked against the
  official implementation during parity work; a divergence is a blocking bug.

### Follow-up

- **ADR 0003 (M1/M3)**: journal format, snapshots, and transactions.
- **ADR 0004 (M6)**: HTTP contract, LAN authentication, and limits.
- Model manifest and tokenizer parity are the next M1 deliverables.
- NEON kernels will be compared against the scalar path before any timing
  claim.
