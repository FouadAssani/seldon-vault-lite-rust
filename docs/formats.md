# Formats and binary primitives

This document describes the formats implemented through milestone M3 and
the validation rules that protect the product from hostile input. ADR 0002 is
the binary format decision record, ADR 0004 covers the text-embedding numerical
path, and ADR 0005 covers persistent storage and hybrid search; this document
is the operational reference.

## Trust boundary

Everything read from disk, the network, or a command line is untrusted until a
parser has validated it. Parsers never panic, never allocate from an unchecked
length, and never execute metadata. Failures are typed:

| Error | Meaning | Exit code |
|---|---|---|
| `Usage` | invalid command or argument | 2 |
| `Io` | input/output failure | 1 |
| `Corrupt` | malformed or truncated input | 1 |
| `LimitExceeded` | a documented cap was reached | 1 |
| `Unsupported` | valid but unimplemented dtype or feature | 1 |
| `Internal` | violated product invariant | 1 |

## JSON

The strict parser lives in `src/json`. It implements the RFC 8259
subset needed by the product:

- values: object, array, string, number, boolean, null;
- strings: UTF-8, escapes, and `\uXXXX` including surrogate pairs;
- numbers: integer or decimal notation with an optional exponent;
- objects: ordered deterministically with a `BTreeMap`.

Rejected:

- duplicate keys;
- trailing commas and trailing content;
- leading zeros, a leading plus sign, `.5`, `1.`, and an
  empty exponent;
- NaN, Infinity, and values that overflow to a non-finite f64;
- unescaped control characters;
- lone surrogates and invalid escapes;
- invalid UTF-8 in byte input.

### Limits

| Limit | Default | Purpose |
|---|---|---|
| Input size | 16 MiB | configuration and Safetensors headers |
| Nesting depth | 128 | recursion and stack safety |
| String size | 1 MiB | avoid giant keys or values |
| Entries per container | 100,000 | avoid quadratic object work |
| Total values | 1,000,000 | bound parser work |

Limits are configurable with `json::Limits`.

### Serialization

`json::to_string` emits compact deterministic JSON. Object keys follow the
BTreeMap order; strings are escaped for quotes, backslashes, and control
characters. A non-finite float is rejected as an internal invariant error
because JSON cannot represent NaN or infinity.

### Numbers

`Number::Unsigned` and `Number::Signed` preserve integer
tokens exactly. `Number::Float` covers decimal and exponent notation
and integers outside the 64-bit ranges. Callers that require exact integers
must use `as_u64` or `as_i64`, which reject float tokens.

## Safetensors

Layout:

```text
+----------------------+ 8 bytes, little-endian
| header length (u64)  |
+----------------------+
| JSON header          | header length bytes, UTF-8
+----------------------+
| data region          | remaining bytes
+----------------------+
```

The header is an object. The optional `__metadata__` key maps strings
to strings. Every other key is a tensor name whose value is an object with
exactly `dtype`, `shape`, and `data_offsets`.

Supported dtypes: `F32` and `BF16`. `F16` is
rejected because FP16 is forbidden for activations.

### Validation

- header length fits `usize` and is at most 16 MiB;
- header is valid UTF-8 and a JSON object;
- at most 100,000 tensors and 10,000 metadata entries;
- dtype, shape, and offsets are present; unknown fields are rejected;
- shape entries are non-negative integer tokens, rank at most 8, element count
  checked for overflow;
- `begin <= end`; `end` is within the data region;
- the byte span equals `numel * dtype_size`;
- non-empty tensors are aligned to the element size;
- zero-sized tensors occupy no byte and cannot overlap;
- non-empty tensors may not overlap;
- metadata values are strings and are never executed.

### Decoding

`TensorEntry::to_f32` returns an owned FP32 `Tensor`. F32
bytes are read little-endian. BF16 is widened exactly. The result must respect
the tensor element cap.

### Fingerprint

`SafetensorsFile::fingerprint_hex` returns the lowercase SHA-256 of
the complete file. It is intended for the model manifest and for verifying
weights before use.

## BF16 and FP32

- BF16 is the upper 16 bits of FP32; decoding is exact.
- Encoding rounds to nearest, ties to even.
- NaN is quieted with its sign preserved.
- Values above the largest finite BF16 round to infinity.
- Activations are always FP32. BF16 is storage only.
- FP16 is rejected at dtype parsing.

## Tensors

- `Shape`: rank at most 8, dimension at most `u32::MAX`,
  element count checked at construction.
- `Tensor`: owned FP32 data with a validated shape; at most
  `2^30` elements (4 GiB).
- `TensorView`: borrowed shape and data, no copy.
- Row-major layout; strides are in elements.
- Non-finite values are detected before they enter the embedding space.

## Scalar kernels

The reference path is deterministic, single-threaded, and unsafe-free:

| Kernel | Notes |
|---|---|
| add, sub, mul, scale | element-wise, length-checked |
| dot | checked, rejects non-finite result |
| matmul | row-major, reusable output buffer |
| softmax | max subtraction, explicit infinity handling |
| RMSNorm | non-negative epsilon, weight applied after normalization |
| GELU tanh | Gemma-style approximation |
| GeGLU | gelu(gate) * up |
| RoPE | rotate-half, even head dimension; precomputed inverse frequencies |
| attention | scaled dot product, optional causal mask |
| bidirectional attention | grouped-query, symmetric sliding-window mask |
| linear | [out, in] weight rows, reusable output buffer |
| RMSNorm rows | per-row normalization with or without a learned scale |
| mean pooling | mean over rows of a [sequence, width] matrix |
| truncation | leading dimensions followed by L2 renormalization |
| L2 normalization | rejects zero vectors |

## Fixture

`fixtures/formats/mini.safetensors` is a deterministic 281-byte
fixture generated outside the product with Python's `json` and
`struct` modules. It contains:

- metadata `generator` and `version`;
- `alpha`: F32 `[2, 2]` with values 1, 2, 3, 4;
- `beta`: BF16 `[3]` with values 0.5, -1.0, 2.0;
- `empty`: F32 `[0]` with zero bytes.

SHA-256:
`caca3b030faeb4ded52275d54c2fca0f2ee0d6b1aeeb8ab1819e0785b1f89ca7`

The fixture is not a model parity fixture. It proves format handling only.

## Storage journal

Milestone M3 adds an append-only journal with framed records, CRC-32 header and
payload checksums, a written-last footer, monotonic sequence numbers, atomic
transactions, fsync at commit, and deterministic replay. It also adds versioned
snapshots with a bounded header, transaction records, a base sequence, and
preserved chunk ids, plus crash-safe compaction and lock-free read-only views.
The complete layouts, transaction payload schema, recovery rules, compaction
windows, and limits are documented in [docs/storage.md](storage.md) and decided
in [ADR 0005](adr/0005-persistent-storage-and-hybrid-search.md) and
[ADR 0006](adr/0006-snapshots-compaction-and-read-only-views.md).

The journal is treated as hostile input. A torn tail is truncated; committed
corruption, sequence gaps, unknown versions, invalid transaction semantics, and
non-normalized stored embeddings are reported as typed errors rather than
silently accepted.

## Internal weight container (planned)

The packer will produce a Rust-defined container with:

- magic and version;
- framed sections with bounded lengths;
- a manifest with names, dtypes, shapes, offsets, and SHA-256;
- a checksum over the container;
- no executable metadata.

The format will be implemented at the packaging milestone (M12). Until then,
weights remain external development files.

## Tokenization

Tokenization is implemented at M1 and text inference at M2. The model is frozen at revision
`f9972ac22e696a105c482202646979e33e4b5a1e`, and
`fixtures/model/manifest.json` records the tokenizer inventory and
file fingerprints. The Rust BPE implementation matches golden ids produced by
the official `tokenizers 0.23.3` library. See
`docs/tokenizer.md` and ADR 0003 for the pipeline, limits, fixtures,
and parity evidence.
