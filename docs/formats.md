# Formats and binary primitives

This document describes the formats implemented through milestone M4 (image
path started) and the validation rules that protect the product from hostile
input. ADR 0002 is the binary format decision record, ADR 0004 covers the
text-embedding numerical path, ADR 0005 covers persistent storage and hybrid
search, ADR 0007 covers native PNG/DEFLATE decoding, and ADR 0008 covers native
baseline JPEG decoding; this document is the operational reference.

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

## PNG and DEFLATE

Milestone M4 starts the image path with a native PNG decoder and the zlib/
DEFLATE decoder it needs. No external image library or process is used. ADR
0007 is the decision record; this section is the operational summary.

### PNG

Supported variants:

- non-interlaced 8-bit samples;
- grayscale (color type 0), RGB (2), and RGBA (6);
- compression method 0 and filter method 0;
- filters None, Sub, Up, Average, and Paeth.

Rejected as unsupported: palette, 16-bit samples, interlacing, preset
dictionaries, tRNS, and unknown critical chunks. Malformed lengths, CRC
mismatches, missing IHDR/IDAT/IEND, trailing bytes, invalid filters, and
decompressed-length mismatches are corruption errors.

### Caps

| Limit | Value |
|---|---|
| PNG file size | 64 MiB |
| Chunk size | 32 MiB |
| Width or height | 16,384 |
| Pixel count | 16,777,216 |
| Decompressed scanline bytes | 64 MiB |
| Compressed zlib input | 64 MiB |

Dimensions and expected decompressed size are checked before inflation. Every
inflated byte is checked against the output cap.

### DEFLATE

The zlib decoder accepts valid CMF/FLG headers, stored, fixed-Huffman, and
dynamic-Huffman blocks, LZ77 back-references, and an exact Adler-32 trailer.
Truncated streams, invalid or oversubscribed Huffman tables, impossible
distances, bad checksums, preset dictionaries, and trailing bytes are rejected.

The decoder lives in src/media/deflate.rs; the PNG parser lives in
src/media/png.rs. Fixtures and provenance are documented in
[fixtures/media/README.md](../fixtures/media/README.md).

## JPEG

Milestone M4 adds a native baseline and extended-sequential Huffman JPEG
decoder. No external image library or process is used. ADR 0008 is the decision
record; this section is the operational summary.

### Supported variants

- 8-bit sample precision;
- one component (grayscale) or three components (YCbCr or R/G/B component ids);
- one interleaved scan containing every frame component;
- sampling factors from 1x1 to 4x4, with at most 10 blocks per MCU;
- 8-bit quantization tables;
- canonical DC and AC Huffman tables with ids 0 to 3;
- restart intervals declared by DRI, with RST0 to RST7 markers.

Rejected as unsupported: progressive JPEG (SOF2), arithmetic-coded JPEG,
lossless and differential JPEG, 12-bit samples, 16-bit quantization tables,
four or more components (CMYK/YCCK), and non-interleaved multi-scan files.
Unknown markers are unsupported; malformed lengths, missing tables, invalid
Huffman code spaces, invalid spectral selection, truncated entropy data, wrong
restart markers, missing EOI, and trailing bytes are corruption errors.

### Marker and entropy validation

The parser requires SOI and exactly one SOF, validates every segment length,
DQT precision and zigzag values, DHT code lengths and symbol counts, SOF
component ids and sampling factors, SOS component coverage and baseline
spectral selection (Ss=0, Se=63, Ah=Al=0), and EOI. APPn and COM segments are
skipped without interpreting their content.

Entropy data is read MSB-first with byte stuffing. Canonical Huffman decoding
rejects oversubscribed tables and invalid codes. DC categories above 11 and AC
categories above 10 are rejected; EOB and ZRL are supported; runs beyond the 63
AC positions are corruption. DC predictors are reset at each restart marker.

### Reconstruction and color

The inverse DCT is a separable FP32 transform with precomputed cosine basis,
level shift by 128, rounding, and clamping. Chroma upsampling is bilinear with
sample-center alignment. Three components with R/G/B ids are treated as direct
RGB; otherwise frame order is treated as YCbCr and converted with BT.601
coefficients. An Adobe APP14 marker is validated: transform 1 is YCbCr,
transform 0 requires R/G/B component ids, and transform 2 or an unknown
transform is unsupported. The result is the shared media::Image type with one
or three channels.

Bilinear upsampling is deterministic and bounded, but it is not claimed to be
bit-identical to any particular reference decoder; decoder comparisons use
documented per-byte tolerances.

### Caps

| Limit | Value |
|---|---|
| JPEG file size | 64 MiB |
| Width or height | 16,384 |
| Pixel count | 16,777,216 |
| Frame components | 3 |
| MCU count | 4,194,304 |
| Decoded 8x8 blocks | 4,000,000 |
| Decoded memory (planes plus output) | 128 MiB |

Dimensions, pixel count, component count, MCU count, block count, plane sizes,
and output size are checked with checked arithmetic before large allocations.

### Fixtures

The committed JPEG fixtures are rgb_2x2.jpg, gray_2x2.jpg,
gradient_16x16.jpg, quadrants_16x16.jpg, and quadrants_16x16_444.jpg. Their
reference PNGs are rgb_2x2_ref.png, gray_2x2_ref.png, gradient_16x16_ref.png,
quadrants_16x16_ref.png, and quadrants_16x16_444_ref.png. Provenance, sizes,
SHA-256, sampling, and expected comparisons are recorded in
[fixtures/media/README.md](../fixtures/media/README.md).

The decoder lives in src/media/jpeg.rs; the shared image type lives in
src/media/image.rs.

## Image preprocessing

Milestone M4 adds the official EmbeddingGemma 2 image preprocessing in
src/media/preprocess.rs. It accepts the shared decoded-image type and returns
padded 16x16 patches plus (x, y) grid positions for the vision tower.

Pipeline:

1. Convert to RGB. Grayscale is replicated across three channels. RGBA is
   composited over white with the float tensor path and rounded half to even.
2. Compute the official aspect-ratio preserving target size: the largest size
   that fits the patch budget and whose sides are multiples of 48.
3. Resize with antialiased bicubic interpolation. The implementation follows
   the torchvision uint8 path: separable horizontal then vertical passes, Keys
   cubic convolution with a = -0.5, per-output normalized weights, 16-bit
   fixed-point quantization, and round-half-up integer accumulation. It is
   bit-exact with the reference on the committed fixtures.
4. Rescale bytes to [0, 1] by 1/255.
5. Emit patches in pixel-major, channel-last order: for each of the 16x16
   pixels, red, green, and blue.
6. Emit (x, y) positions and pad both the patch tensor and positions to the
   patch budget. Padding patches are zero and their positions are (-1, -1).

### Caps

| Limit | Value |
|---|---|
| Input side | 16,384 |
| Input pixels | 16,777,216 |
| Visual token budgets | 70, 140, 280, 560, 1,120 |
| Default soft tokens | 280 |
| Patch budget | 10,080 |
| Resize taps | 8,192 |
| Resize weights | 1,048,576 |
| Padded patch values | 7,741,440 |

The resized size, patch grid, patch count, and soft-token count are checked
before allocation. The resized sides are always divisible by 48, so the 3x3
pooling grid produces an integral number of soft tokens.

### Fixtures

fixtures/media/preprocess_golden.json records nine cases with source image,
dimensions, budget, target size, patch grid, valid patches, soft tokens,
SHA-256 of the resized bytes, padded patch values, and positions, plus selected
first and last patch values. The cases cover grayscale, RGB, RGBA, gradients,
quadrants, 90-degree and 180-degree rotations, and 2:1 and 1:2 aspect ratios.
Provenance and fingerprints are recorded in fixtures/media/README.md. The
generator is scripts/generate_image_goldens.py and is laboratory-only.

Image documents are stored with modality image and indexed from the CLI;
text-to-image and image-to-image vector search are implemented and covered by
reference ranking tests. Broader ranking quality and vision performance remain
future work. See ADR 0009.

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
