# Formats and binary primitives

This document describes the formats implemented through milestone M8 (macOS
release validation) and the validation rules that protect the product from
hostile input. M8 does not change a format; it adds the sanitized local-use
release validation. ADR 0002 is the binary format decision record, ADR
0004 covers the text-embedding numerical path, ADR 0005 covers persistent
storage and hybrid search, ADR 0007 covers native PNG/DEFLATE decoding, ADR 0008
covers native baseline JPEG decoding, ADR 0010 covers native WAV decoding and
audio embeddings, and ADR 0012 covers the NEON path and the model bundle; this
document is the operational reference.

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

## FP32 kernels and the NEON path

The scalar reference path is deterministic, single-threaded, and unsafe-free:

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

Milestone M7 adds a runtime-selected NEON path in `src/tensor/neon.rs` for
dot products, elementwise operations, matmul, linear projections, RMSNorm, L2
normalization, mean pooling, and attention scores. Every wrapper checks
`is_aarch64_feature_detected!("neon")` and falls back to the scalar function
when the feature is unavailable. The scalar path remains the correctness
oracle. Differential tests compare each NEON kernel with the scalar reference,
including tail lengths and real model shapes; the official text, image, and
audio parity suites are replayed with their existing thresholds.

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

## WAV and audio preprocessing

Milestone M5 adds a native RIFF/WAVE decoder in src/media/wav.rs, mono
conversion and resampling in src/media/audio.rs, and the official log-mel
processor in src/media/audio_preprocess.rs.

### Supported WAV variants

- RIFF/WAVE container.
- PCM (format tag 1) with 8, 16, 24, or 32 bits per sample.
- IEEE float (format tag 3) with 32 or 64 bits per sample.
- WAVE_FORMAT_EXTENSIBLE (0xFFFE) when its sub-format is PCM or IEEE float.
- One to eight channels, 8 kHz to 192 kHz.

Rejected as unsupported: ADPCM, A-law, mu-law, MP3, and every other compressed
codec; 12-bit PCM; float widths other than 32 or 64. Rejected as corrupt:
truncated RIFF headers, chunk sizes that cross the RIFF end, missing `fmt ` or
`data`, duplicate chunks, block alignment or byte rate that disagrees with the
declared channels and bits, a data size that is not a multiple of the block
alignment, a zero-frame payload, and non-finite float samples.

Channels are downmixed by averaging. PCM values are converted to FP32 in
[-1, 1]. The decoder returns mono FP32 at the source rate.

### Resampling

The model rate is 16 kHz. A non-16 kHz source is converted with a normalized
windowed-sinc kernel: 32 taps, a Hann window, and a cutoff of
`min(1, target/source)` to suppress aliasing when downsampling. The output is
truncated to 480,000 samples, matching the official feature-extractor defaults
`max_length=480000` and `truncation=True`. The conversion is deterministic and
uses only `std`.

### Caps

| Limit | Value |
|---|---|
| WAV file size | 64 MiB |
| WAV data chunk | 64 MiB |
| Channels | 8 |
| Source sample rate | 8 kHz to 192 kHz |
| Source frames | 5,760,000 (30 s at 192 kHz) |
| Model samples | 480,000 (30 s at 16 kHz) |
| Mel frames | 3,000 |
| Mel features | 128 |

### Official log-mel pipeline

1. Pad the 16 kHz mono waveform to a multiple of 128 samples.
2. Prepend 160 zeros (semicausal padding).
3. Extract 321-sample windows every 160 samples; drop the last sample of each
   window because preemphasis is zero.
4. Multiply by a periodic Hann window `0.5 - 0.5 cos(2πn/320)`.
5. Zero-pad to 512 samples and compute the real FFT; keep 257 magnitude bins.
6. Project onto 128 HTK mel filters from 0 Hz to 8000 Hz, with no area
   normalization.
7. Compute `ln(mel + 0.001)`.
8. Mark a frame valid only when the last sample of its 321-sample analysis
   window is real audio. Invalid frames are exactly zero.

The FFT and mel projection use f64 accumulation and round to f32 once. The
preprocessing golden compares frame counts, mask counts, and first values; the
path is not claimed bit-exact because a different radix-2 FFT can differ in the
last f32 bit. The observed maximum absolute difference on the short reference
clip is `4.77e-7`.

### Fixtures

fixtures/media/audio/ contains six short WAV excerpts. The synthetic clips were
generated with deterministic Python code; the French speech was generated with
the macOS `say` French voice and converted to 16 kHz mono PCM.
fixtures/media/audio_preprocess_golden.json records the padded sample count,
mel-frame count, valid-frame count, feature fingerprint, and first feature
values for every clip. Provenance and fingerprints are recorded in
fixtures/media/README.md. The golden generator is
scripts/generate_audio_goldens.py and is laboratory-only.

Audio documents are stored with modality audio and indexed from the CLI;
text-to-audio and audio-to-audio vector search are implemented and covered by
reference ranking tests. MP3, streaming audio, and broader ranking quality
remain future work. See ADR 0010.

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

## Model bundle (M7)

Milestone M7 adds `src/bundle`, a versioned container for the frozen model
files, and `tools/model-packer`, a Rust `std` packer. The development loaders
read a model directory or a `.svlm` bundle file. The final binary embeds the
bundle with `include_bytes!` and reads sections from the embedded bytes.

### Layout

All integers are little-endian.

| Offset | Size | Field |
|---|---|---|
| 0 | 8 | magic `SVLMDL01` |
| 8 | 2 | format version (1) |
| 10 | 4 | header length |
| 14 | 4 | section count |
| 18 | 8 | total bundle length |
| 26 | 4 | header CRC-32 |
| 30 | header length - 30 | section table |
| header length | sum of section lengths | section data |
| total length - 32 | 32 | catalog digest |

The header CRC-32 is computed over every header byte except the CRC field
itself. The catalog digest is the SHA-256 of the header bytes followed by the
ordered 32-byte section digests. It binds the section table to the section
content without hashing the data twice.

Each section entry is:

| Size | Field |
|---|---|
| 2 | path length |
| 1 | role |
| 1 | flags (0) |
| 8 | offset |
| 8 | length |
| 32 | SHA-256 of the section bytes |
| path length | UTF-8 relative path |

Roles: 0 manifest, 1 configuration, 2 tokenizer, 3 processor, 4 weights, 5
other.

### Validation

- magic and format version;
- header length and section count caps;
- declared total length equals the input length;
- header CRC-32;
- section table fits inside the header;
- paths are UTF-8, relative, normalized, and free of empty, `.`, and `..`
  components;
- paths are strictly sorted and unique;
- offsets are contiguous from the end of the header;
- every section ends before the catalog digest;
- the catalog digest matches;
- `verify` checks every section SHA-256.

The parser never allocates from an unchecked length and never executes
metadata. A corrupt or truncated bundle returns `Error::Corrupt`; a documented
cap returns `Error::LimitExceeded`.

### Caps

| Limit | Value |
|---|---|
| Total bundle | 8 GiB |
| Header | 1 MiB |
| Sections | 1,024 |
| Path | 4,096 bytes |

### Packer

The packer validates the frozen manifest revision, each selected file's size,
and its SHA-256 before writing. It writes to a temporary file, calls
`sync_all`, renames atomically, then reads the result back, parses it, and
verifies every section digest.

```bash
cargo build --manifest-path tools/model-packer/Cargo.toml --release
tools/model-packer/target/aarch64-apple-darwin/release/model-packer pack \
  --manifest fixtures/model/manifest.json \
  --model-dir .model-cache/f9972ac22e696a105c482202646979e33e4b5a1e \
  --output .model-cache/f9972ac22e696a105c482202646979e33e4b5a1e/model.svlm
```

The frozen revision produced 9 sections and 1,525,784,026 bytes, verified by
the `verify` command. The bundle includes the required `config.json`,
`model.safetensors`, and `tokenizer.json`, plus the official prompt, processor,
pooling, normalization, and tokenizer model files when the manifest lists them.

Weights are never committed to Git. The bundle is a development artifact in the
ignored model cache until the embedded loading path is implemented.

### Bundle-backed loading

`src/model/source.rs` defines `ModelFiles`, a source that is either a model
directory or a parsed bundle. The text, image, and audio loaders use the same
validation and decoding code for both. A `--model-dir` path that is a regular
file is read, structurally parsed, and fully verified once, then loaded from the
bundle sections. The returned model, tokenizer, and configuration own their
data, so the bundle bytes can be dropped after loading.

The loader reuses the section SHA-256 recorded in the bundle header as the
weights fingerprint instead of hashing the 1.5 GB Safetensors section a second
time. Measured on the reference Apple M2 Pro with the frozen revision:

| Source | Wall time | Maximum RSS | Embedding |
|---|---:|---:|---|
| Model directory | 6.96 s | 2.58 GiB | reference |
| `.svlm` bundle file | 7.40 s | 2.62 GiB | identical |

Both runs used `embed --ids 2,5,6,7,1 --dim 8` and produced the same bytes.
The bundle path additionally verifies the header CRC-32, the catalog digest, and
every section SHA-256.

The text, image, and audio bundle paths are covered by
`tests/bundle_loading.rs`. The tiny text fixture proves directory/bundle
equivalence in the fast suite; the ignored tests load the frozen 1.5 GB bundle
and check the text, vision, and audio models.

### Embedded loading

The `embedded-model` Cargo feature embeds the bundle at compile time:

```bash
SELDON_MODEL_BUNDLE=/absolute/path/to/model.svlm \
  cargo build --release --features embedded-model --target aarch64-apple-darwin
```

`src/model/source.rs` includes the file with
`include_bytes!(env!("SELDON_MODEL_BUNDLE"))`, parses and verifies it once in
a `OnceLock`, and all text, image, and audio loaders use the embedded sections.
The `--model-dir` argument is ignored in this mode. The feature has no
third-party dependency and is disabled by default.

A tiny-bundle embedded build was verified first: the binary produced the same
8-dimensional vector as the directory-loaded tiny model, ignored a nonexistent
`--model-dir`, and was 1.5 MB instead of 1.4 MB.

The full frozen 1,525,784,026-byte bundle was then embedded. The resulting
Mach-O is 1.4 GiB, links only `/usr/lib/libSystem.B.dylib`, and records
`minos 11.0`. The embedded binary produced byte-identical 768-dimensional
vectors to the directory-loaded binary for text token ids, a PNG image at the
70-token budget, and the 100 ms WAV tone. Cold startup was 18.68 s (paging the
1.4 GiB binary from disk); warm startup was 6.75 s. Maximum RSS was about
2.44 GiB and peak memory footprint about 1.02 GiB. See
[benchmarks.md](benchmarks.md) for the detailed measurements.

The binary is ad-hoc linker-signed. `spctl` rejects it because it is not
notarized; local execution works without a developer account. No Developer ID
signing or notarization is planned, and distribution to other Macs is out of
scope for the local-use release.

## Tokenization

Tokenization is implemented at M1 and text inference at M2. The model is frozen at revision
`f9972ac22e696a105c482202646979e33e4b5a1e`, and
`fixtures/model/manifest.json` records the tokenizer inventory and
file fingerprints. The Rust BPE implementation matches golden ids produced by
the official `tokenizers 0.23.3` library. See
`docs/tokenizer.md` and ADR 0003 for the pipeline, limits, fixtures,
and parity evidence.
