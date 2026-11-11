# ADR 0008 — Native baseline JPEG decoding for the image path

- **Status**: accepted
- **Date**: 2026-11-11
- **Milestone**: M4 (image path)
- **Decision maker**: project maintainer
- **Supersedes**: none
- **Superseded by**: none

## Context

Milestone M4 requires PNG and JPEG images to be decoded entirely in Rust std,
without an external process, image library, FFI, or third-party crate. ADR 0007
delivered the PNG parser and its bounded DEFLATE foundation. JPEG is the second
required image format and uses a different pipeline: marker segments,
quantization tables, canonical Huffman tables, entropy-coded 8x8 blocks, an
inverse discrete cosine transform, chroma subsampling, and YCbCr color
conversion.

JPEG is a hostile input. A file can claim huge dimensions, use unsupported
coding modes, contain truncated or invalid entropy data, reference missing
tables, request an impossible MCU layout, or trigger a decompression bomb. The
decoder must reject these cases with typed errors and must never allocate
beyond documented caps or panic on malformed bytes.

This ADR covers the second verifiable image-path step: native JPEG decoding. It
does not claim official image preprocessing, visual embeddings, or cross-modal
search.

## Decisions

### D1 — Supported JPEG variants

The decoder accepts baseline and extended-sequential Huffman JPEG files
(SOF0 and SOF1) with:

- 8-bit sample precision;
- one component (grayscale) or three components (YCbCr or RGB component ids);
- a single interleaved scan that contains every frame component;
- sampling factors from 1x1 to 4x4, with at most 10 blocks per MCU;
- 8-bit quantization tables;
- DC and AC Huffman tables with table ids 0 to 3;
- restart intervals declared by DRI, with RST0 to RST7 markers.

The decoder rejects progressive JPEG (SOF2), arithmetic-coded JPEG, lossless
and differential JPEG, 12-bit samples, 16-bit quantization tables, four or more
components (CMYK/YCCK), and non-interleaved multi-scan files as unsupported.
These variants are not approximated. Expanding the matrix requires new
fixtures and a documented decision.

### D2 — Marker parsing

The parser:

- requires the SOI marker and exactly one SOF marker;
- validates every segment length, including the two length bytes;
- validates DQT table ids, precision, nonzero 8-bit values, and the zigzag
  ordering of the 64 entries;
- validates DHT table class, id, code-length counts, symbol count, and code
  space before decoding;
- validates SOF component ids, sampling factors, quantization table ids,
  dimensions, pixel count, MCU count, block count, and decoded memory;
- validates SOS component ids, table selectors, component coverage, and the
  baseline spectral selection Ss=0, Se=63, Ah=Al=0;
- skips APPn and COM segments without executing or interpreting their content;
- rejects unknown markers as unsupported;
- requires an EOI marker and rejects trailing bytes after it.

Every marker segment is a bounded slice of the input. No length is trusted
before it is checked against the remaining file size.

### D3 — Entropy decoding

Entropy data is read MSB-first with JPEG byte stuffing: an 0xFF byte followed
by 0x00 represents a data byte 0xFF. A marker inside entropy data before all
blocks are decoded is corruption. Canonical Huffman decoding validates every
code against its length and rejects oversubscribed tables.

Each 8x8 block decodes one DC difference and up to 63 AC coefficients:

- DC categories above 11 and AC categories above 10 are rejected;
- EOB terminates the block; ZRL skips sixteen zero coefficients;
- runs that exceed the 63 AC positions are rejected;
- DC predictors are checked for overflow and reset at every restart marker;
- restart markers must match the expected RST index, and missing or wrong
  markers are corruption errors.

Truncated bitstreams, invalid Huffman codes, invalid symbols, and impossible
runs are typed corruption errors.

### D4 — Inverse DCT, upsampling, and color

The inverse DCT is a separable FP32 transform with precomputed cosine basis
values, level shift by 128, rounding, and clamping to 0 to 255. The scalar path
is deterministic and uses no unsafe code.

Chroma upsampling is bilinear with sample-center alignment. It is deterministic
and bounded, but it is not claimed to be bit-identical to any particular
reference decoder; decoder comparisons therefore use a documented tolerance.
Luma samples are passed through directly when the sampling factors match the
maximum factors.

For three components, component ids R/G/B (case-insensitive) are treated as
direct RGB. Otherwise the frame order is treated as YCbCr and converted with
the standard BT.601 coefficients:

- R = Y + 1.402 (Cr - 128)
- G = Y - 0.344136 (Cb - 128) - 0.714136 (Cr - 128)
- B = Y + 1.772 (Cb - 128)

An Adobe APP14 marker is validated: transform 1 is treated as YCbCr, transform
0 is accepted only when the component ids are R/G/B, and transform 2 (YCCK) or
an unknown transform is rejected as unsupported. This prevents silently
misdecoding Adobe RGB or YCCK files with numeric component ids.

The output is the shared media::Image type with one channel for grayscale or
three channels for RGB, matching the PNG decoder's representation.

### D5 — Explicit caps

| Limit | Value |
|---|---|
| JPEG file size | 64 MiB |
| Image width | 16,384 |
| Image height | 16,384 |
| Pixel count | 16,777,216 |
| Frame components | 3 |
| MCU count | 4,194,304 |
| Decoded 8x8 blocks | 4,000,000 |
| Decoded memory (planes plus output) | 128 MiB |

Dimensions, pixel count, component count, MCU count, block count, plane sizes,
and output size are checked with checked arithmetic before any large
allocation. The decoded-memory cap covers component planes plus the final
interleaved image.

### D6 — Fixtures and provenance

Five small JPEG fixtures are committed under fixtures/media:

- rgb_2x2.jpg: three components, 4:2:0 sampling, DRI interval 1;
- gray_2x2.jpg: one component, grayscale, 8x8 block;
- gradient_16x16.jpg: three components, 4:2:0 sampling, smooth gradients;
- quadrants_16x16.jpg: three components, 4:2:0 sampling, sharp color edges;
- quadrants_16x16_444.jpg: three components, 4:4:4 sampling.

Five reference PNGs are committed next to them for decoder comparisons:
rgb_2x2_ref.png, gray_2x2_ref.png, gradient_16x16_ref.png,
quadrants_16x16_ref.png, and quadrants_16x16_444_ref.png.

The JPEG files were generated outside the product from committed PNG or PPM
inputs with macOS sips 316 and cjpeg 3.1.0 (libjpeg-turbo). The reference PNGs
were rendered by sips from the JPEG files. These tools are laboratory-only and
are never launched by the product. Each fixture's size, SHA-256, dimensions,
sampling, and color layout is recorded in fixtures/media/README.md.

Corrupted variants are generated inside Rust tests by truncation or mutation;
they are not committed. The synthetic JPEG tests build minimal files in memory
to exercise exact DC-only IDCT output, restart markers, missing tables, invalid
Huffman code spaces, and caps.

### D7 — Deferred image work

Not claimed by this ADR:

- official image preprocessing (resize, patch placement, positions, token
  budget, normalization);
- the vision encoder and projection into the common embedding space;
- text-to-image and image-to-image search;
- image storage and HTTP ingestion;
- JPEG variants outside the documented baseline matrix.

These remain M4 work and must be validated against official reference vectors
before any image modality is declared supported.

### D8 — Verification

Tests cover:

- grayscale, 4:2:0 RGB, 4:4:4 RGB, smooth-gradient, and sharp-edge fixtures;
- every truncation of a valid JPEG fixture;
- bad SOI, trailing bytes, missing EOI, and missing entropy data;
- progressive, lossless, 12-bit, four-component, and 16-bit-quantization
  rejection;
- invalid Huffman code spaces, missing tables, invalid SOS parameters, and
  non-interleaved scans;
- Adobe APP14 YCbCr acceptance and YCCK/numeric-RGB rejection;
- restart intervals, wrong restart markers, and missing restart markers;
- dimension, height, and pixel caps;
- exact DC-only IDCT output and a nonzero DC coefficient;
- public API decoding, file decoding, and shared Image use by PNG and JPEG.

Fixture comparisons enforce per-byte tolerances: 2 for grayscale, 4 for the
small RGB fixture, 3 for 4:4:4 RGB, and 24 for 4:2:0 images with sharp color
edges or bilinear chroma differences. These tolerances cover decoder IDCT and
upsampling differences; they do not hide dimension, channel, or corruption
errors.

## Consequences

### Positive

- The image path now decodes PNG and baseline JPEG natively with no external
  dependency.
- JPEG entropy, Huffman, IDCT, chroma, and color paths have reproducible
  fixtures and synthetic exact tests.
- Hostile dimensions, MCU layouts, tables, and entropy streams are rejected
  before large allocations.
- PNG and JPEG share one decoded Image type, so preprocessing can consume
  either format without conversion glue.

### Costs and risks

- The supported JPEG matrix is deliberately narrow; progressive, arithmetic,
  CMYK, and 12-bit files are rejected.
- Bilinear upsampling is not bit-identical to every reference decoder, so
  comparisons use tolerances and official preprocessing parity remains open.
- The FP32 separable IDCT is a scalar reference; it is not optimized yet.
- Official image preprocessing and the vision encoder are still required
  before image embeddings can be claimed.

### Follow-up

- Inspect and reproduce the official image processor for resize, patch
  placement, positions, and the 280-token budget.
- Implement the vision encoder and projection into the shared space.
- Validate image embeddings against official goldens before claiming support.
- Add text-to-image and image-to-image ranking tests after parity is proven.
