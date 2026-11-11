# ADR 0007 — Native PNG and DEFLATE decoding for the image path

- **Status**: accepted
- **Date**: 2026-11-11
- **Milestone**: M4 (image path started)
- **Decision maker**: project maintainer
- **Supersedes**: none
- **Superseded by**: none

## Context

Milestone M4 begins the image path. The product must decode PNG images entirely
in Rust std, without an external process, image library, FFI, or third-party
crate. PNG stores scanlines as a zlib stream, so a native DEFLATE decoder is a
prerequisite for the PNG parser and for any later compressed media format.

PNG and DEFLATE are hostile inputs. A file can claim huge dimensions, contain a
decompression bomb, corrupt a chunk checksum, truncate a chunk, use unsupported
color types, or encode impossible Huffman codes. The decoder must reject these
cases with typed errors and never allocate beyond documented caps.

This ADR covers the first verifiable image-path step: PNG decoding and its
DEFLATE foundation. It does not claim image preprocessing, visual embeddings, or
cross-modal search.

## Decisions

### D1 — Supported PNG variants

The decoder accepts non-interlaced PNG images with:

- 8-bit samples;
- color type 0 (grayscale, one channel);
- color type 2 (RGB, three channels);
- color type 6 (RGBA, four channels);
- compression method 0 and filter method 0;
- all five scanline filters: None, Sub, Up, Average, and Paeth.

The decoder rejects palette images, 16-bit samples, interlaced images, preset
zlib dictionaries, unknown critical chunks, and the tRNS transparency chunk.
These are reported as unsupported, not approximated. Expanding the variant
matrix requires new fixtures and a documented decision.

### D2 — Native zlib and DEFLATE

The zlib decoder accepts:

- a valid CMF/FLG header with method 8, window at most 32 KiB, and valid check
  bits;
- no preset dictionary;
- stored, fixed-Huffman, and dynamic-Huffman DEFLATE blocks;
- LZ77 back-references with overlapping copies;
- a four-byte big-endian Adler-32 trailer that matches the decompressed bytes.

The stream must end exactly after the Adler-32 trailer; trailing bytes are
corruption. Truncated bitstreams, invalid Huffman codes, oversubscribed tables,
missing end-of-block symbols, impossible distances, and invalid repeat counts
are corruption errors.

Huffman decoding uses canonical codes and validates code lengths before use.
The bit reader is bounded and never reads past the input.

### D3 — Explicit caps

| Limit | Value |
|---|---|
| PNG file size | 64 MiB |
| PNG chunk size | 32 MiB |
| Image width | 16,384 |
| Image height | 16,384 |
| Pixel count | 16,777,216 |
| Decompressed scanline bytes | 64 MiB |
| Compressed zlib input | 64 MiB |

Dimensions, pixel count, and expected decompressed length are checked before
inflation. Every output byte is checked against the decompressed cap, including
stored-block copies and LZ77 back-references. A decompression bomb therefore
fails with Error::LimitExceeded instead of allocating unbounded memory.

### D4 — PNG chunk validation

The parser:

- requires the eight-byte PNG signature;
- requires IHDR to be the first chunk and to appear exactly once;
- validates every chunk length against the remaining bytes and the chunk cap;
- validates every chunk CRC-32 over the chunk type and data;
- requires at least one IDAT chunk and a zero-length IEND;
- rejects trailing bytes after IEND;
- ignores unknown ancillary chunks;
- rejects unknown critical chunks, PLTE, and tRNS as unsupported.

IHDR validation rejects zero dimensions, excessive dimensions or pixel counts,
unsupported bit depth, unsupported color type, non-zero compression or filter
methods, and non-zero interlace methods.

### D5 — Scanline reconstruction

The decompressed stream must equal height multiplied by one filter byte plus
width multiplied by channels. Each scanline's filter byte is validated before
reconstruction. Sub, Up, Average, and Paeth predictors use wrapping 8-bit
arithmetic and the previously reconstructed row. The result is interleaved
row-major pixels with one, three, or four channels.

### D6 — Fixtures and provenance

Four small PNG fixtures are committed under fixtures/media:

- rgb_2x2.png: RGB8 with None and Sub filters;
- rgba_2x2.png: RGBA8 with None and Paeth filters;
- gray_2x2.png: grayscale8 with Average and Up filters;
- gradient_16x16.png: RGB8 using a dynamic-Huffman zlib block.

They were generated outside the product with Python standard-library modules
only. Each fixture's size, SHA-256, dimensions, color type, row filters, and
zlib block type are recorded in fixtures/media/README.md. Corrupted variants
are generated inside Rust tests by truncation or mutation; they are not
committed.

### D7 — Deferred image work

Not claimed by this ADR:

- JPEG decoding;
- official image preprocessing (color conversion, resize, patch placement,
  positions, token budgets);
- the vision encoder and projection into the common embedding space;
- text-to-image and image-to-image search;
- image storage and HTTP ingestion.

These remain M4 work and must be validated against official reference vectors
before any image modality is declared supported.

### D8 — Verification

Tests cover:

- RGB, RGBA, grayscale, and dynamic-Huffman gradient fixtures;
- all five scanline filters;
- every truncation of a valid fixture;
- bad signature, chunk CRC mismatch, missing IEND or IDAT, invalid filter,
  decompressed length mismatch, and trailing bytes;
- unsupported bit depth, palette, transparency, interlacing, unknown critical
  chunks, and preset dictionaries;
- dimension, pixel-count, decompressed-size, chunk-size, and compressed-input
  caps;
- stored, fixed, and dynamic DEFLATE blocks, empty output, Adler-32 mismatch,
  truncated streams, bad headers, trailing bytes, and output caps.

## Consequences

### Positive

- The image path now has a native, bounded PNG decoder with no external
  dependency.
- The DEFLATE decoder is reusable by later compressed formats.
- Corruption and decompression bombs are rejected before large allocations.
- Fixtures make decoder regressions visible.

### Costs and risks

- The supported PNG variant matrix is deliberately narrow; palette, 16-bit, and
  interlaced images are rejected.
- The decoder is not yet connected to official image preprocessing or the
  vision encoder.
- The per-row allocation during unfiltering is simple and bounded; it is not
  optimized yet.
- JPEG and official preprocessing will add their own parsers, caps, and
  fixtures.

### Follow-up

- Implement the JPEG decoder with the same caps and fixture discipline.
- Inspect and reproduce the official image processor for resize, patches,
  positions, and the 280-token budget.
- Implement the vision encoder and projection.
- Validate image embeddings against official goldens before claiming support.
