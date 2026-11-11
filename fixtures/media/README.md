# PNG decoder fixtures

These tiny PNG files exercise the native decoder at the start of milestone M4.
They are committed because they are a few hundred bytes each, not model-scale
data.

## Provenance

The files were generated outside the product with Python 3 standard-library
modules only (struct, zlib, hashlib). No third-party image library, external
decoder, or network access was used. The generator computed PNG chunks, CRCs,
per-row filters, zlib compression, and Adler-32 checksums directly.

The fixtures are not model-parity data and do not claim official image
preprocessing. They prove PNG chunk parsing, zlib/DEFLATE decoding, filter
reconstruction, dimension caps, and corruption rejection.

## Inventory

| File | Size | SHA-256 | Dimensions | Color | Row filters | zlib first block |
|---|---|---|---|---|---|---|
| rgb_2x2.png | 78 | edbfc60195995fc33f65a2d3cfaa65fffa3c0c68246af114c7076fec71e12c89 | 2x2 | RGB8 | 0, 1 (Sub) | fixed |
| rgba_2x2.png | 80 | fa6cf6c3de0c8a91ab7bb255e4b7b77d2397cd4f21732b7168dd6ec375bd0db8 | 2x2 | RGBA8 | 0, 4 (Paeth) | fixed |
| gray_2x2.png | 71 | e1777c3fabfd5ed4697f373347c367393cc219b80c7d6e4c609a41afb24be198 | 2x2 | gray8 | 3 (Average), 2 (Up) | fixed |
| gradient_16x16.png | 625 | 87b57cb2838c25880a9695ddc42063c44d32c31b8087fb05c8c74f8e90bc6428 | 16x16 | RGB8 | 0 for every row | dynamic Huffman |

Expected pixels:

- rgb_2x2: row 0 red, green; row 1 blue, white.
- rgba_2x2: row 0 opaque red, half-transparent green; row 1 quarter-alpha
  blue, transparent white.
- gray_2x2: row 0 values 0, 64; row 1 values 128, 255.
- gradient_16x16: pixel (x, y) is (17x, 17y, 8(x + y)) modulo 256.

## Regeneration contract

Any replacement fixture must record its size, SHA-256, dimensions, color type,
row filters, and zlib block type here, and must be reproducible with the same
standard-library-only procedure. Corrupted variants are generated inside the
Rust tests by truncating or mutating these files; they are not committed.
