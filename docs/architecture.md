# Architecture — milestone M4 state

## Purpose

This document describes the repository organization at milestone M4 and the
boundaries that later milestones must respect. It does not replace ADR 0001,
ADR 0002, ADR 0003, ADR 0004, ADR 0005, ADR 0006, ADR 0007, or ADR 0008,
which are authoritative for foundational, format, model, text-embedding,
storage/index, snapshot/compaction, PNG/DEFLATE, and JPEG decisions.

## Current state

The repository contains:

    Cargo.toml              # zero dependencies; Rust 1.89 minimum
    Cargo.lock              # reproducible locking
    rust-toolchain.toml     # stable, clippy, rustfmt, ARM64
    .cargo/config.toml      # aarch64-apple-darwin target, macOS 11.0
    src/
    ├── main.rs             # translates the CLI result into an exit code
    ├── lib.rs              # version, target, frozen revision constants
    ├── error.rs            # central error type
    ├── cli/mod.rs          # version, help, embed, index, search, verify, compact
    ├── binary/
    │   ├── mod.rs
    │   ├── reader.rs       # bounded byte reader and file cap
    │   ├── crc32.rs        # streaming CRC-32/ISO-HDLC
    │   ├── sha256.rs       # streaming SHA-256
    │   ├── bf16.rs         # BF16 <-> FP32 conversion
    │   └── safetensors.rs  # strict read-only Safetensors reader
    ├── json/mod.rs         # strict bounded JSON parser and serializer
    ├── model/
    │   ├── mod.rs
    │   ├── config.rs       # strict text configuration and tensor inventory
    │   ├── weights.rs      # validated FP32 text weight loader
    │   ├── text.rs         # EmbeddingGemma 2 text forward pass
    │   ├── prompt.rs       # official prompt prefixes
    │   └── tokenizer.rs    # exact BPE tokenizer for the frozen model
    ├── media/
    │   ├── mod.rs
    │   ├── image.rs        # shared validated decoded-image type
    │   ├── deflate.rs      # bounded zlib/DEFLATE decoder
    │   ├── png.rs          # strict PNG parser and scanline reconstruction
    │   └── jpeg.rs         # strict baseline JPEG parser and decoder
    ├── storage/
    │   ├── mod.rs
    │   ├── codec.rs        # bounded little-endian encoder/decoder
    │   ├── model.rs        # documents, chunks, metadata, transactions
    │   ├── journal.rs      # framed records, append, scan, recovery
    │   ├── snapshot.rs     # versioned header and atomic snapshot creation
    │   └── store.rs        # lock, replay, put/delete, compact, read-only, verify
    ├── index/
    │   ├── mod.rs          # filters, requests, hits, hybrid index
    │   ├── tokenize.rs     # deterministic Unicode tokenizer
    │   ├── lexical.rs      # BM25 inverted index
    │   ├── vector.rs       # exact cosine search
    │   └── hybrid.rs       # Reciprocal Rank Fusion
    └── tensor/
        ├── mod.rs
        ├── dtype.rs        # F32/BF16 dtype policy
        ├── shape.rs        # validated shapes and strides
        ├── owned.rs        # owned FP32 tensors and views
        └── kernels.rs      # scalar reference kernels and model operations
    tests/
    ├── cli.rs              # binary integration tests
    ├── formats.rs          # public API format/tensor integration tests
    ├── media.rs            # public API PNG/JPEG/DEFLATE integration tests
    ├── text_embeddings.rs  # tiny and full-model text parity tests
    └── tokenizer.rs        # golden tokenizer and manifest parity tests
    docs/
    ├── adr/0001-foundations.md
    ├── adr/0002-formats-and-tensors.md
    ├── adr/0003-model-freeze-and-tokenizer.md
    ├── adr/0004-text-embeddings.md
    ├── adr/0005-persistent-storage-and-hybrid-search.md
    ├── adr/0006-snapshots-compaction-and-read-only-views.md
    ├── adr/0007-native-png-and-deflate-decoding.md
    ├── adr/0008-native-jpeg-decoding.md
    ├── architecture.md
    ├── formats.md
    ├── storage.md
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
    ├── tokenizer/
    │   ├── golden.json      # official token ids for the golden corpus
    │   └── mini_tokenizer.json # reduced tokenizer for reproducible tests
    └── media/
        ├── README.md        # PNG/JPEG provenance, hashes, and tolerances
        ├── *.png            # PNG decoder inputs
        ├── *.jpg            # JPEG decoder inputs
        └── *_ref.png        # independent reference decodes
    tools/
    └── model-packer/
        └── README.md       # role planned at the packaging milestone
    scripts/
        └── verify.sh       # full milestone quality gate

No api or jobs module is created empty. They will appear with their first
testable feature, at the corresponding milestone. The media module now contains
the native zlib/DEFLATE decoder, the PNG decoder, the baseline JPEG decoder, and
the shared decoded-image type. Official image preprocessing and the vision
encoder are not implemented yet. Image embeddings and audio inference are not
claimed. Snapshots and compaction are implemented; see ADR 0006.

## Layers and dependencies

The seldon_vault_lite library is the testable root; main.rs only calls the CLI
and converts errors into exit codes. Dependencies point downward:

    cli / api  ->  model / index / storage  ->  binary / tensor / media  ->  error

Rules:

- error depends on no other internal module.
- json depends only on error.
- binary depends on error, json, and tensor.
- tensor depends only on error.
- media depends on binary (bounded file reads and CRC-32) and error; it does
  not depend on storage, model, index, CLI, or the HTTP API.
- storage depends on binary (bounded file reads and CRC-32) and error; it does
  not depend on the model, index, CLI, or HTTP API.
- index depends on storage and error; it does not depend on the model or CLI.
- model depends on binary, json, tensor, and error; it does not depend on
  storage or the HTTP API.
- cli depends on model, index, storage, json, and error.
- Parsers do not depend on inference layers.
- Storage does not depend on the HTTP API.
- No module depends on a third-party crate.

## CLI contract

| Command | Effect | Output | Code |
|---|---|---|---|
| version | prints name, version, target | stdout | 0 |
| help | prints help, commands, and codes | stdout | 0 |
| embed | loads the text model and prints one JSON vector | stdout | 0 |
| index | ingests a strict JSONL document file into the store | stdout | 0 |
| search | runs hybrid, lexical, or vector search | stdout | 0 |
| verify | validates the snapshot and journal prefix read-only | stdout | 0 |
| compact | materializes a snapshot and replaces the journal | stdout | 0 |
| --version, -V | alias of version | stdout | 0 |
| --help, -h | alias of help | stdout | 0 |
| no command | usage error | stderr | 2 |
| unknown command | usage error | stderr | 2 |
| extra argument | usage error | stderr | 2 |
| non-UTF-8 argument | usage error | stderr | 2 |
| invalid embed/index/search option | usage error | stderr | 2 |
| dimension above the model output | usage error | stderr | 2 |
| out-of-vocabulary token id | corruption error | stderr | 1 |
| malformed JSONL or unknown field | usage or corruption error | stderr | 1 or 2 |
| model or weight loading failure | execution error | stderr | 1 |
| storage lock contention | I/O error (WouldBlock) | stderr | 1 |
| journal corruption | corruption error | stderr | 1 |
| write failure | execution error | stderr (best-effort) | 1 |
| closed pipe | normal end | — | 0 |

embed accepts --model-dir, --prompt, --dim, --text, --ids, and positional
text. With no text or ids, it reads standard input with a 1 MiB cap. Text is
prefixed with the official prompt before tokenization; --ids bypasses the
tokenizer and the prefix for diagnostics.

index requires --data-dir and one JSONL file or -. Each line is an object with
id, text, optional title, optional source, and optional metadata. Duplicate
ids, unknown fields, malformed JSON, oversized lines, and more than 10,000
documents are rejected before any write. Embeddings use the Document prompt
and the frozen tokenizer. A diagnostic --ids mode accepts an ids array and
uses TextModel::embed_ids instead of the tokenizer; text is still stored for
BM25.

search requires --data-dir and accepts --model-dir, --dim, --mode
hybrid|lexical|vector, --limit, --offset, repeatable --filter KEY=VALUE, and a
diagnostic --ids. Lexical mode does not load the model. Vector and hybrid modes
embed the query with the SearchQuery prompt. The output is a JSON array with
chunk id, document id, ordinal, score, lexical and vector ranks, title,
source, modality, text, and metadata.

verify requires --data-dir and prints records, snapshot_records, documents,
chunks, and truncated_bytes. It does not acquire the writer lock and does not
modify the snapshot or journal.

compact requires --data-dir and prints snapshot_records, snapshot_bytes,
documents, chunks, and data_dir. It opens the writer store, materializes the
current state into snapshot.sv, and replaces journal.log with an empty file.

serve remains explicitly refused until M6. No stub returns a fake result.

## Error model

src/error.rs defines Error and Result<T>. The variants are:

- Usage (exit code 2);
- Io (exit code 1);
- Corrupt (exit code 1);
- LimitExceeded (exit code 1);
- Unsupported (exit code 1);
- Internal (exit code 1).

Display produces a user-facing message; source() exposes the technical cause
for I/O errors. The HTTP API will reuse the same taxonomy to choose statuses.

## Format boundaries

### Binary reader

ByteReader returns borrowed slices, uses checked arithmetic for every range,
and allocates only through read_vec with an explicit cap. read_file_bounded
checks metadata and reads through Read::take.

### JSON

src/json is a strict RFC 8259 subset with configurable caps on input size,
depth, string size, container entries, and total values. It rejects duplicate
keys, trailing commas, non-finite numbers, lone surrogates, and trailing
content. Objects are ordered with a BTreeMap. The serializer emits compact,
deterministic JSON with BTreeMap key order and rejects non-finite numbers.

### Safetensors

SafetensorsFile borrows the file bytes, validates the header and every tensor
range, rejects overlaps, and exposes FP32 decoding and a SHA-256 fingerprint.
FP16 is rejected because activations must stay FP32.

### Tensors and kernels

Shape validates rank, dimensions, and element count at construction. Tensor is
owned FP32 data capped at 2^30 elements. TensorView borrows without copying.
src/tensor/kernels.rs is the deterministic scalar reference. M2 adds linear
projection, per-row RMSNorm, mean pooling, truncation with renormalization,
RoPE, and bidirectional grouped-query attention with an optional sliding
window. SIMD will be compared against this path before any optimization is
accepted.

### Storage and journal

The storage format is documented in docs/storage.md and decided in ADR 0005
and ADR 0006. The journal is an append-only sequence of records with magic,
format version, record type, sequence, payload length, header CRC, payload CRC,
payload, and a footer. The footer is written last; a record is committed only
when all fields validate. A torn tail is truncated; committed corruption is
reported.

Transactions are bounded and atomic. The store applies them in memory only
after the record is synchronized. Replay validates schema version, sequence,
chunk-id order, document references, duplicate ordinals, embedding dimension,
and normalized embeddings before accepting a transaction.

A snapshot materializes the current documents and chunks in a versioned header
and transaction records. It preserves chunk ids, next_chunk_id, embedding
dimension, and a base journal sequence. Compaction writes snapshot.tmp,
synchronizes and renames it to snapshot.sv, then replaces journal.log with an
empty journal.tmp. A crash leaves the old generation, the new snapshot with the
old journal, or the new snapshot with an empty journal; all recover. Journal
records at or below the snapshot base are skipped.

The store holds an exclusive std file lock for writers. Compaction additionally
takes an exclusive compaction lock; read-only views take it shared while
loading. ReadOnlyStore replays the snapshot and committed journal prefix without
writer.lock and without modifying either file. A concurrent torn append is
reported and not applied. verify uses the same read-only path.

### Index

src/index rebuilds a BM25 inverted index and an exact vector view from the
committed store. Filters are applied before ranking. Lexical scores use BM25
with k1 = 1.2 and b = 0.75. Vector scores use cosine similarity with f64
accumulation. Hybrid scores use RRF with k = 60. All rankings break ties by
ascending chunk id, and pagination slices the deterministic ranking.

No approximate index is introduced at M3. The exact vector path is the
correctness oracle for any future HNSW work.

### Media

src/media/deflate.rs implements a bounded zlib/DEFLATE decoder with stored,
fixed-Huffman, and dynamic-Huffman blocks, LZ77 back-references, and an exact
Adler-32 trailer. It rejects truncated streams, invalid or oversubscribed
Huffman tables, impossible distances, bad checksums, preset dictionaries, and
trailing bytes.

src/media/png.rs implements a strict PNG parser for non-interlaced 8-bit
grayscale, RGB, and RGBA images. It validates the signature, IHDR fields, every
chunk length and CRC, dimensions, pixel count, decompressed size, filter bytes,
and the zlib stream. Palette, 16-bit, interlaced, tRNS, and unknown critical
chunks are rejected as unsupported. Caps are 64 MiB per file, 16,384 per
dimension, 16,777,216 pixels, and 64 MiB decompressed.

src/media/jpeg.rs implements a strict baseline and extended-sequential Huffman
JPEG decoder. It accepts 8-bit grayscale and three-component images with one
interleaved scan, 1x1 to 4x4 sampling factors, 8-bit quantization tables,
canonical DC/AC Huffman tables, and restart intervals. Progressive,
arithmetic-coded, lossless, differential, 12-bit, 16-bit-quantization, CMYK, and
non-interleaved multi-scan files are rejected as unsupported. The decoder
validates markers, segment lengths, tables, component coverage, entropy
stuffing, Huffman codes, coefficient runs, and restart markers before
producing pixels. The inverse DCT is a deterministic FP32 scalar transform;
chroma upsampling is bilinear; Adobe APP14 color transforms are validated.
Caps are 64 MiB per file, 16,384 per dimension,
16,777,216 pixels, 3 components, 4,194,304 MCUs, 4,000,000 blocks, and 128 MiB
of decoded planes plus output.

Both decoders return the shared media::Image type with interleaved 8-bit
samples and one, three, or four channels. Official image preprocessing, the
vision encoder, image storage, and cross-modal search are not implemented yet.
No image modality is claimed. See ADR 0007 and ADR 0008.

## Model boundary

The model is frozen at revision
f9972ac22e696a105c482202646979e33e4b5a1e. The manifest is
fixtures/model/manifest.json.

### Configuration

model::config::TextConfig parses the composite config.json (using its
text_config member) or a standalone text configuration. It rejects an unknown
model type, an unsupported dtype, FP16, a non-GELU activation, attention bias,
non-zero attention dropout, an unknown layer type, an unknown RoPE type,
inconsistent layer counts, a non-full last layer, and mismatched head layouts.
It exposes the expected tensor inventory so the loader can reject a wrong
weight file before decoding.

Limits include 16,384 hidden size, 128 layers, 128 heads, 1,024 head
dimension, 65,536 intermediate size, 1,000,000 vocabulary entries, 16,384
embedding dimension, and 1,048,576 positions.

### Weight loading

model::weights::TextWeights reads the frozen Safetensors file with a 4 GiB cap,
ignores audio and vision tensors, accepts only the 413 language_model. text
tensors, validates every name and shape, rejects extra or missing text tensors
and non-finite values, and converts BF16 or F32 storage to FP32 once. The
source SHA-256 is retained for manifest comparison. The FP32 weight set is
capped at 2^30 elements.

### Tokenizer

model::tokenizer::Tokenizer loads the frozen tokenizer.json with bounded JSON
limits and validates the BPE pipeline before use:

- model type BPE, byte fallback, fuse_unk, no ignore_merges;
- normalizer replaces a space with the SentencePiece marker;
- pre-tokenizer splits on a space with MergedWithPrevious;
- post-processor prepends <bos> and appends <eos>;
- decoder applies Replace, ByteFallback, and Fuse.

Encoding matches added tokens in raw text, normalizes spaces, starts from
single characters or byte tokens, and applies merges by rank. The input is
capped at 1 MiB and the output at 262,144 tokens. The full tokenizer.json is a
development asset outside Git; the reduced fixture reproduces it on the golden
corpus.

### Prompts

model::prompt exposes the official prefixes and a Prompt enum. Prompt::from_name
parses the official names. A query and a document are distinct roles; a test
compares every constant with the frozen manifest.

### Text inference

model::text::TextModel owns the configuration and FP32 weights. TextEmbedder
adds the tokenizer. The forward pass follows ADR 0004:

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
buffers are capped at 2^28 elements. The model is immutable and Send + Sync;
repeated calls are deterministic.

### Model files

Model files live in .model-cache/<revision>/, which is ignored by Git. Weights
are never committed. The manifest records the weights' SHA-256 from the Hugging
Face LFS object id. The tiny official model fixture is committed because it is
47.5 KiB, not model-scale data.

## Trust boundaries

Everything coming from outside is untrusted: CLI arguments, files, HTTP bodies,
JSON, Safetensors headers, tokenizer documents, images, audio, metadata, and
the journal. Each boundary must:

1. validate types, lengths, offsets, and versions before allocation;
2. cap dimensions and memory;
3. use checked arithmetic;
4. return a typed error, never a panic.

The FP32 scalar path remains the correctness reference for all numerical
computations. CRC-32 detects accidental journal corruption; it is not an
authentication mechanism against a local attacker with write access to the data
directory.

## Build configuration

- Target: aarch64-apple-darwin (enforced by .cargo/config.toml).
- Deployment: MACOSX_DEPLOYMENT_TARGET=11.0 (forced).
- rust-toolchain.toml: stable toolchain, clippy, rustfmt.
- Cargo.toml: no dependencies; rust-version 1.89 because storage uses std file
  locking; release profile opt-level = 3, lto = "thin", codegen-units = 1;
  test profile opt-level = 2 so the full-model parity suite is practical.

## Verification

    cargo fmt --check
    cargo clippy --all-targets -- -D warnings
    cargo test
    cargo build --release --target aarch64-apple-darwin
    cargo tree --edges normal,build,dev
    otool -L target/aarch64-apple-darwin/release/seldon-vault-lite
    vtool -show-build target/aarch64-apple-darwin/release/seldon-vault-lite

The full-model parity test is:

    cargo test --release --test text_embeddings real_model_parity -- --nocapture

It runs automatically when the model cache is present. Set
SELDON_SKIP_MODEL_PARITY=1 to skip it during a quick check.

M4 adds unit and integration tests for the zlib/DEFLATE decoder, the PNG
parser, and the JPEG decoder, including every truncation of valid PNG and JPEG
fixtures, corrupted PNG chunk CRCs, unsupported PNG and JPEG variants,
decompression bombs, all five PNG scanline filters, JPEG entropy and Huffman
validation, restart intervals, Adobe APP14 color-transform validation, exact
DC-only IDCT output, caps, and decoder comparisons against committed reference
PNGs. The media fixtures and their
provenance are recorded in fixtures/media/README.md.

scripts/verify.sh groups these commands and fails if the Cargo graph contains
a dependency or if minos is not 11.0.

## Next steps

1. **M3**: journal, snapshots, safe compaction, read-only views, BM25, exact
   vector search, and RRF are implemented. HNSW remains deferred until volume
   and latency measurements justify it.
2. **M4**: native PNG and baseline JPEG decoding are implemented, with a
   shared decoded-image type. Next: official image preprocessing, the vision
   encoder, and cross-modal tests.
3. **M5**: WAV and compliant audio embeddings.
4. **M6**: HTTP API, bounded job queue, and hardening.
5. **M7**: NEON and macOS packaging.
6. **M8/M9**: final macOS validation, then Linux port.

None of these steps may be declared complete without reproducible tests and
without replaying the golden tests.
