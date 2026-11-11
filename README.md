# Seldon Vault Lite

Local multimodal embedding, indexing, and search engine, written in Rust std
only (no third-party crates), specialized for google/embeddinggemma-2.

**Status: milestone M4 started — native PNG, JPEG, and DEFLATE decoding.**
M0, M1, M2, and M3 are complete: the std-only foundations, strict binary
formats, FP32 tensor core, frozen model manifest, exact BPE tokenizer, official
prompts, the full EmbeddingGemma 2 text encoder with Matryoshka dimensions, and
the persistent journal/snapshot store with BM25, exact vector search, RRF
hybrid fusion, lock-free read-only views, and the CLI index, search, verify,
and compact commands. M4 starts the image path with a native zlib/DEFLATE
decoder, a strict PNG parser for non-interlaced 8-bit grayscale, RGB, and RGBA
images, and a strict baseline JPEG decoder for 8-bit grayscale and
three-component images with Huffman coding, quantization tables, restart
intervals, and bilinear chroma upsampling. Both decoders share one validated
decoded-image type and enforce file-size, dimension, pixel, block, MCU, and
decoded-memory caps. Official image preprocessing, the vision encoder, and
cross-modal search are not implemented yet; image and audio are not declared
supported. HNSW remains deferred until measurements justify it.

## Requirements

- macOS 11 or later on Apple Silicon (aarch64-apple-darwin);
- stable Rust toolchain (validated with 1.99.1), selected by
  rust-toolchain.toml with clippy, rustfmt, and the aarch64-apple-darwin
  target; the crate minimum is 1.89 because storage uses std file locking;
- no other tool is required to build.

The x86_64-apple-darwin target (Intel Mac) is explicitly excluded.

## Build and test

    cargo build
    cargo run -- version
    cargo run -- --help
    cargo test

Text embedding with the frozen model cache:

    cargo run --release -- embed --prompt SearchQuery "What causes the northern lights?"

Token ids without the tokenizer, useful for diagnostics:

    cargo run -- embed --model-dir fixtures/model/tiny_text --ids 1,5,7,9,2 --dim 8

Index a JSONL file and search it:

    cargo run --release -- index --data-dir ./data documents.jsonl
    cargo run --release -- search --data-dir ./data --mode hybrid "ownership in Rust"
    cargo run -- verify --data-dir ./data
    cargo run -- compact --data-dir ./data

For reproducible tests without the full tokenizer, index and vector search
accept a diagnostic --ids path against the tiny official model:

    cargo run -- index --data-dir ./data --model-dir fixtures/model/tiny_text \
      --dim 8 --ids documents.jsonl

The full-model parity test runs when .model-cache/<revision>/model.safetensors
is present. Set SELDON_SKIP_MODEL_PARITY=1 to skip it during a quick check;
the tiny official fixture still runs.

## What the current milestones provide

| Area | Status |
|---|---|
| Bounded byte reader, checked offsets, file-size cap | implemented |
| Streaming SHA-256 and file fingerprints | implemented |
| BF16 to FP32 decoding and FP32 to BF16 encoding | implemented |
| Strict JSON parser and deterministic serializer | implemented |
| Read-only Safetensors parser with offset/overlap validation | implemented |
| FP32 tensors, shapes, views, and element caps | implemented |
| Scalar kernels and deterministic FP32 reference path | implemented |
| Frozen model manifest with SHA-256 for every file | implemented |
| Exact BPE tokenizer and byte fallback | implemented |
| Official prompt prefixes with query/document roles | implemented |
| EmbeddingGemma 2 text encoder and Matryoshka dimensions | implemented |
| Append-only journal with framed CRC-32 records | implemented |
| Atomic commit, fsync, deterministic replay, crash recovery | implemented |
| Versioned snapshots and safe compaction | implemented |
| Crash recovery at every snapshot and compaction write step | implemented |
| Exclusive writer lock and lock-free read-only views | implemented |
| Persistent documents, chunks, metadata, and normalized vectors | implemented |
| BM25 lexical index with filters | implemented |
| Exact cosine vector search (correctness oracle) | implemented |
| Hybrid RRF fusion and pagination | implemented |
| CLI embed, index, search, verify, compact, version, help | implemented |
| Native zlib/DEFLATE decoder (stored, fixed, dynamic) | implemented |
| Strict PNG decoder for 8-bit gray/RGB/RGBA, non-interlaced | implemented |
| PNG dimension, pixel, chunk, and decompression caps | implemented |
| Strict baseline JPEG decoder for 8-bit gray and three-component images | implemented |
| JPEG quantization, Huffman, entropy, restart, and marker validation | implemented |
| JPEG dimension, pixel, MCU, block, and decoded-memory caps | implemented |
| Shared validated decoded-image type for PNG and JPEG | implemented |
| Official image preprocessing and vision encoder | pending (M4) |
| HNSW approximate vector index | deferred until measurements |
| HTTP API, jobs, and ingestion queue | pending (M6) |
| Image and audio embeddings | pending (M4/M5) |

## Documentation

- [Implementation plan](seldon-vault-lite-plan.md)
- [Agent and contributor guidelines](AGENTS.md)
- [ADR 0001 — Foundations](docs/adr/0001-foundations.md)
- [ADR 0002 — Formats and tensors](docs/adr/0002-formats-and-tensors.md)
- [ADR 0003 — Model freeze and tokenizer](docs/adr/0003-model-freeze-and-tokenizer.md)
- [ADR 0004 — Text embeddings](docs/adr/0004-text-embeddings.md)
- [ADR 0005 — Persistent storage and hybrid search](docs/adr/0005-persistent-storage-and-hybrid-search.md)
- [ADR 0006 — Snapshots, compaction, and read-only views](docs/adr/0006-snapshots-compaction-and-read-only-views.md)
- [ADR 0007 — Native PNG and DEFLATE decoding](docs/adr/0007-native-png-and-deflate-decoding.md)
- [ADR 0008 — Native baseline JPEG decoding](docs/adr/0008-native-jpeg-decoding.md)
- [Architecture (M4 state)](docs/architecture.md)
- [Persistent storage and hybrid search](docs/storage.md)
- [Formats and binary primitives](docs/formats.md)
- [Tokenizer and prompts](docs/tokenizer.md)
- [Numerical parity and reference datasets](docs/model-parity.md)
- [Text embedding benchmark baseline](docs/benchmarks.md)
- [Frozen model manifest](fixtures/model/manifest.json)
- [Fixtures policy](fixtures/README.md)

All project documentation and code are written in English.

## Full verification

The scripts/verify.sh script runs the entire quality gate. The equivalent
commands are:

    cargo fmt --check
    cargo clippy --all-targets -- -D warnings
    cargo test
    cargo build --release --target aarch64-apple-darwin

    # "Zero crate" contract: only the root crate must appear.
    cargo tree --edges normal,build,dev

    # Linked libraries and Mach-O deployment target.
    otool -L target/aarch64-apple-darwin/release/seldon-vault-lite
    vtool -show-build target/aarch64-apple-darwin/release/seldon-vault-lite

Or simply:

    ./scripts/verify.sh

## Technical contract

- **Zero third-party crates**: Cargo.toml contains no dependency section; only
  std and core::arch are allowed.
- **100% Rust product code**: no C, C++, FFI, asm!, or dynamic application
  library.
- **Initial target**: aarch64-apple-darwin, macOS 11.0 minimum.
- **Persistent data**: user data, logs, and snapshots live outside the binary;
  weights are not committed.
- **Robustness**: no panic on hostile input, explicit memory bounds, checked
  arithmetic, deterministic FP32 scalar path.
- **Storage**: append-only journal, checksummed framed records, fsync at commit,
  deterministic replay, versioned snapshots, crash-safe compaction, exclusive
  writer lock, and lock-free read-only views.
- **CLI**: data on stdout, errors on stderr; exit codes 0 (success), 1
  (execution), 2 (usage).

## Rust skills (DSH)

The skills from the [actionbook/rust-skills](https://github.com/actionbook/rust-skills)
repository are installed in .dsh/skills/ (version 2.0.9, commit 5c40d3a).

Update:

    git clone --depth 1 https://github.com/actionbook/rust-skills.git /tmp/rust-skills
    cp -R /tmp/rust-skills/skills/. .dsh/skills/
