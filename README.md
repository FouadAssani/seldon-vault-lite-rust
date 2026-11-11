# Seldon Vault Lite

Local multimodal embedding, indexing, and search engine, written in Rust std
only (no third-party crates), specialized for google/embeddinggemma-2.

**Status: milestone M8 implemented on branch feat/m8-release-validation — a local-use end-to-end release harness, an automated release lifecycle test, ADR 0013, and the macOS local-use operations guide are added. The artifact is ad-hoc signed and not notarized; no separate clean-machine gate and no Developer ID signing or notarization are planned. The crate remains at 0.1.0 and no 1.0.0 distribution is claimed; distribution to other Macs is out of scope.**
M0, M1, M2, and M3 are complete: the std-only foundations, strict binary
formats, FP32 tensor core, frozen model manifest, exact BPE tokenizer, official
prompts, the full EmbeddingGemma 2 text encoder with Matryoshka dimensions, and
the persistent journal/snapshot store with BM25, exact vector search, RRF
hybrid fusion, lock-free read-only views, and the CLI index, search, verify,
and compact commands. M4 adds a native zlib/DEFLATE decoder, a strict PNG
parser for non-interlaced 8-bit grayscale, RGB, and RGBA images, and a strict
baseline JPEG decoder for 8-bit grayscale and three-component images. It also
adds the official image preprocessing (bit-exact antialiased bicubic resize,
RGB/RGBA conversion, 16x16 patches, and positions), the 16-layer Gemma 4 vision
tower with axial RoPE and 3x3 pooling, and the image embedder that connects
vision soft tokens to the text encoder. Image-only embeddings are validated
against the official reference for grayscale, RGB, RGBA, gradient, quadrant,
rotation, and non-square aspect-ratio fixtures. The CLI embeds images, indexes
image documents, and runs text-to-image and image-to-image vector search; both
ranking paths are covered by reference tests. M5 adds a native RIFF/WAVE decoder
for PCM 8/16/24/32 and IEEE float 32/64, mono downmix, deterministic
windowed-sinc resampling to 16 kHz, the official 128-bin log-mel preprocessing,
and the 12-layer Gemma 4 audio tower with sinusoidal relative positions,
chunked local attention, clipped linears, and light convolution. Audio-only
embeddings are validated against official goldens for silence, short and long
tones, French speech, music, and rain-like noise. The CLI embeds audio, indexes
audio documents, and runs text-to-audio and audio-to-audio vector search; both
ranking paths are covered by reference tests. The scalar vision and audio
towers are correctness baselines, not optimized results; broader ranking quality
and performance remain future work. M6 adds a native std::net::TcpListener
HTTP/1.1 server with a strict parser and bounded request bodies, the documented
routes health, ready, models, stats, embeddings, documents, search, jobs, and
shutdown, asynchronous ingestion through a bounded worker queue, cooperative
cancellation, constant-time bearer authentication, loopback-only shutdown, and
no TLS. The HTTP API accepts media only as bounded base64 JSON, never as a
client-supplied path. MP3, streaming audio, TLS, keep-alive, and HNSW remain
future work. M7 adds runtime-selected NEON kernels for dot products, elementwise
operations, matmul, linear projections, RMSNorm, L2 normalization, mean pooling,
and attention scores, always with a tested scalar fallback. The M2 text p50 fell
from 1.485 s to 159 ms and concurrent throughput rose from 2.43 to 24.47
requests/s on the reference Apple M2 Pro; the frozen text, image, and audio
parity suites still pass. The versioned model container and Rust std packer
are implemented and a real frozen bundle is verified. Bundle-backed loading
accepts a model directory or a `.svlm` file and produces identical embeddings;
a compile-time `embedded-model` feature embeds the bundle and ignores the
model path. The full 1.5 GB embedded build is verified: a 1.4 GiB Mach-O linking
only libSystem, `minos 11.0`, byte-identical text/image/audio vectors, 18.68 s
cold / 6.75 s warm startup, and about 2.44 GiB maximum RSS. The binary is
ad-hoc signed; local use works, while notarized distribution is out of scope.
M8 adds the local-use release validation: ADR 0013, docs/release.md,
scripts/release_e2e.sh, and tests/release_e2e.rs. The harness stages the
embedded binary and fixtures in a sanitized environment, verifies text, image,
and audio embeddings, ingestion, lexical/vector/hybrid search, verify, compact,
restart persistence, and the loopback HTTP lifecycle, and records the artifact
linkage, deployment target, signature, Gatekeeper assessment, and zero-crate
graph. No separate clean-machine gate and no Developer ID signing or
notarization are planned; the version remains 0.1.0 and no 1.0.0 distribution is
claimed.

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

The sanitized release harness is documented in
[docs/release.md](docs/release.md):

    scripts/release_e2e.sh --binary target/embedded-full/aarch64-apple-darwin/release/seldon-vault-lite

Text embedding with the frozen model cache:

    cargo run --release -- embed --prompt SearchQuery "What causes the northern lights?"

Image embedding with the frozen model cache (PNG or JPEG; expensive on the
scalar path):

    cargo run --release -- embed --image fixtures/media/quadrants_16x16.png --dim 768

Audio embedding with the frozen model cache (WAV; the clip is converted to
16 kHz mono and capped at 30 seconds):

    cargo run --release -- embed --audio fixtures/media/audio/short_tone_100ms.wav --dim 768

Token ids without the tokenizer, useful for diagnostics:

    cargo run -- embed --model-dir fixtures/model/tiny_text --ids 1,5,7,9,2 --dim 8

Start the local HTTP API (default 127.0.0.1:8080; a non-loopback bind requires
a token of at least 16 bytes, and there is no TLS):

    cargo run --release -- serve --data-dir ./data
    curl -s http://127.0.0.1:8080/health
    curl -s http://127.0.0.1:8080/v1/models
    curl -s -X POST http://127.0.0.1:8080/v1/shutdown \
      -H 'Content-Type: application/json'

See [docs/api.md](docs/api.md) for the request schemas, limits, error codes, job
lifecycle, and security notes.

Index a JSONL file and search it:

    cargo run --release -- index --data-dir ./data documents.jsonl
    cargo run --release -- search --data-dir ./data --mode hybrid "ownership in Rust"
    cargo run -- verify --data-dir ./data
    cargo run -- compact --data-dir ./data

Index an image document and search it with a text query:

    echo '{"id":"img-1","image":"fixtures/media/quadrants_16x16.png","title":"Quadrants"}' > images.jsonl
    cargo run --release -- index --data-dir ./data --vision-tokens 70 images.jsonl
    cargo run --release -- search --data-dir ./data --mode vector "a colorful four-quadrant test pattern"

Image lines have no BM25 text; they are retrieved by exact vector search. The
vision budget is one of 70, 140, 280, 560, or 1120; 70 keeps the scalar path
practical.

Index an audio document and search it with text or another audio clip:

    echo '{"id":"rain-1","audio":"fixtures/media/audio/rain_noise_2s.wav","title":"Rain"}' > audio.jsonl
    cargo run --release -- index --data-dir ./data audio.jsonl
    cargo run --release -- search --data-dir ./data --mode vector "rain falling"
    cargo run --release -- search --data-dir ./data --mode vector --audio fixtures/media/audio/music_chords_2s.wav

Audio lines have no BM25 text; they are retrieved by exact vector search. The
30-second cap is enforced by the native decoder and preprocessing path; no
transcription is performed.

For reproducible tests without the full tokenizer, index and vector search
accept a diagnostic --ids path against the tiny official model:

    cargo run -- index --data-dir ./data --model-dir fixtures/model/tiny_text \
      --dim 8 --ids documents.jsonl

The full-model parity test runs when .model-cache/<revision>/model.safetensors
is present. Set SELDON_SKIP_MODEL_PARITY=1 to skip it during a quick check;
the tiny official fixture still runs. The image preprocessing test always runs.
The image embedding parity test runs when the model cache is present and is
expensive because the scalar vision tower processes 576 patches; run it with:

    cargo test --release --test image_embeddings image_embedding_matches -- --nocapture

The full seven-case image corpus is an ignored test:

    cargo test --release --test image_embeddings -- --ignored --nocapture

The tiny audio fixture test runs by default without the frozen model cache and
covers the audio tower, projection, and text connection. Audio preprocessing,
WAV decoding, and the short-tone audio embedding parity test run when the model
cache is present. The full six-case audio parity and text-to-audio ranking
tests are ignored because they load the frozen model and embed several clips:

    cargo test --release --test audio_embeddings audio_embedding_matches_the_official_reference -- --nocapture
    cargo test --release --test audio_embeddings all_audio_embeddings_match_the_official_reference -- --ignored --nocapture
    cargo test --release --test audio_embeddings text_queries_rank_the_expected_audio_clip_first -- --ignored --nocapture
    cargo test --release --test cli audio_documents_can_be_indexed_and_found_by_text_or_audio -- --ignored --nocapture

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
| CLI embed (text, token ids, images, and audio), index (text, image, and audio), search, verify, compact, version, help | implemented |
| Native zlib/DEFLATE decoder (stored, fixed, dynamic) | implemented |
| Strict PNG decoder for 8-bit gray/RGB/RGBA, non-interlaced | implemented |
| PNG dimension, pixel, chunk, and decompression caps | implemented |
| Strict baseline JPEG decoder for 8-bit gray and three-component images | implemented |
| JPEG quantization, Huffman, entropy, restart, and marker validation | implemented |
| JPEG dimension, pixel, MCU, block, and decoded-memory caps | implemented |
| Shared validated decoded-image type for PNG and JPEG | implemented |
| Official image preprocessing with bit-exact resize and patches | implemented |
| CLI image embedding through the official preprocessing and vision tower | implemented |
| Vision tower, axial RoPE, 3x3 pooling, and text projection | implemented |
| Image-only embeddings in the shared 768-dimensional space | implemented |
| Image storage and CLI image indexing | implemented |
| Vision parity for rotation and aspect-ratio fixtures | implemented |
| Text-to-image and image-to-image ranking smoke tests (two images) | implemented |
| Native RIFF/WAVE decoder for PCM 8/16/24/32 and IEEE float 32/64 | implemented |
| WAV file, data, channel, rate, and frame caps; codec rejection | implemented |
| Mono downmix and deterministic windowed-sinc resampling to 16 kHz | implemented |
| Official 128-bin log-mel preprocessing and frame mask | implemented |
| Audio tower, sinusoidal relative positions, chunked local attention, light convolution | implemented |
| Audio-only embeddings in the shared 768-dimensional space | implemented |
| Audio storage and CLI audio indexing | implemented |
| Text-to-audio and audio-to-audio ranking smoke tests (six clips) | implemented |
| Broader ranking quality, vision performance, and audio performance | future work |
| MP3 and streaming audio | future work |
| HNSW approximate vector index | deferred until measurements |
| Native HTTP/1.1 server with strict request parsing | implemented (M6) |
| GET health, ready, models, stats; POST embeddings, documents, search, shutdown | implemented (M6) |
| DELETE documents and jobs; GET job status and cancellation | implemented (M6) |
| Bounded connections, request bodies, headers, and read/write timeouts | implemented (M6) |
| Bounded ingestion queue, worker threads, and cooperative cancellation | implemented (M6) |
| Bearer token for non-loopback binds; loopback-only shutdown | implemented (M6) |
| Base64 media input; no client-supplied local paths | implemented (M6) |
| Runtime-selected NEON kernels with scalar fallback | implemented (M7) |
| Differential scalar-vs-NEON kernel tests | implemented (M7) |
| Frozen text, image, and audio parity replay after NEON | implemented (M7) |
| Versioned model container and Rust std packer | implemented (M7) |
| Bundle-backed loading from a model directory or .svlm file | implemented (M7) |
| Compile-time embedded model bundle (embedded-model feature) | implemented (M7) |
| Full 1.5 GB embedded Mach-O size/startup verification | implemented (M7) |
| Sanitized release lifecycle harness | implemented (M8) |
| Automated release lifecycle integration test | implemented (M8) |
| macOS local-use release and operations guide | implemented (M8) |
| Local-use release validation (no clean-machine or signing gate) | implemented (M8) |
| TLS, keep-alive, and HTTP/2 | future work |

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
- [ADR 0009 — Image preprocessing and vision tower](docs/adr/0009-image-preprocessing-and-vision.md)
- [ADR 0010 — Native WAV and audio embeddings](docs/adr/0010-native-wav-and-audio-embeddings.md)
- [ADR 0011 — Native HTTP API, jobs, and hardening](docs/adr/0011-http-api-and-jobs.md)
- [ADR 0012 — NEON kernels and macOS ARM64 packaging](docs/adr/0012-neon-simd-and-packaging.md)
- [ADR 0013 — macOS local-use release validation](docs/adr/0013-macos-release-validation.md)
- [macOS local-use release and operations guide](docs/release.md)
- [HTTP API contract](docs/api.md)
- [Architecture (M8 state)](docs/architecture.md)
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
- **HTTP API**: native std::net::TcpListener HTTP/1.1, strict bounded parsing,
  JSON errors, bounded connections and jobs, loopback default, bearer token for
  non-loopback binds, no TLS, and loopback-only graceful shutdown.

## Rust skills (DSH)

The skills from the [actionbook/rust-skills](https://github.com/actionbook/rust-skills)
repository are installed in .dsh/skills/ (version 2.0.9, commit 5c40d3a).

Update:

    git clone --depth 1 https://github.com/actionbook/rust-skills.git /tmp/rust-skills
    cp -R /tmp/rust-skills/skills/. .dsh/skills/
