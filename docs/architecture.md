# Architecture — milestone M6 state

## Purpose

This document describes the repository organization at milestone M5 and the
boundaries that later milestones must respect. It does not replace ADR 0001,
ADR 0002, ADR 0003, ADR 0004, ADR 0005, ADR 0006, ADR 0007, ADR 0008,
ADR 0009, ADR 0010, or ADR 0011, which are authoritative for foundational,
format, model, text-embedding, storage/index, snapshot/compaction, PNG/DEFLATE,
JPEG, image preprocessing/vision, WAV/audio, and HTTP API/jobs decisions.

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
    ├── cli/mod.rs          # version, help, serve, embed, index, search, verify, compact
    ├── api/
    │   ├── mod.rs          # public server configuration and running-server surface
    │   ├── base64.rs       # strict bounded base64 decoder for media
    │   ├── engine.rs       # text model at startup, lazy vision/audio towers
    │   ├── http.rs         # strict HTTP/1.1 parser and response writer
    │   └── server.rs       # accept loop, routes, auth, limits, and ingestion task
    ├── jobs/
    │   └── mod.rs          # bounded queue, workers, cancellation, job snapshots
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
    │   ├── vision.rs       # strict vision configuration, weights, and tower
    │   ├── image.rs        # image prompt, vision-to-text connection
    │   ├── audio.rs        # strict audio configuration, weights, and tower
    │   ├── audio_embedder.rs # audio prompt, audio-to-text connection
    │   ├── prompt.rs       # official prompt prefixes
    │   └── tokenizer.rs    # exact BPE tokenizer for the frozen model
    ├── media/
    │   ├── mod.rs
    │   ├── image.rs        # shared validated decoded-image type
    │   ├── preprocess.rs   # official resize, RGB, patches, and positions
    │   ├── audio.rs        # mono samples, downmix, and windowed-sinc resampling
    │   ├── audio_preprocess.rs # official log-mel spectrogram and frame mask
    │   ├── wav.rs          # strict RIFF/WAVE PCM and IEEE-float decoder
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
    ├── tensor/
        ├── mod.rs
        ├── dtype.rs        # F32/BF16 dtype policy
        ├── shape.rs        # validated shapes and strides
        ├── owned.rs        # owned FP32 tensors and views
        └── kernels.rs      # scalar reference kernels and model operations
    tests/
    ├── cli.rs              # binary integration tests
    ├── formats.rs          # public API format/tensor integration tests
    ├── media.rs            # public API PNG/JPEG/DEFLATE integration tests
    ├── preprocess.rs       # official image preprocessing parity tests
    ├── image_embeddings.rs # image embedding parity tests
    ├── audio.rs            # WAV decoding and log-mel preprocessing tests
    ├── audio_embeddings.rs # audio embedding and text-to-audio ranking tests
    ├── audio_tiny.rs       # tiny official audio tower and text-connection tests
    ├── text_embeddings.rs  # tiny and full-model text parity tests
    ├── http_api.rs         # real-server HTTP/1.1 integration tests
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
    ├── adr/0009-image-preprocessing-and-vision.md
    ├── adr/0010-native-wav-and-audio-embeddings.md
    ├── adr/0011-http-api-and-jobs.md
    ├── api.md
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
    │   ├── image_golden.json # image embeddings and vision layer probes
    │   ├── audio_golden.json # six audio embeddings and 12 layer probes
    │   ├── tiny_text/       # complete 3-layer official model and golden outputs
    │   └── tiny_audio/      # 2-layer official audio tower and golden outputs
    ├── tokenizer/
    │   ├── golden.json      # official token ids for the golden corpus
    │   └── mini_tokenizer.json # reduced tokenizer for reproducible tests
    └── media/
        ├── README.md        # PNG/JPEG/audio/preprocessing provenance and hashes
        ├── preprocess_golden.json # official resize/patch/position hashes
        ├── audio_preprocess_golden.json # official log-mel frame fingerprints
        ├── audio/           # short WAV excerpts for decoding and parity
        ├── *.png            # PNG decoder inputs
        ├── *.jpg            # JPEG decoder inputs
        └── *_ref.png        # independent reference decodes
    tools/
    └── model-packer/
        └── README.md       # role planned at the packaging milestone
    scripts/
        ├── verify.sh       # full milestone quality gate
        ├── generate_image_goldens.py # laboratory-only image reference generator
        ├── generate_audio_goldens.py # laboratory-only audio reference generator
        └── generate_tiny_audio_fixture.py # laboratory-only tiny audio model generator

The api module contains the strict HTTP/1.1 parser and writer, the server
accept loop and routes, the bounded base64 media decoder, and the model engine
that loads text at startup and vision/audio lazily. The jobs module contains the
bounded queue, worker threads, cooperative cancellation, job snapshots, and
retained history. The media module contains the native zlib/DEFLATE decoder, the
PNG decoder, the baseline JPEG decoder, the shared decoded-image type, the
official image preprocessing pipeline, the WAV decoder, mono conversion and
resampling, and the official log-mel processor. The model module contains the
text encoder, the vision tower, the audio tower, and the image and audio
embedders that connect them to the text space. Text, image, and audio embeddings
are implemented and validated against the official reference; text, image, and
audio documents are stored, indexed, and searched through the CLI and the HTTP
API. Snapshots and compaction are implemented; see ADR 0006. The HTTP API and
jobs are implemented; see ADR 0011. HNSW and MP3 remain future work.

## Layers and dependencies

The seldon_vault_lite library is the testable root; main.rs only calls the CLI
and converts errors into exit codes. Dependencies point downward:

    cli / api  ->  jobs / model / index / storage  ->  binary / tensor / media  ->  error

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
- cli depends on api, model, index, storage, json, and error.
- api depends on jobs, model, index, storage, json, media, and error.
- jobs depends on error; it does not depend on the model or storage. The API
  supplies tasks that hold shared store and engine handles.
- api does not depend on cli.
- Parsers do not depend on inference layers.
- Storage does not depend on the HTTP API.
- The HTTP parser does not depend on the model, storage, or index.
- No module depends on a third-party crate.

## CLI contract

| Command | Effect | Output | Code |
|---|---|---|---|
| version | prints name, version, target | stdout | 0 |
| help | prints help, commands, and codes | stdout | 0 |
| serve | starts the local HTTP API and ingestion workers | JSON listening/stopped records | 0 |
| embed | loads the text, image, or audio path and prints one JSON vector | stdout | 0 |
| index | ingests a strict JSONL text/image/audio document file into the store | stdout | 0 |
| search | runs hybrid, lexical, or vector search over stored documents | stdout | 0 |
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

embed accepts --model-dir, --prompt, --dim, --text, --ids, --image, --audio,
--vision-tokens, and positional text. With no text, ids, image, or audio, it
reads standard input with a 1 MiB cap. Text is prefixed with the official prompt
before tokenization; --ids bypasses the tokenizer and the prefix for
diagnostics. Image input is decoded with the native PNG/JPEG decoders and
embedded through the official preprocessing and vision tower. Audio input is
decoded with the native WAV decoder, converted to 16 kHz mono with native
resampling, and embedded through the official log-mel processor and audio tower.
The four input families are mutually exclusive. The 30-second audio cap applies.

index requires --data-dir and one JSONL file or -. Each line is an object with
id and exactly one of text, image, or audio, plus optional title, optional
source, and optional metadata. Duplicate ids, unknown fields, malformed JSON,
oversized lines, and more than 10,000 documents are rejected before any write.
Text embeddings use the Document prompt and the frozen tokenizer. Image lines
carry an image path instead of text, are decoded with the native PNG/JPEG
decoders, and are embedded through the official preprocessing and vision tower.
Audio lines carry a WAV path, are decoded and preprocessed natively, and are
embedded through the audio tower. Image and audio chunks have empty text, so
they participate in vector search and modality filters but not BM25. When
source is omitted, the image or audio path becomes the source. A diagnostic
--ids mode accepts an ids array and uses TextModel::embed_ids instead of the
tokenizer; text is still stored for BM25. embed --image exposes the image path
directly, embed --audio exposes the audio path directly, and --vision-tokens
selects one of the official visual budgets.

search requires --data-dir and accepts --model-dir, --dim, --mode
hybrid|lexical|vector, --limit, --offset, repeatable --filter KEY=VALUE, --ids,
--image, and --audio. Lexical mode does not load the model. Vector and hybrid
modes embed the query with the SearchQuery prompt. Image and audio queries
require vector mode and cannot be combined with query text or ids. The output
is a JSON array with chunk id, document id, ordinal, score, lexical and vector
ranks, title, source, modality, text, and metadata.

verify requires --data-dir and prints records, snapshot_records, documents,
chunks, and truncated_bytes. It does not acquire the writer lock and does not
modify the snapshot or journal.

compact requires --data-dir and prints snapshot_records, snapshot_bytes,
documents, chunks, and data_dir. It opens the writer store, materializes the
current state into snapshot.sv, and replaces journal.log with an empty file.

serve requires --data-dir and accepts --bind, --model-dir, --dim,
--vision-tokens, --token, --max-connections, --request-timeout-ms,
--max-body-bytes, --queue-capacity, and --job-workers. It validates the
configuration, loads the text model, opens the store, binds the listener, prints
one JSON listening record, and serves until a loopback POST /v1/shutdown. No
stub returns a fake result.

## HTTP API

The M6 service is documented in docs/api.md and ADR 0011. It binds to
127.0.0.1:8080 by default. A non-loopback bind requires an access token of at
least 16 bytes. When a token is set, every request must present a bearer token
or X-API-Key; the comparison is constant-time. There is no TLS implementation,
so the service must not be exposed directly to the Internet.

The server accepts HTTP/1.1 only, one request per connection, with a strict
request-line and header parser. Request-line, header-line, total-header, header
count, and body caps are checked before allocation. Transfer-Encoding,
Expect, absolute request targets, fragments, obsolete line folding, bare LF,
duplicate Host/Content-Length/Authorization/Expect headers, and unknown methods
are rejected. POST requires application/json; unknown JSON fields are rejected.

Routes:

- GET /health: process liveness;
- GET /ready: storage and text-model readiness, declared and loaded modalities;
- GET /v1/models: model id, revision, dimensions, modalities, and limits;
- GET /v1/stats: request, connection, job, storage, and recovery counters;
- POST /v1/embeddings: text, token-id, image, or audio embedding;
- POST /v1/documents: validates and enqueues asynchronous ingestion;
- DELETE /v1/documents/{id}: idempotent document deletion;
- POST /v1/search: lexical, vector, or hybrid search with modality and metadata
  filters;
- GET /v1/jobs/{id}: ingestion status;
- DELETE /v1/jobs/{id}: cooperative cancellation;
- POST /v1/shutdown: loopback-only graceful shutdown.

Connections and job workers are bounded. A full connection limit returns 503
after draining the request; a full ingestion queue returns 429. Ingestion tasks
check a cancellation flag before and after the embedding phase and before the
storage write. Searches open a ReadOnlyStore view and rebuild the in-memory
index without holding the writer lock. Writes commit one atomic transaction
through a Mutex<Store>.

The text model is loaded at startup. The tokenizer is optional: a model without
tokenizer.json still serves explicit token-id embeddings and returns 501 for
text input. Vision and audio towers are loaded lazily on first use, behind a
mutex; their configurations are validated at startup so /v1/models does not
claim an undeclared modality. The tiny audio fixture has no tokenizer; its
AudioEmbedder accepts the absent tokenizer for audio-only use and returns 501
for text embedding through that embedder.

Media is accepted only as strict bounded base64 in the JSON body, never as a
client-supplied path. PNG versus JPEG is detected from the decoded bytes. The
native decoders apply their own caps before preprocessing or inference.

## Error model

src/error.rs defines Error and Result<T>. The variants are:

- Usage (exit code 2);
- Io (exit code 1);
- Corrupt (exit code 1);
- LimitExceeded (exit code 1);
- Unsupported (exit code 1);
- Internal (exit code 1).

Display produces a user-facing message; source() exposes the technical cause
for I/O errors. The HTTP API reuses the same taxonomy: Usage -> 400, Corrupt ->
400, LimitExceeded -> 413, Unsupported -> 501, Io/Internal -> 500, with 503 for
a saturated connection limit.

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
window. M7 adds src/tensor/neon.rs: runtime-selected NEON kernels for dot
products, elementwise operations, matmul, linear projections, RMSNorm, L2
normalization, mean pooling, and attention scores. Every NEON wrapper checks
`is_aarch64_feature_detected!("neon")`, falls back to the scalar path when the
feature is unavailable, and is covered by scalar-vs-NEON differential tests
including tail lengths and real model shapes. The scalar path remains the
correctness oracle.

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
samples and one, three, or four channels.

src/media/preprocess.rs implements the official Gemma4 image preprocessing:
grayscale replication, RGBA compositing over white with half-to-even rounding,
the aspect-ratio preserving target size, antialiased bicubic resize with the
torchvision uint8 fixed-point path, rescaling to [0, 1], 16x16 patchification,
and padded (x, y) positions. It is bit-exact with the reference tensor path on
the committed fixtures. Caps are 16,384 per side, 16,777,216 input pixels,
10,080 patches, 8,192 resize taps, and 7,741,440 padded FP32 values.

src/model/vision.rs implements the Gemma 4 vision tower: patch projection,
learned two-axis positions, 16 bidirectional layers with axial 2-D RoPE,
grouped-query attention, RMSNorm variants, gated GELU feed-forward, 3x3
pooling, and the 768 to 512 projection. Only real patches are processed;
padding patches are masked out in the reference and discarded by the pooler.
src/model/image.rs builds the image-only prompt and runs the text encoder over
the projected soft tokens.

Image embeddings are validated against the official reference. Image documents
are stored with modality image, indexed from the CLI, and retrieved by text or
image vector queries. Text-to-image and image-to-image ranking tests pass on a
two-image reference corpus. Broader ranking quality and vision performance
remain future work. See ADR 0007, ADR 0008, and ADR 0009.

src/media/wav.rs implements a strict RIFF/WAVE decoder for PCM with 8, 16, 24,
or 32 bits per sample and IEEE float with 32 or 64 bits per sample, including
WAVE_FORMAT_EXTENSIBLE when its sub-format is PCM or float. It validates the
RIFF size, every chunk boundary and padding byte, the format fields, block
alignment, byte rate, data size, frame count, channel count, and sample rate
before reading. It rejects compressed codecs, 12-bit PCM, unsupported float
widths, non-finite samples, truncated chunks, and inconsistent headers. Caps
are 64 MiB per file, 64 MiB of data, eight channels, 8 kHz to 192 kHz, and
5,760,000 source frames (30 seconds at 192 kHz). Channels are downmixed by
averaging and returned as FP32 mono at the source rate.

src/media/audio.rs converts mono audio to the 16 kHz model rate with a
normalized windowed-sinc kernel (32 taps, Hann window, cutoff
`min(1, target/source)`). It truncates to 480,000 samples, matching the official
30-second feature-extractor defaults, and rejects non-finite samples and
out-of-range rates. The conversion is deterministic and uses only `std`.

src/media/audio_preprocess.rs reproduces the official Gemma 4 audio feature
extractor: 320-sample frames, 160-sample hop, 512-point FFT, periodic Hann
window, 160-sample semicausal left padding, 128 HTK mel filters from 0 to
8000 Hz with floor 0.001, natural logarithm, right-padding to a multiple of 128
samples, and a frame mask that marks only complete analysis windows. Invalid
frames are exactly zero. FFT and mel projection use f64 accumulation and round
to f32 once. The committed audio preprocessing golden compares frame counts,
mask counts, and first feature values; the path is not claimed bit-exact
because a different radix-2 FFT can differ in the last f32 bit. The observed
maximum absolute difference on the short reference clip is 4.77e-7.

Audio embeddings are validated against the official reference on six clips:
silence, a short tone, a long tone, French speech, a music chord progression,
and rain-like noise. Audio documents are stored with modality audio, indexed
from the CLI, and retrieved by text or audio vector queries. Text-to-audio
ranking tests pass on the six-clip corpus. MP3, streaming audio, and broader
audio ranking quality remain future work. See ADR 0010.

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
and non-finite values, and converts BF16 or F32 storage to FP32 once.

model::vision::VisionWeights reads the same file and validates the vision
inventory: the patch projection, the two-axis position table, four norms and
three linears per layer, query/key norms, and the 768 to 512 embedding
projection. Unexpected vision tensors, missing tensors, shape mismatches, and
non-finite values are rejected. The image embedder reads the Safetensors file
once and shares the bytes between both loaders.

model::audio::AudioWeights reads the same file and validates the audio
inventory: two subsampling convolutions and norms, the subsampling projection,
two feed-forward branches, the chunked attention projections and per-layer
relative-key projection, per-dimension scales, the causal depthwise light
convolution, layer norms, the 1024 to 1536 output projection with bias, and the
1536 to 512 embedding projection. Every clipped linear also carries its scalar
input/output bounds. Unexpected audio tensors, missing tensors, shape
mismatches, inverted bounds, and non-finite values are rejected. The audio
embedder reads the Safetensors file once and shares the bytes between the text
and audio loaders.

### Vision inference

model::vision::VisionModel reproduces the frozen gemma4_vision configuration:
patch size 16, pooling 3, hidden 768, 16 layers, 12 heads of dimension 64,
intermediate size 3,072, axial RoPE theta 100, RMSNorm epsilon 1e-6, and GELU
tanh. The tower rejects clipped linears, feature standardization, non-axial
RoPE, FP16, attention bias, and non-zero dropout instead of approximating
them.

model::image::ImageEmbedder validates the special token ids against the
tokenizer and runs the image-only prompt <bos><|image><|image|>*soft<image|><eos>.
The vision soft tokens replace the placeholder embeddings, PLE is computed
from the resulting inputs_embeds, and mean pooling, projection, normalization,
and Matryoshka truncation follow the text path.

model::audio::AudioModel reproduces the frozen gemma4_audio configuration:
hidden 1024, 12 layers, eight heads of dimension 128, convolution kernel 5,
subsampling channels [128, 32], chunk size 12, left context 13, right context 0,
logit cap 50, RMSNorm epsilon 1e-6, output projection 1536, residual weight 0.5,
gradient clipping 1e10, SiLU, and clipped linears. It rejects FP16, a non-SiLU
activation, a non-clipped linear configuration, inconsistent head layouts, and
missing bounds instead of approximating them. The tower applies two stride-2
convolution/LayerNorm/ReLU blocks with mask striding, sinusoidal relative
positions projected per layer, chunked local attention with per-head relative
bias and soft-capped scores, causal depthwise light convolution, two
feed-forward branches, and the output projection. Padding queries are skipped;
the tower is causal along time, so trailing padding cannot influence a valid
query. Captured layer probes select valid rows only.

model::audio_embedder::AudioEmbedder validates the audio special token ids
against the tokenizer and runs the audio-only prompt
<bos><|audio><|audio|>*soft<audio|><eos>. The projected audio soft tokens
replace the placeholder embeddings, PLE is computed from the resulting
inputs_embeds, and mean pooling, projection, normalization, and Matryoshka
truncation follow the text path. The embedder converts the decoded WAV to
16 kHz mono, computes the official log-mel features, runs the tower, and builds
the prompt. No transcription or file-name fallback is used.

The source SHA-256 is retained for manifest comparison. The FP32 text, vision,
and audio weight sets are each capped at 2^30 elements.

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
PNGs.

The preprocessing tests compare resized bytes, padded patch values, and
positions against SHA-256 fingerprints from the official tensor path for
grayscale, RGB, RGBA, gradient, and quadrant fixtures. The image embedding test
loads the frozen model when the cache is present and checks the patch embedder,
all 16 vision layer probes, the projected soft tokens, the final 768-dimensional
embedding (cosine and absolute tolerance), and Matryoshka truncation. The full
seven-case image corpus is an ignored test because one scalar vision pass is
expensive. The media fixtures and their provenance are recorded in
fixtures/media/README.md.

M5 adds WAV decoding tests for PCM 8/16/24/32, IEEE float 32, stereo downmix,
truncated and malformed headers, unsupported codecs, non-finite samples, and
resampling behavior. The audio preprocessing test compares frame counts, mask
counts, and first log-mel values against the official feature extractor for six
clips. The audio embedding test loads the frozen model when the cache is present
and checks the SSCP output, all 12 encoder layer probes, the projected audio
features, the final 768-dimensional embedding (cosine and absolute tolerance),
and Matryoshka truncation. The full six-case audio corpus and the text-to-audio
ranking test are ignored tests because they load the frozen model and embed
several clips. The short-tone parity case runs by default. Audio fixtures and
their provenance are recorded in fixtures/media/README.md and
fixtures/model/README.md.

M6 adds unit tests for the strict HTTP parser and writer, base64 decoding, job
queue lifecycle, cancellation, pending cap, configuration validation, route
tables, token parsing, and constant-time token comparison. The HTTP integration
suite starts the real server on an ephemeral loopback port and speaks raw
HTTP/1.1 over std::net::TcpStream. It covers health, readiness, models, stats,
token embeddings, audio embeddings and audio document ingestion on the tiny
fixture, lexical/vector/hybrid search, metadata filters, Unicode JSON, percent-encoded
deletion, duplicate-id job failure, bearer authentication, malformed requests,
oversized bodies, transfer encoding, Expect, connection limits, slow-request
timeouts, concurrent health requests, and graceful shutdown. Full-model text and
image HTTP embedding tests are ignored because they load the frozen model; the
existing text/image parity suites cover the numerical path.

scripts/verify.sh groups these commands and fails if the Cargo graph contains
a dependency or if minos is not 11.0.

## Next steps

1. **M3**: journal, snapshots, safe compaction, read-only views, BM25, exact
   vector search, and RRF are implemented. HNSW remains deferred until volume
   and latency measurements justify it.
2. **M4**: complete. Native PNG and baseline JPEG decoding, official image
   preprocessing, the vision tower, image embeddings, image storage and CLI
   indexing, and text-to-image and image-to-image vector search are implemented
   and validated against the official reference. Rotation and aspect-ratio
   fixtures cover preprocessing and vision parity. Broader ranking quality and
   vision performance remain future work.
3. **M5**: complete on branch feat/m5-audio-embeddings. Native WAV decoding,
   mono conversion and resampling, official log-mel preprocessing, the audio
   tower, audio embeddings, audio storage and CLI indexing, and text-to-audio
   and audio-to-audio vector search are implemented and validated against the
   official reference. MP3, streaming audio, and broader audio ranking quality
   remain future work.
4. **M6**: implemented on branch feat/m6-http-api. Native HTTP/1.1 API, strict
   parser and bounded base64 media input, bearer authentication for LAN binds,
   bounded connections, asynchronous ingestion jobs with cooperative
   cancellation, observability counters, and loopback graceful shutdown.
   HNSW, TLS, keep-alive, and MP3 remain future work.
5. **M7**: in progress on branch feat/m7-neon-simd-packaging. NEON kernels
   with runtime dispatch, scalar fallback, differential tests, and frozen-model
   parity replay are implemented. The versioned model container, Rust std
   packer, embedded loading path, and Mach-O size/startup verification remain
   in progress.
6. **M8/M9**: final macOS validation, then Linux port.

None of these steps may be declared complete without reproducible tests and
without replaying the golden tests.
