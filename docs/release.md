# macOS 1.0 release and operations guide

**Status:** milestone M8 is in progress on branch `feat/m8-release-validation`.
This guide is the operator-facing contract for the macOS Apple Silicon release.
The crate version remains `0.1.0` until the clean-machine gate and the signing
decision described below are complete; only then is the release version bumped
to `1.0.0` and tagged `v1.0.0`.

## Supported platform

- **CPU:** Apple Silicon (arm64). The Intel macOS target
  (`x86_64-apple-darwin`) is explicitly excluded and is not validated.
- **OS:** macOS 11.0 or later. The Mach-O deployment target is `minos 11.0`;
  it is checked with `vtool`.
- **Runtime:** no third-party crate, no FFI, no external inference engine, no
  external database, no network download, and no developer account for local
  use.
- **Build:** Rust stable, minimum 1.89. The delivered artifact is built for
  `aarch64-apple-darwin` with the `embedded-model` feature.

Linux is milestone M9. MP3, streaming audio, TLS, HTTP keep-alive, HNSW, GPU,
transcription, and generative output are not part of this release.

## The artifact

The release artifact is one Mach-O executable:

    target/aarch64-apple-darwin/release/seldon-vault-lite

for a development build, or the embedded build:

    SELDON_MODEL_BUNDLE="$PWD/.model-cache/f9972ac22e696a105c482202646979e33e4b5a1e/model.svlm" \
      CARGO_TARGET_DIR=target/embedded-full \
      cargo build --release --features embedded-model --target aarch64-apple-darwin

The embedded binary contains the verified frozen model bundle. It does not read
`.model-cache`, a model directory, or any download at runtime. The development
directory build remains useful for fast iteration but is not the release
artifact.

Expected properties of the embedded artifact:

| Property | Expected value | Check |
|---|---|---|
| Architecture | arm64 | `file` or `lipo -info` |
| Linked libraries | only `/usr/lib/libSystem.B.dylib` | `otool -L` |
| Deployment target | `minos 11.0` | `vtool -show-build` |
| Cargo graph | only the root crate | `cargo tree --edges normal,build,dev` |
| Model source | embedded `.svlm` bundle | `scripts/release_e2e.sh` probe |
| Network at runtime | none for the CLI; loopback for the local API | harness report |
| External files | user data, logs, snapshots only | data directory |

The release notes must record the artifact SHA-256, size, model revision, macOS
version, hardware, and measured results.

## Install and first run

Copy the executable to a local directory and run it. No installer, package
manager, Homebrew, Python, or inference tool is required.

    cp seldon-vault-lite /usr/local/bin/seldon-vault-lite
    seldon-vault-lite version
    seldon-vault-lite help

The version line is:

    seldon-vault-lite 0.1.0 (aarch64-apple-darwin)

The release version becomes `1.0.0` only after the gates in this document
pass.

### Quick checks

    seldon-vault-lite embed --text "The northern lights are caused by solar wind." --dim 768
    seldon-vault-lite embed --image fixtures/media/quadrants_16x16.png --dim 768
    seldon-vault-lite embed --audio fixtures/media/audio/short_tone_100ms.wav --dim 768

Each command prints a normalized JSON vector. The text, image, and audio vectors
share the same 768-dimensional embedding space; Matryoshka dimensions 512, 256,
and 128 are supported and renormalized after truncation.

The scalar vision tower is deliberately a correctness baseline, not a fast
path. An image embedding can take tens of seconds on the reference machine.
Audio is faster for short clips. The NEON path accelerates the text encoder and
the shared kernels; the image and audio towers still spend most of their time
in the scalar path.

## Offline behavior

The product requires no network at runtime:

- the model is embedded in the executable;
- the tokenizer and processors are in the bundle;
- no telemetry, license check, or update check exists;
- no auxiliary process is launched;
- the HTTP API binds to loopback by default.

The automated harness runs the CLI path under `sandbox-exec` with network
denied when that tool is usable. The HTTP path is loopback-only by design. A
clean-machine run must record the network-isolation mode; do not claim offline
validation when isolation was unavailable.

To validate offline manually, disconnect the Mac from all networks and run the
text, image, and audio commands above. They must not fail, hang on a download,
or write outside the chosen data directory.

## Data directory and lifecycle

Indexed documents, chunks, vectors, and metadata live in the directory passed
to `--data-dir`. The binary does not write into the application directory or
the model cache.

Typical layout:

    data/
    ├── journal.log       # append-only framed records with CRC-32
    ├── snapshot.sv       # versioned snapshot after compaction
    ├── writer.lock       # exclusive writer lock
    └── compaction.lock   # compaction lock

The writer lock prevents a second writer. Read-only views do not take the
writer lock. The journal is replayed deterministically on open; an uncommitted
torn tail is truncated. Snapshot and journal corruption is reported as a typed
error rather than silently repaired.

### Backup and restore

1. Stop the server or ensure no writer is active.
2. Copy the entire data directory, including `journal.log` and `snapshot.sv`.
3. Restore by copying the directory back and running `verify`.
4. Do not edit, truncate, or reorder the journal or snapshot by hand.

    seldon-vault-lite verify --data-dir ./data

The verify command is read-only and reports records, snapshot records, documents,
chunks, and any uncommitted trailing bytes.

### Compaction

Compaction materializes the current state into a snapshot and replaces the
journal with an empty file:

    seldon-vault-lite compact --data-dir ./data

Run `verify` after compaction and after any interruption. The crash-injection
tests cover every journal byte, every snapshot temporary-file byte, every
compaction journal temporary-file byte, and truncated snapshot files.

### Recovery after an interruption

Open the store normally; it replays the committed prefix and truncates an
uncommitted tail. If `verify` reports corruption, do not delete the directory
until a backup has been made. The product does not silently discard a corrupt
committed prefix.

### Deletion

The CLI and HTTP API delete a document and its chunks atomically. The HTTP
delete is idempotent: deleting a missing document returns `"deleted":false`.

## Local HTTP API

Start the service on loopback:

    seldon-vault-lite serve --data-dir ./data

The default bind is `127.0.0.1:8080`. The service prints one JSON listening
record and serves until a loopback `POST /v1/shutdown`.

    curl -s http://127.0.0.1:8080/health
    curl -s http://127.0.0.1:8080/v1/models
    curl -s -X POST http://127.0.0.1:8080/v1/shutdown \
      -H 'Content-Type: application/json'

Security rules:

- Bind to loopback by default.
- A non-loopback bind requires an access token of at least 16 bytes.
- With a token configured, every request must present a bearer token or
  `X-API-Key`; the comparison is constant-time.
- There is no TLS, no proxy trust, no CORS policy, and no cookie session. Do not
  expose the service directly to the Internet. For a trusted LAN, use a token
  and a firewall or an external TLS-terminating proxy.
- Media is accepted only as bounded base64 JSON, never as a client-supplied
  local path.
- Connections, request bodies, headers, media, token lists, ingestion queues,
  and workers are bounded. See `docs/api.md` for the exact limits and errors.

The HTTP API and job lifecycle are documented in `docs/api.md` and ADR 0011.

## Signing, Gatekeeper, and distribution

### Local-use build

The embedded release is ad-hoc linker-signed:

    codesign -dv --verbose=4 seldon-vault-lite
    # Signature=adhoc

Local execution works without a developer account. If the binary is copied to
another Mac, Gatekeeper may reject it. An ad-hoc build is a **local-use build**,
not a general macOS 1.0 distribution. Do not instruct users to strip quarantine
attributes as a substitute for signing.

### Distributed build

A distributed macOS 1.0 build must be signed with a Developer ID Application
identity, use the hardened runtime, and be notarized and stapled:

    codesign --force --options runtime --timestamp \
      --sign "Developer ID Application: <Team Name> (<TEAMID>)" \
      seldon-vault-lite
    ditto -c -k --keepParent seldon-vault-lite seldon-vault-lite.zip
    xcrun notarytool submit seldon-vault-lite.zip \
      --keychain-profile "<profile>" --wait
    xcrun stapler staple seldon-vault-lite
    codesign --verify --strict --verbose=2 seldon-vault-lite
    spctl --assess --type execute --verbose=4 seldon-vault-lite

The last two commands must pass on a clean Mac before distribution. The exact
team identity and notarytool profile are release-environment inputs; they are
never committed and never required by the product at runtime.

If notarization is not available, publish the artifact explicitly as a
local-use build and do not claim general distribution.

## Performance and resources

The M7 measurements on the reference Apple M2 Pro are recorded in
`docs/benchmarks.md`. Summary for the full embedded artifact:

| Metric | Value |
|---|---|
| Mach-O size | about 1.4 GiB (1.5 GB bundle plus code) |
| Linked libraries | only libSystem |
| Deployment target | minos 11.0 |
| Cold startup | 18.68 s |
| Warm startup | 6.75 s |
| Maximum RSS | about 2.44 GiB |
| Peak footprint | about 1.02 GiB |
| Text p50 (M7 NEON) | 159 ms |
| Concurrent text throughput | 24.47 requests/s |

These are measurements of one machine and one frozen revision, not a promise
for every Mac. The image and audio towers are scalar correctness baselines;
their latency is higher. Record the actual macOS version, hardware, compiler,
model revision, and parameters with any new measurement.

## Known limitations

- No Intel macOS target and no universal binary.
- No Linux artifact in this release (M9).
- No MP3, streaming audio, or transcription.
- No TLS, HTTP keep-alive, HTTP/2, or Internet-facing deployment.
- No HNSW approximate index; exact cosine search is the correctness oracle.
- No GPU acceleration, Metal, Accelerate, or external inference engine.
- Broader ranking quality is not claimed; only the validated reference cases
  and the two- and six-item ranking smoke tests are covered.
- The scalar vision and audio towers are slow compared with the NEON text path.
- An ad-hoc build is local-use only; general distribution requires Developer ID
  signing and notarization.
- The storage format is versioned; a future schema change requires a documented
  migration.

## Clean-machine validation protocol

This is the release gate. Run it on a clean Apple Silicon Mac that has never
built the project. The machine must not use Homebrew, Python, Cargo, or an
inference engine for the validation.

### Prepare

1. Build the embedded release artifact on a development machine:

       SELDON_MODEL_BUNDLE="$PWD/.model-cache/f9972ac22e696a105c482202646979e33e4b5a1e/model.svlm" \
         CARGO_TARGET_DIR=target/embedded-full \
         cargo build --release --features embedded-model --target aarch64-apple-darwin

2. Record the artifact SHA-256 and size.
3. Copy the binary, `scripts/release_e2e.sh`, and `fixtures/media/` to the
   clean Mac over offline media.
4. Verify the SHA-256 on the clean Mac.
5. Disconnect the clean Mac from all networks.

### Run the automated harness

    chmod +x scripts/release_e2e.sh
    scripts/release_e2e.sh \
      --binary ./seldon-vault-lite \
      --workdir ./release-validation \
      --report ./release-validation-report.json

The harness stages the binary and fixtures, runs with a sanitized environment
and a system-only `PATH`, and exercises:

- version and help;
- embedded-model probe;
- text embedding (twice, to check determinism), image embedding, audio
  embedding;
- text, image, and audio document ingestion;
- lexical, vector text, vector image, vector audio, and hybrid search;
- verify, compact, verify after compaction, and search after restart;
- HTTP health, ready, models, embeddings, document ingestion, job polling,
  hybrid search, deletion, shutdown, restart, and persisted-document search;
- linked libraries, deployment target, code signature, Gatekeeper assessment,
  and the zero-crate Cargo graph when Cargo is available.

It writes a JSON report with the timestamp, platform, OS version, binary
SHA-256 and size, version, signing mode, Gatekeeper result, network-isolation
mode, dimension, and document count.

### Manual checklist

Use the manual checklist even when the harness passes. Record the result for
each item.

| # | Check | Expected |
|---|---|---|
| 1 | `version` | name, version, `aarch64-apple-darwin` |
| 2 | `help` | usage, commands, exit codes |
| 3 | Text embed | 768 values, unit norm |
| 4 | Image embed | 768 values, unit norm |
| 5 | Audio embed | 768 values, unit norm |
| 6 | Index text, image, audio | documents 3, chunks 3 |
| 7 | Lexical search | text document found |
| 8 | Vector text search | text document first |
| 9 | Vector image search | image document first |
| 10 | Vector audio search | audio document first |
| 11 | Hybrid search | text document found |
| 12 | `verify` | documents 3, chunks 3, no truncated bytes |
| 13 | `compact` | snapshot written, documents 3 |
| 14 | Restart search | documents still found |
| 15 | HTTP health and ready | `ok`, `ready:true` |
| 16 | HTTP models | 768 dimensions, text/image/audio declared |
| 17 | HTTP embeddings | normalized vector |
| 18 | HTTP ingestion job | succeeds |
| 19 | HTTP search | ingested document found |
| 20 | HTTP delete | `deleted:true` |
| 21 | HTTP shutdown | 202, clean stopped record |
| 22 | HTTP restart | persisted documents found |
| 23 | `otool -L` | only libSystem |
| 24 | `vtool -show-build` | `minos 11.0` |
| 25 | `codesign --verify --strict` | passes |
| 26 | `spctl --assess` | recorded; accepted only for notarized distribution |
| 27 | Offline CLI | passes with network disabled |
| 28 | No model cache | no `.model-cache` created |

### Record

Archive the following with the release candidate:

- macOS version, hardware model, and date;
- artifact SHA-256 and byte size;
- model revision;
- harness JSON report;
- manual checklist with pass/fail and operator;
- network-isolation mode;
- signing mode and Gatekeeper assessment;
- any deviation, with a written justification.

A development-machine run is useful early feedback but does not satisfy the
physical clean-Mac gate.

## Release checklist

- [ ] `cargo fmt --check` passes.
- [ ] `cargo clippy --all-targets -- -D warnings` passes.
- [ ] `SELDON_SKIP_MODEL_PARITY=1 cargo test` passes.
- [ ] Frozen text, image, and audio parity suites pass.
- [ ] The model bundle verifies (header CRC-32, catalog digest, section SHA-256).
- [ ] The embedded `aarch64-apple-darwin` build succeeds.
- [ ] `scripts/release_e2e.sh` passes on a clean Apple Silicon Mac.
- [ ] `otool -L` shows only libSystem.
- [ ] `vtool -show-build` reports `minos 11.0`.
- [ ] `cargo tree --edges normal,build,dev` shows only the root crate.
- [ ] The binary is ad-hoc for local use, or Developer ID signed and notarized
      for distribution.
- [ ] `spctl --assess` is recorded and matches the signing mode.
- [ ] README, AGENTS.md, architecture, parity, benchmarks, API, and this guide
      describe the same state.
- [ ] The version is bumped to `1.0.0` only after the gates pass.
- [ ] The release notes record the exact artifact and evidence.
- [ ] No weights, secrets, or model files are committed.

## Diagnostics and support

- `seldon-vault-lite verify --data-dir <dir>` checks storage without modifying
  it.
- `GET /v1/stats` exposes request, connection, job, storage, and recovery
  counters.
- `GET /v1/models` reports the model revision, dimensions, declared modalities,
  and limits.
- The CLI writes typed errors with exit codes: 2 for usage errors, 1 for I/O,
  corruption, limit, or execution errors.
- A closed pipe is normal Unix behavior and exits 0.

There is no telemetry and no remote support channel. Collect the version line,
the harness report, the relevant data directory metadata, and the exact command
that failed.

## References

- [Implementation plan](../seldon-vault-lite-plan.md)
- [ADR 0013 — macOS 1.0 release validation and distribution](adr/0013-macos-release-validation.md)
- [ADR 0012 — NEON kernels and macOS ARM64 packaging](adr/0012-neon-simd-and-packaging.md)
- [Architecture](architecture.md)
- [HTTP API](api.md)
- [Numerical parity](model-parity.md)
- [Benchmarks](benchmarks.md)
- [Storage](storage.md)
- [Fixtures](../fixtures/README.md)
