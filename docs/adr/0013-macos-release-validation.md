# ADR 0013 — macOS 1.0 release validation and distribution

- **Status:** accepted
- **Date:** 2026-11-11
- **Milestone:** M8
- **Supersedes:** none

## Context

M0–M7 produced a tested engine, a persistent hybrid store, native text, image,
and audio embeddings, a local HTTP API, runtime-selected NEON kernels, and a
verified single-file embedded model bundle. The remaining macOS release gap is
not another numerical feature: it is the evidence that the artifact works on a
clean Apple Silicon Mac, without the development tree, model cache, Homebrew,
Python, or an installed inference tool, and that the final documentation tells
an operator exactly what is supported and what is not.

The M7 artifact is ad-hoc linker-signed. Local execution works, but Gatekeeper
rejects the binary when it is copied to another Mac. Distribution therefore
requires a Developer ID Application signature and notarization, which require a
developer account. That account must not become a runtime dependency of the
product or a prerequisite for local use.

A successful compilation and a passing test suite on the development machine
are necessary, but they are not proof that the release is clean-machine ready.
The release gate must be an explicit protocol with recorded evidence.

## Decision

### 1. The release gate is a clean-machine protocol

A macOS 1.0 release candidate is accepted only after the end-to-end lifecycle
has been executed on a clean Apple Silicon Mac that has never built the project.
The machine must have:

- macOS 11 or later on Apple Silicon (aarch64);
- no Homebrew, Python, Cargo, or inference engine used by the validation;
- no network connection for the CLI validation;
- no model directory and no `.model-cache`;
- only the release binary, the committed media fixtures, and the validation
  harness.

The automated harness is `scripts/release_e2e.sh`. It stages the binary and
fixtures in a temporary install root, runs the binary with `env -i` and a
system-only `PATH`, a temporary `HOME` and `TMPDIR`, and no model cache. It
checks text, image, and audio embeddings; text, image, and audio ingestion;
lexical, vector, and hybrid search; verification; compaction; restart
persistence; the loopback HTTP health, ready, models, embeddings, document
ingestion, job, search, deletion, and shutdown routes; and a second HTTP restart
that proves persisted documents survive. It also checks the Mach-O linkage,
deployment target, code signature, Gatekeeper assessment, and the zero-crate
Cargo graph when Cargo is available.

When `sandbox-exec` is usable, the CLI path runs with network access denied.
The HTTP path uses loopback only. The harness records the network-isolation
mode in its JSON report; a clean-machine run must not claim offline validation
when isolation was unavailable.

The harness is necessary but not sufficient. The physical clean-Mac run is a
separate gate because the development machine has the source tree, the model
cache, and developer tooling that can hide missing-file assumptions.

### 2. Release artifact

The macOS artifact is one `aarch64-apple-darwin` Mach-O binary built with the
`embedded-model` feature and the verified frozen `.svlm` bundle. The model is
inside the executable; no external model directory is required. Only user data,
logs, and snapshots are external persistent files.

The deployment floor is macOS 11.0, enforced by `.cargo/config.toml` and
checked with `vtool`. The binary must link only `/usr/lib/libSystem.B.dylib`;
no third-party dynamic application library, no FFI, and no Cargo dependency are
allowed. The binary is not built for `x86_64-apple-darwin`.

### 3. Signing and distribution

Two signing modes are distinguished:

- **Local-use build:** ad-hoc linker signature. It works on the machine that
  built it. Gatekeeper may reject it when it is copied to another Mac. It must
  be labeled local-use only and must not be presented as a general macOS 1.0
  distribution.
- **Distributed build:** Developer ID Application signature with the hardened
  runtime, a secure timestamp, notarization through `notarytool`, and a stapled
  ticket. `codesign --verify --strict` and `spctl --assess --type execute`
  must both pass on a clean Mac before distribution.

No developer account is required for local use. A release candidate that is not
notarized is not a general distribution; the release notes and the README must
say so. Bypassing Gatekeeper by removing quarantine attributes is not a
supported distribution procedure.

### 4. Version and claim discipline

The crate remains at `0.1.0` while the clean-machine gate and the signing
decision are open. The release commit that passes both gates will bump the
version to `1.0.0` and tag `v1.0.0`. A successful development-machine run does
not justify a 1.0 claim by itself.

No modality is declared supported because a vector was produced from a file
name or a textual marker. The text, image, and audio paths in this release are
the paths already validated against the official reference fixtures. MP3,
streaming audio, TLS, HTTP keep-alive, HNSW, GPU, transcription, and generative
output remain unsupported.

### 5. Final documentation

`docs/release.md` is the operator-facing macOS 1.0 guide: supported platform,
install, offline behavior, data lifecycle, backup and recovery, local API
security, signing and Gatekeeper, performance expectations, known limitations,
and the release checklist. The README, AGENTS.md, architecture, parity,
benchmarks, and API documents are updated to the same state. The release notes
must state the exact artifact, model revision, macOS version, hardware, and
measured results. No result is fabricated or inferred from an unrelated run.

### 6. Data safety and compatibility

The data directory contains the journal, snapshot, and writer locks. It is
external user data, not part of the binary. Operators must back it up before an
upgrade, run `verify` after an interruption, and use `compact` to materialize
a snapshot. The writer lock prevents two writers; read-only views do not take
the writer lock. Recovery replays the committed journal prefix and truncates an
uncommitted tail. The storage schema version is explicit; a future schema change
requires a documented migration, not silent in-place reinterpretation.

## Alternatives considered

- **Declare 1.0 after the development test suite:** rejected. It does not prove
  that the binary runs without the source tree, the model cache, or developer
  tooling.
- **Require Homebrew, Python, or an inference server on the target Mac:**
  rejected. It violates the zero-dependency and zero-external-engine contract.
- **Ship the model as an external directory:** rejected for the release
  artifact. It breaks the single-file goal; directory loading remains a
  development path.
- **Treat ad-hoc signing as sufficient for distribution:** rejected. Gatekeeper
  rejects it on other Macs, and bypassing quarantine is not supported.
- **Notarize with an ad-hoc signature:** impossible. Notarization requires a
  Developer ID identity.
- **Run the harness only on the development machine and call it clean:** rejected.
  The harness reduces manual error, but the physical clean-Mac gate remains.

## Consequences

- M8 has a repeatable automated harness and a documented manual gate. The
  harness can be run on the development machine for early feedback, but its
  result is not a substitute for the clean-Mac run.
- The local binary remains usable offline without an account. Distribution to
  other Macs requires credentials and notarization.
- The version stays at 0.1.0 until both gates pass. The 1.0.0 claim is tied to a
  specific artifact and recorded evidence.
- The release documentation must be updated whenever the artifact, model
  revision, signing mode, or supported platform changes.
- No runtime behavior, numerical tolerance, or third-party dependency is
  changed by this ADR.

## Verification

- `cargo fmt --check`
- `cargo clippy --all-targets -- -D warnings`
- `SELDON_SKIP_MODEL_PARITY=1 cargo test`
- `cargo test --release --test text_embeddings real_model_parity -- --nocapture`
- `cargo test --release --test image_embeddings image_embedding_matches -- --nocapture`
- `cargo test --release --test audio_embeddings audio_embedding_matches_the_official_reference -- --nocapture`
- `cargo build --release --target aarch64-apple-darwin`
- `SELDON_MODEL_BUNDLE=<verified .svlm> cargo build --release --features embedded-model --target aarch64-apple-darwin`
- `scripts/release_e2e.sh --binary <embedded binary>`
- `cargo tree --edges normal,build,dev` shows only the root crate.
- `otool -L` shows only `/usr/lib/libSystem.B.dylib`.
- `vtool -show-build` reports `minos 11.0`.
- `codesign --verify --strict` passes; `spctl --assess --type execute` is
  recorded (rejected for ad-hoc, accepted for a notarized Developer ID build).
- The physical clean-Mac run records the macOS version, hardware, date,
  artifact SHA-256, harness output, and network-isolation mode.

## References

- `scripts/release_e2e.sh` — automated clean-environment lifecycle harness.
- `docs/release.md` — operator-facing macOS 1.0 release guide.
- `docs/adr/0012-neon-simd-and-packaging.md` — bundle format and embedded build.
- `docs/benchmarks.md` — measured startup, memory, and embedding performance.
- `seldon-vault-lite-plan.md` sections 12 and 13 — packaging and release tests.
