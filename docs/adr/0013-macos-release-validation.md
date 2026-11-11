# ADR 0013 — macOS local-use release validation

- **Status:** accepted
- **Date:** 2026-11-11
- **Amended:** 2026-11-11 — local-use only; clean-machine and signing gates
  removed by project owner decision.
- **Milestone:** M8
- **Supersedes:** none

## Context

M0–M7 produced a tested engine, a persistent hybrid store, native text, image,
and audio embeddings, a local HTTP API, runtime-selected NEON kernels, and a
verified single-file embedded model bundle. The remaining macOS release work is
not another numerical feature: it is the evidence that the artifact runs the
full text/image/audio and storage/HTTP lifecycle from the embedded bundle, and
that the final documentation tells an operator exactly what is supported and
what is not.

The project owner has decided that this release is a **local-use build**:

- no separate clean Apple Silicon Mac is required;
- no Developer ID signing or notarization is planned;
- distribution to other Macs is out of scope.

The M7 artifact is ad-hoc linker-signed. Local execution works. Gatekeeper may
reject the binary when it is copied to another Mac, which is expected for an
ad-hoc build and is recorded rather than treated as a release failure.

## Decision

### 1. Release validation runs on the development Apple Silicon machine

A macOS release candidate is accepted after the end-to-end lifecycle has been
executed on the development Apple Silicon machine with the embedded release
artifact. No separate clean-machine gate is required.

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
The HTTP path uses loopback only. The harness records the network-isolation mode
in its JSON report. When isolation is unavailable, the report says so; the
release notes must not claim an isolated offline run that was not performed.

### 2. Release artifact

The macOS artifact is one `aarch64-apple-darwin` Mach-O binary built with the
`embedded-model` feature and the verified frozen `.svlm` bundle. The model is
inside the executable; no external model directory is required. Only user data,
logs, and snapshots are external persistent files.

The deployment floor is macOS 11.0, enforced by `.cargo/config.toml` and
checked with `vtool`. The binary must link only `/usr/lib/libSystem.B.dylib`;
no third-party dynamic application library, no FFI, and no Cargo dependency are
allowed. The binary is not built for `x86_64-apple-darwin`.

### 3. Local-use signing only

The release artifact is ad-hoc linker-signed:

- `codesign --verify --strict` passes on the artifact;
- `spctl --assess --type execute` rejects it because it is not notarized;
- this rejection is expected and is recorded in the harness report.

No Developer ID Application signature, hardened-runtime distribution, secure
timestamp, notarization, or stapled ticket is planned. The artifact is a
**local-use build** and must not be presented as a general macOS distribution.
Distribution to other Macs is out of scope. Bypassing Gatekeeper by removing
quarantine attributes is not a supported procedure.

### 4. Version and claim discipline

The crate remains at `0.1.0`. This release validation is complete for the
local-use artifact, but no `1.0.0` distribution is claimed. A version bump and
tag would require a separate distribution decision; they are not part of this
milestone.

No modality is declared supported because a vector was produced from a file
name or a textual marker. The text, image, and audio paths in this release are
the paths already validated against the official reference fixtures. MP3,
streaming audio, TLS, HTTP keep-alive, HNSW, GPU, transcription, and generative
output remain unsupported.

### 5. Final documentation

`docs/release.md` is the operator-facing guide for the local-use macOS build:
supported platform, install, offline behavior, data lifecycle, backup and
recovery, local API security, Gatekeeper behavior, performance expectations,
known limitations, release validation, and the release checklist. The README,
AGENTS.md, architecture, parity, benchmarks, and API documents are updated to
the same state. The release notes must state the exact artifact, model revision,
macOS version, hardware, and measured results. No result is fabricated or
inferred from an unrelated run.

### 6. Data safety and compatibility

The data directory contains the journal, snapshot, and writer locks. It is
external user data, not part of the binary. Operators must back it up before an
upgrade, run `verify` after an interruption, and use `compact` to materialize
a snapshot. The writer lock prevents two writers; read-only views do not take
the writer lock. Recovery replays the committed journal prefix and truncates an
uncommitted tail. The storage schema version is explicit; a future schema change
requires a documented migration, not silent in-place reinterpretation.

## Alternatives considered

- **Require a separate clean Apple Silicon Mac:** rejected by the project owner.
  The sanitized harness on the development machine is the release validation.
- **Require Developer ID signing and notarization:** rejected by the project
  owner. The artifact is local-use only and distribution to other Macs is out of
  scope.
- **Claim a 1.0 distribution without notarization:** rejected. The version stays
  at 0.1.0 and no general distribution is claimed.
- **Require Homebrew, Python, or an inference server on the target Mac:**
  rejected. It violates the zero-dependency and zero-external-engine contract.
- **Ship the model as an external directory:** rejected for the release
  artifact. It breaks the single-file goal; directory loading remains a
  development path.
- **Run the harness only on the development machine and call it a distribution:**
  rejected. It is a local-use validation, not a notarized distribution.

## Consequences

- M8 has a repeatable automated harness and final documentation for the
  local-use artifact. No clean-machine gate and no signing/notarization step are
  tracked.
- The local binary remains usable offline without an account. Gatekeeper may
  reject it on another Mac; distribution to other Macs is out of scope.
- The version stays at 0.1.0. No `1.0.0` tag or distribution claim is made.
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
  recorded as rejected for the ad-hoc local-use build.
- The harness report records the macOS version, hardware, date, artifact
  SHA-256, network-isolation mode, and lifecycle results.

## References

- `scripts/release_e2e.sh` — local-use release lifecycle harness.
- `docs/release.md` — operator-facing local-use macOS release guide.
- `docs/adr/0012-neon-simd-and-packaging.md` — bundle format and embedded build.
- `docs/benchmarks.md` — measured startup, memory, and embedding performance.
- `seldon-vault-lite-plan.md` sections 12 and 13 — packaging and release tests.
