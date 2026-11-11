# ADR 0012 — NEON kernels and macOS ARM64 packaging

- **Status:** accepted for the NEON path; the packaging design is in progress
- **Date:** 2026-11-11
- **Milestone:** M7
- **Supersedes:** none

## Context

M0–M6 produced a tested engine, persistent store, hybrid index, native
text/image/audio embeddings, and a local HTTP API. The numerical reference is
the deterministic FP32 scalar path in `src/tensor/kernels.rs`. M7 must add
NEON kernels for macOS Apple Silicon and move toward a verified single-file
macOS ARM64 distribution, without a third-party crate, without FFI, and with
`unsafe` confined to justified `core::arch` SIMD kernels.

The scalar path is not fast enough for the product: the M2 text baseline was
1.485 s p50 per 768-dimensional embedding, and the M4/M5 vision and audio towers
were measured in tens of seconds. The kernels are dominated by dot products and
linear projections, which map directly to NEON.

## Decision

### 1. Runtime feature detection with a scalar fallback

Every NEON entry point is a safe wrapper. The wrapper checks a cached
`std::arch::is_aarch64_feature_detected!("neon")` result. If the feature is
not available, it calls the scalar reference instead. The scalar path is never
removed, so a processor without NEON cannot execute an illegal instruction and
the correctness oracle remains available.

The check is cached in a `OnceLock<bool>` because it sits on every dispatch
path.

### 2. `unsafe` is confined to the intrinsics

`src/tensor/neon.rs` contains:

- safe wrappers that validate or assume validated shapes and fall back to the
  scalar path;
- `#[target_feature(enable = "neon")]` kernels;
- small `load`/`store` helpers with `# Safety` contracts;
- a `// SAFETY:` comment on every unsafe block explaining the bounds or the
  feature guarantee.

The public `kernels` functions validate lengths, shapes, allocation caps, and
non-finite results before dispatch. No parser, storage, HTTP, or business module
uses `unsafe`.

### 3. Kernels covered in this step

The NEON path covers the operations that dominate the model:

| Operation | Scalar reference | NEON path |
|---|---|---|
| dot product | `dot_scalar` | four-lane accumulator with FMA |
| add, sub, mul, scale, in-place add | scalar loops | vector loops with scalar tails |
| matmul | `matmul_into_scalar` | broadcast over the output row |
| linear projection | `linear_into_scalar` | vectorized dot per output row |
| RMSNorm | `rms_norm_scalar` | vector sum of squares and weighted scale |
| row-wise RMSNorm | `rms_norm_rows_scalar` | vector sum and scale per row |
| unscaled row-wise RMSNorm | `rms_norm_rows_unscaled_scalar` | vector sum and scale per row |
| L2 normalization | `l2_normalize_scalar` | vector sum and scale |
| mean pooling | `mean_pool_scalar` | vector accumulation and scale |
| attention scores | scalar loop | `dot_raw` dispatch |

Tail elements are always handled by the scalar loop, so lengths that are not a
multiple of four are covered.

### 4. Numerical policy

NEON reductions accumulate in four independent lanes and use fused
multiply-add, so sums differ from the sequential scalar reduction by a few
units in the last place. This is acceptable only because:

- the scalar path remains the oracle;
- differential tests compare every kernel against the scalar path, including
  tail lengths and the real model shapes;
- the frozen-model parity suites are replayed after the change.

The differential tolerances are `2e-5` relative for dot, `2e-4` relative for
linear and matmul, `2e-5` for RMSNorm, and `1e-6` for L2 normalization and
mean pooling. The official text, image, and audio parity tests keep their
existing `1e-4` absolute/relative and `0.9999` cosine thresholds; no threshold
was loosened.

### 5. Packaging design (in progress)

The packaging step will use a versioned, bounded container instead of a bare
directory:

- magic `SVLMDL01`, version 1;
- a fixed header with header length, section count, total length, and a CRC-32
  of the header;
- a sorted section table with relative path, role, offset, length, and SHA-256;
- one contiguous data area;
- a 32-byte catalog digest over the header and the section digests;
- explicit caps on header size, section count, path length, section length, and
  total bundle size;
- a packer written in Rust `std` that validates the frozen manifest hashes
  before writing, and a reader that validates structure, CRC, catalog digest,
  and section hashes.

The final binary will embed the container with `include_bytes!` (or an
equivalent Rust mechanism) and load model files from the embedded bytes, with
no first-start download. The linker, startup time, binary size, and resident
memory limits will be measured on Apple Silicon before the packaging step is
declared complete. Only user data, logs, and snapshots remain external files.

The packer is a development tool; it is not part of the delivered runtime
binary. It uses only `std` and first-party code.

## Alternatives considered

- **Compile-time NEON only:** simpler, but violates the requirement to avoid
  illegal instructions on processors without the feature and removes the
  scalar oracle from the shipped binary.
- **Third-party SIMD crate:** forbidden by the zero-crate contract and adds an
  unaudited dependency.
- **Approximate reciprocal square root:** faster, but changes the numerical
  contract and would require a new tolerance review. Not used.
- **Keep the scalar path for the final projection only:** would preserve some
  accuracy but leave most of the cost untouched; the parity tests show the
  full NEON path is within tolerance.
- **Embed weights directly as Rust arrays:** bloats source files and defeats
  deterministic packaging; the container is the supported mechanism.
- **Read the model from a directory in the final distribution:** violates the
  single-executable goal; it remains the development path only.

## Consequences

- The scalar path is still compiled and tested, so the binary carries both
  paths. The dispatch branch is negligible compared with the kernel cost.
- NEON is used on every supported AArch64 macOS machine. The runtime check
  keeps the door open for future targets.
- The measured M7 text p50 is 159 ms, down from 1.485 s; concurrent throughput
  is 24.47 requests/s, up from 2.43 requests/s. The frozen text, image, and
  audio parity suites pass.
- Packaging is not complete yet. The container format, packer, embedded
  loading path, and Mach-O size/startup measurements remain open.
- No third-party crate, FFI, or `unsafe` outside the NEON intrinsics is
  introduced.

## Verification

- `cargo fmt --check`
- `cargo clippy --all-targets -- -D warnings`
- `SELDON_SKIP_MODEL_PARITY=1 cargo test`
- `cargo test --release --test text_embeddings real_model_parity -- --nocapture`
- `cargo test --release --test image_embeddings image_embedding_matches_the_official_reference -- --nocapture`
- `cargo test --release --test audio_embeddings audio_embedding_matches_the_official_reference -- --nocapture`
- `cargo test --release --lib neon_microbenchmark -- --ignored --nocapture`
- `cargo build --release --target aarch64-apple-darwin`
- `cargo tree --edges normal,build,dev` shows only the root crate.
- `otool -L` shows only `libSystem`; `vtool -show-build` reports `minos 11.0`.

## References

- `src/tensor/neon.rs` — NEON wrappers and kernels.
- `src/tensor/kernels.rs` — scalar oracle, validation, and dispatch.
- `docs/benchmarks.md` — M7 measurements.
- `seldon-vault-lite-plan.md` sections 9 and 12 — optimization and packaging.
- ADR 0001 — `unsafe` policy.
- ADR 0002 — formats and tensors.
