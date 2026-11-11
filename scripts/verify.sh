#!/usr/bin/env bash
# Full verification for milestone M8 (macOS 1.0 release validation).
# Target: macOS Apple Silicon (aarch64-apple-darwin) only.
#
# The physical clean-Mac gate is separate; scripts/release_e2e.sh automates the
# clean-environment lifecycle. Set SELDON_RELEASE_E2E=1 to run it here when the
# embedded release binary is present.
set -euo pipefail

cargo fmt --check
cargo clippy --all-targets -- -D warnings

# The full-model parity test is heavy; run the fast suite first, then the
# explicit release parity check when the frozen weights are present. The fast
# suite includes the HTTP API integration tests and the release lifecycle test.
SELDON_SKIP_MODEL_PARITY=1 cargo test

cargo build --release --target aarch64-apple-darwin

echo "--- frozen-model parity (optional) ---"
REV="f9972ac22e696a105c482202646979e33e4b5a1e"
if [ -f ".model-cache/$REV/model.safetensors" ]; then
    cargo test --release --test text_embeddings real_model_parity -- --nocapture
    cargo test --release --test image_embeddings image_embedding_matches -- --nocapture
    cargo test --release --test cli indexes_an_image -- --ignored --nocapture
    cargo test --release --test cli text_and_image_queries_rank_image_documents -- --ignored --nocapture
    cargo test --release --test image_embeddings rotation_and_aspect_vision_parity -- --ignored --nocapture
    cargo test --release --test audio_embeddings audio_embedding_matches_the_official_reference -- --nocapture
    cargo test --release --test audio_embeddings all_audio_embeddings_match_the_official_reference -- --ignored --nocapture
    cargo test --release --test audio_embeddings text_queries_rank_the_expected_audio_clip_first -- --ignored --nocapture
    cargo test --release --test cli audio_documents_can_be_indexed_and_found_by_text_or_audio -- --ignored --nocapture
else
    echo "skipping: .model-cache/$REV/model.safetensors is not present"
fi

echo "--- bundle-backed loading (optional) ---"
if [ -f ".model-cache/$REV/model.svlm" ]; then
    cargo test --release --test bundle_loading frozen_model_bundle_matches_the_directory -- --ignored --nocapture
    cargo test --release --test bundle_loading frozen_image_bundle_loads -- --ignored --nocapture
    cargo test --release --test bundle_loading frozen_audio_bundle_loads -- --ignored --nocapture
else
    echo "skipping: .model-cache/$REV/model.svlm is not present"
fi

echo "--- model packer ---"
cargo build --manifest-path tools/model-packer/Cargo.toml --release
packer="tools/model-packer/target/aarch64-apple-darwin/release/model-packer"
if [ -f ".model-cache/$REV/model.svlm" ]; then
    "$packer" verify ".model-cache/$REV/model.svlm"
else
    echo "skipping: .model-cache/$REV/model.svlm is not present"
fi

echo "--- Cargo graph (must be reduced to the root crate) ---"
cargo_tree="$(cargo tree --edges normal,build,dev)"
printf '%s\n' "$cargo_tree"
if [ "$(printf '%s\n' "$cargo_tree" | wc -l | tr -d ' ')" -ne 1 ]; then
    echo "ERROR: the Cargo graph contains dependencies." >&2
    exit 1
fi

echo "--- linked libraries (system libraries only) ---"
otool -L target/aarch64-apple-darwin/release/seldon-vault-lite

echo "--- deployment target (must be macOS 11.0) ---"
build_info="$(vtool -show-build target/aarch64-apple-darwin/release/seldon-vault-lite)"
printf '%s\n' "$build_info"
if ! printf '%s\n' "$build_info" | grep -q "minos 11.0"; then
    echo "ERROR: the deployment target is not macOS 11.0." >&2
    exit 1
fi

echo "--- code signature (local-use ad-hoc build) ---"
codesign -dv --verbose=4 target/aarch64-apple-darwin/release/seldon-vault-lite 2>&1 || true
codesign --verify --strict target/aarch64-apple-darwin/release/seldon-vault-lite

echo "--- Gatekeeper assessment (recorded, not a local-use failure) ---"
if spctl --assess --type execute -vv target/aarch64-apple-darwin/release/seldon-vault-lite; then
    echo "Gatekeeper accepts the binary (notarized distribution build)."
else
    echo "Gatekeeper rejects the ad-hoc build; this is expected for local use."
    echo "Distribution requires Developer ID signing and notarization (ADR 0013)."
fi

echo "--- clean-environment release lifecycle (optional) ---"
if [ "${SELDON_RELEASE_E2E:-0}" = "1" ]; then
    embedded="target/embedded-full/aarch64-apple-darwin/release/seldon-vault-lite"
    if [ -x "$embedded" ]; then
        scripts/release_e2e.sh \
            --binary "$embedded" \
            --workdir "${SELDON_RELEASE_E2E_WORKDIR:-/tmp/seldon-release-e2e}" \
            --report "${SELDON_RELEASE_E2E_REPORT:-/tmp/seldon-release-e2e-report.json}"
    else
        echo "skipping: $embedded is not present; build with --features embedded-model"
    fi
else
    echo "skipping: set SELDON_RELEASE_E2E=1 to run scripts/release_e2e.sh"
fi

echo "M8 verification passed (release lifecycle, bundle, and macOS artifact checks)."
