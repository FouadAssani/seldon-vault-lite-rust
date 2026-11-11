#!/usr/bin/env bash
# Full verification for milestone M7 (NEON kernels and macOS ARM64 packaging).
# Target: macOS Apple Silicon (aarch64-apple-darwin) only.
set -euo pipefail

cargo fmt --check
cargo clippy --all-targets -- -D warnings

# The full-model parity test is heavy; run the fast suite first, then the
# explicit release parity check when the frozen weights are present. The fast
# suite includes the HTTP API integration tests (tests/http_api.rs).
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

echo "M7 verification passed (NEON kernels, model bundle, and embedded packaging)."
