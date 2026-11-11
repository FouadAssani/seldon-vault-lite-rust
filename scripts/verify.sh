#!/usr/bin/env bash
# Full verification for milestone M4 (image path).
# Target: macOS Apple Silicon (aarch64-apple-darwin) only.
set -euo pipefail

cargo fmt --check
cargo clippy --all-targets -- -D warnings

# The full-model parity test is heavy; run the fast suite first, then the
# explicit release parity check when the frozen weights are present.
SELDON_SKIP_MODEL_PARITY=1 cargo test

cargo build --release --target aarch64-apple-darwin

echo "--- frozen-model parity (optional) ---"
REV="f9972ac22e696a105c482202646979e33e4b5a1e"
if [ -f ".model-cache/$REV/model.safetensors" ]; then
    cargo test --release --test text_embeddings real_model_parity -- --nocapture
    cargo test --release --test image_embeddings image_embedding_matches -- --nocapture
else
    echo "skipping: .model-cache/$REV/model.safetensors is not present"
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

echo "M4 verification passed (image path)."
