#!/usr/bin/env bash
# Full verification for milestone M1.
# Target: macOS Apple Silicon (aarch64-apple-darwin) only.
set -euo pipefail

cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build --release --target aarch64-apple-darwin

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

echo "M1 verification passed."
