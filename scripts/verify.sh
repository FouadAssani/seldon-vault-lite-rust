#!/usr/bin/env bash
# Vérification complète du jalon M0.
# Cible : macOS Apple Silicon (aarch64-apple-darwin) uniquement.
set -euo pipefail

cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build --release --target aarch64-apple-darwin

echo "--- graphe Cargo (doit être réduit à la crate racine) ---"
cargo_tree="$(cargo tree --edges normal,build,dev)"
printf '%s\n' "$cargo_tree"
if [ "$(printf '%s\n' "$cargo_tree" | wc -l | tr -d ' ')" -ne 1 ]; then
    echo "ERREUR: le graphe Cargo contient des dépendances." >&2
    exit 1
fi

echo "--- bibliothèques liées (bibliothèques système uniquement) ---"
otool -L target/aarch64-apple-darwin/release/seldon-vault-lite

echo "--- cible de déploiement (doit être macOS 11.0) ---"
build_info="$(vtool -show-build target/aarch64-apple-darwin/release/seldon-vault-lite)"
printf '%s\n' "$build_info"
if ! printf '%s\n' "$build_info" | grep -q "minos 11.0"; then
    echo "ERREUR: la cible de déploiement n'est pas macOS 11.0." >&2
    exit 1
fi

echo "Vérification M0 réussie."
