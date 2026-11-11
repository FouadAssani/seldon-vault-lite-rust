# Seldon Vault Lite

Moteur local d'embeddings, d'indexation et de recherche multimodale, écrit en
Rust `std` uniquement (aucune crate tierce), spécialisé pour
`google/embeddinggemma-2`.

**État : jalon M0 — fondations.** La CLI expose `version` et `help`. Les
commandes `serve`, `embed`, `index`, `search` et `verify` seront
ajoutées aux jalons suivants. Aucune modalité (texte, image, audio) n'est
encore déclarée prise en charge : aucun résultat factice n'est renvoyé.

## Prérequis

- macOS 11 ou ultérieur sur Apple Silicon (`aarch64-apple-darwin`) ;
- chaîne Rust stable (validée avec 1.99.1), sélectionnée par
  `rust-toolchain.toml` avec `clippy`, `rustfmt` et la cible
  `aarch64-apple-darwin` ;
- aucun autre outil n'est requis pour compiler.

La cible `x86_64-apple-darwin` (Mac Intel) est explicitement exclue.

## Compiler et tester

```bash
cargo build
cargo run -- version
cargo run -- --help
```

## Vérification complète

Le script `scripts/verify.sh` exécute toute la porte de qualité. Les commandes
équivalentes sont :

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build --release --target aarch64-apple-darwin

# Contrat « zéro crate » : seul le crate racine doit apparaître.
cargo tree --edges normal,build,dev

# Bibliothèques liées et cible de déploiement du Mach-O.
otool -L target/aarch64-apple-darwin/release/seldon-vault-lite
vtool -show-build target/aarch64-apple-darwin/release/seldon-vault-lite
```

Ou simplement :

```bash
./scripts/verify.sh
```

## Contrat technique

- **Zéro crate tierce** : `Cargo.toml` ne contient aucune section de
  dépendances ; seuls `std` et `core::arch` sont autorisés.
- **Code produit 100 % Rust** : pas de C, C++, FFI, `asm!` ni bibliothèque
  applicative dynamique.
- **Cible initiale** : `aarch64-apple-darwin`, macOS 11.0 minimum.
- **Données persistantes** : les données utilisateur, journaux et snapshots
  vivront hors du binaire ; les poids ne sont pas commités.
- **Robustesse** : aucune panique sur entrée hostile, bornes mémoire explicites,
  arithmétique vérifiée, déterminisme du chemin scalaire FP32.
- **CLI** : données sur `stdout`, erreurs sur `stderr` ; codes de sortie
  `0` (succès), `1` (exécution), `2` (usage).

## Documentation

- [Plan de réalisation](plan-seldon-vault-lite-rust-pur.md)
- [ADR 0001 — Fondations](docs/adr/0001-fondations.md)
- [Architecture (état M0)](docs/architecture.md)
- [Parité numérique et jeux de référence](docs/model-parity.md)
- [Politique des fixtures](fixtures/README.md)

## Skills Rust (DSH)

Les skills du dépôt [actionbook/rust-skills](https://github.com/actionbook/rust-skills)
sont installés dans `.dsh/skills/` (version 2.0.9, commit `5c40d3a`).

Mise à jour :

```bash
git clone --depth 1 https://github.com/actionbook/rust-skills.git /tmp/rust-skills
cp -R /tmp/rust-skills/skills/. .dsh/skills/
```
