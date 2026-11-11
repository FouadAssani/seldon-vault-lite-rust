# Architecture — état du jalon M0

## Objet

Ce document décrit l'organisation du dépôt au jalon M0 et les frontières que
les jalons suivants devront respecter. Il ne remplace pas l'ADR 0001, qui fait
foi pour les décisions de fond.

## État actuel

Le dépôt contient :

```text
Cargo.toml              # zéro dépendance
Cargo.lock              # verrouillage reproductible
rust-toolchain.toml     # stable (validée 1.99.1), clippy, rustfmt, ARM64
.cargo/config.toml      # cible aarch64-apple-darwin, macOS 11.0
src/
├── main.rs             # traduction du résultat CLI en code de sortie
├── lib.rs              # constantes de version/cible et racine des modules
├── error.rs            # type d'erreur central
└── cli/mod.rs          # parsing et exécution de `version`/`help`
tests/
└── cli.rs              # tests d'intégration du binaire
docs/
├── adr/0001-fondations.md
├── architecture.md
└── model-parity.md
fixtures/
└── README.md           # politique des golden tests (aucune fixture réelle)
tools/
└── model-packer/
    └── README.md       # rôle prévu au jalon M12
scripts/
└── verify.sh           # porte de qualité complète du jalon
```

Aucun module `api`, `binary`, `tensor`, `model`, `media`, `index`,
`storage` ou `jobs` n'est créé à vide. Ils apparaîtront avec leur première
fonctionnalité testable, au jalon correspondant.

## Couches et dépendances

La bibliothèque `seldon_vault_lite` est la racine testable ; le binaire
`main.rs` ne fait qu'appeler la CLI et convertir les erreurs en codes de
sortie. Les futurs modules suivront les dépendances descendantes :

```text
cli / api  →  model / index / storage  →  tensor / binary / media  →  error
```

Règles :

- `error` ne dépend d'aucun autre module interne.
- Les parseurs ne dépendent pas des couches d'inférence.
- Le stockage ne dépend pas de l'API HTTP.
- Aucun module ne dépend d'une crate tierce.

## Contrat CLI

| Commande | Effet | Sortie | Code |
|---|---|---|---|
| `version` | Affiche `seldon-vault-lite <version> (<cible>)` | stdout | 0 |
| `help` | Affiche l'aide, les commandes et les codes | stdout | 0 |
| `--version`, `-V` | Alias de `version` | stdout | 0 |
| `--help`, `-h` | Alias de `help` | stdout | 0 |
| aucune commande | Erreur d'usage | stderr | 2 |
| commande inconnue | Erreur d'usage | stderr | 2 |
| argument superflu | Erreur d'usage | stderr | 2 |
| argument non UTF-8 | Erreur d'usage | stderr | 2 |
| échec d'écriture | Erreur d'exécution | stderr (best-effort) | 1 |
| tuyau fermé | Fin normale | — | 0 |

Les commandes `serve`, `embed`, `index`, `search` et `verify` sont
planifiées mais **ne sont pas encore acceptées**. Les refuser explicitement
vaut mieux que renvoyer un résultat factice.

## Modèle d'erreurs

`src/error.rs` définit `Error` et `Result<T>`. Les variantes actuelles sont
`Usage(String)` et `Io(io::Error)`. L'énumération est `#[non_exhaustive]`
pour permettre l'ajout des variantes `Corrupt`, `LimitExceeded` et
`Internal` sans casser les appelants.

Règles :

- `?` propage les erreurs ; aucun `unwrap()` sur une entrée externe.
- `Display` produit un message destiné à l'utilisateur ; `source()` expose la
  cause technique.
- `exit_code()` centralise la traduction en code de sortie.
- L'API HTTP réutilisera la même taxonomie pour choisir les statuts.

## Frontières de confiance

Tout ce qui vient de l'extérieur est non fiable : arguments CLI, fichiers,
corps HTTP, JSON, en-têtes Safetensors, images, audio et métadonnées. Chaque
frontière doit :

1. valider les types, longueurs, offsets et versions avant allocation ;
2. plafonner les dimensions et la mémoire ;
3. utiliser une arithmétique vérifiée ;
4. renvoyer une erreur typée, jamais une panique.

Le chemin scalaire FP32 reste la référence de correction pour tous les calculs
numériques. Les optimisations SIMD ne seront acceptées qu'après concordance
avec ce chemin et rejeu des golden tests.

## Configuration de compilation

- Cible : `aarch64-apple-darwin` (imposée par `.cargo/config.toml`).
- Déploiement : `MACOSX_DEPLOYMENT_TARGET=11.0` (forcé).
- `rust-toolchain.toml` : chaîne stable (validée avec Rust 1.99.1),
  `clippy`, `rustfmt`.
- `Cargo.toml` : aucune dépendance ; profil release `opt-level = 3`,
  `lto = "thin"`, `codegen-units = 1`.

## Vérification

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build --release --target aarch64-apple-darwin
cargo tree --edges normal,build,dev
otool -L target/aarch64-apple-darwin/release/seldon-vault-lite
vtool -show-build target/aarch64-apple-darwin/release/seldon-vault-lite
```

Le script `scripts/verify.sh` regroupe ces commandes et échoue si le graphe
Cargo contient une dépendance ou si `minos` n'est pas 11.0.

La sortie de `cargo tree` doit être réduite à
`seldon-vault-lite v0.1.0`. `otool -L` ne doit lister que
`/usr/lib/libSystem.B.dylib` (et éventuellement les bibliothèques système
attendues par le runtime Rust). `vtool` doit confirmer `minos 11.0`.

## Prochaines étapes

1. **M1** : lecteur d'octets borné, JSON strict, Safetensors, BF16 ↔ FP32,
   tenseurs et kernels scalaires FP32.
2. **M2** : tokenisation exacte et embeddings texte conformes.
3. **M3** : stockage persistant, index BM25/KNN et fusion RRF.
4. **M4/M5** : décodeurs PNG/JPEG/WAV et embeddings multimodaux.
5. **M6** : API HTTP, jobs et durcissement.
6. **M7** : NEON et packaging macOS.
7. **M8/M9** : validation finale macOS, puis portage Linux.

Aucune de ces étapes ne doit être déclarée terminée sans tests reproductibles
et sans rejeu des golden tests.
