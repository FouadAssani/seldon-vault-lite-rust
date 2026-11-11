# AGENTS.md — Seldon Vault Lite

**Projet :** moteur d'embeddings, d'indexation et de recherche multimodale local, 100 % Rust, sans dépendance tierce, spécialisé pour `google/embeddinggemma-2`.
**Document de référence :** [`plan-seldon-vault-lite-rust-pur.md`](plan-seldon-vault-lite-rust-pur.md). En cas de doute ou de conflit, le plan fait foi. Ce fichier en est le résumé opérationnel destiné aux agents de codage et aux contributeurs.

## 0. Règle d'or

Une étape n'est terminée que si **ses tests réussissent, que ses résultats sont reproductibles et qu'aucune régression du périmètre déjà validé n'est constatée**. Une compilation qui réussit n'est pas une preuve. Ne jamais déclarer une modalité « prise en charge » si elle ne renvoie qu'un vecteur calculé à partir d'un nom de fichier ou d'un marqueur textuel.

État constaté à la rédaction (2026-11-11) : le dépôt contient le plan, le README et les skills Rust présents dans `.dsh/skills/` ; il n'y a encore ni `Cargo.toml` ni `src/`. Le premier travail est le jalon **M0**.

## 1. Mission et périmètre

- Texte et code : embeddings de requêtes et de documents.
- Images : PNG et JPEG, prétraitement et encodeur visuel natifs.
- Audio : WAV PCM d'abord, puis MP3 seulement si la matrice finale des formats l'exige ; encodage audio natif, **sans transcription**.
- Embeddings intermodaux comparables : texte ↔ image, texte ↔ audio, médias ↔ médias.
- Dimensions 768, 512, 256 et 128, avec **renormalisation après troncature**.
- Documents, métadonnées, filtres, indexation persistante, recherche hybride.
- API HTTP locale, contrôle des ressources, diagnostic, arrêt propre, reprise après interruption.
- **Hors périmètre :** LLM génératif, transcription, GPU, orchestration externe.

## 2. Contrat technique — non négociable

| Règle | Conséquence pratique | Vérification |
|---|---|---|
| **Zéro crate tierce** | Aucune entrée dans `[dependencies]`, `[dev-dependencies]` ou `[build-dependencies]`. Seuls `std` et les intrinsics de `core::arch` sont autorisés. | `cargo tree --edges normal,build,dev` ne doit afficher que le crate racine. |
| **Code produit 100 % Rust** | Pas de C, C++, Go, Python, assembleur externe, FFI ni bibliothèque applicative dynamique. Les outils de comparaison/référence restent **en laboratoire uniquement**. | Revue du code et `otool -L` sur le binaire final. |
| **Zéro moteur d'inférence externe** | Ni llama.cpp, Candle, ONNX Runtime, PyTorch, ni serveur d'inférence. | Aucun processus auxiliaire lancé par le produit. |
| **Zéro base de données externe** | Ni SQLite ni serveur DB : stockage, journal, index vectoriel et recherche textuelle sont écrits en Rust. | Aucune dépendance de stockage dans le graphe Cargo. |
| **Zéro réseau requis à l'exécution** | Fonctionne hors ligne, sans clé ni téléchargement. | Test sur machine propre et hors ligne. |
| **`unsafe` justifié uniquement** | Interdit dans les parseurs et le code métier. Autorisé pour les kernels SIMD `core::arch`, avec commentaire `// SAFETY:`, wrapper sûr et chemin scalaire équivalent testé. | Revue ligne par ligne de chaque bloc `unsafe`. |
| **Précision** | Activations en **FP32** ; poids BF16 acceptés ; **FP16 interdit pour les activations**. | Types et tests de parité. |
| **Pas de panique sur entrée hostile** | Pas de `unwrap()`/`expect()` sur une entrée externe ; erreurs typées et propagées avec `?`. `expect()` réservé aux invariants démontrés ; `unwrap()` toléré dans les tests. | `cargo clippy`, revue, tests de corruption. |
| **Bornes mémoire explicites** | Toute allocation issue d'une entrée externe est plafonnée (dimensions, pixels, taille décompressée, corps HTTP, file d'ingestion, pool de threads). | Tests de tailles extrêmes et de bombes de décompression. |
| **Déterminisme** | Le chemin scalaire FP32 est la référence de correction ; toute optimisation doit rester dans les tolérances. | Golden tests rejoués après chaque optimisation. |
| **Pas de cible macOS Intel** | Phase 1 : `aarch64-apple-darwin` uniquement. Aucune compilation ni validation `x86_64-apple-darwin`. | `rustup target list`, scripts de build. |

## 3. Cibles de compilation

1. **Phase 1 (unique cible initiale) :** `aarch64-apple-darwin`, macOS 11 ou ultérieur selon les capacités réellement utilisées. Vérifier le `MACOSX_DEPLOYMENT_TARGET` effectif du binaire.
2. **Phase 2 (portage Linux) :** `x86_64-unknown-linux-musl`, puis `aarch64-unknown-linux-musl`, en statique.
3. **Exclu :** `x86_64-apple-darwin` (Mac Intel). Aucune promesse de binaire universel.

## 4. Arborescence cible

```text
seldon-vault-lite/
├── Cargo.toml                 # zéro dépendance
├── Cargo.lock
├── README.md
├── AGENTS.md
├── docs/
│   ├── adr/                   # décisions : std seul, formats, persistance, unsafe, modèle
│   ├── architecture.md
│   ├── model-parity.md
│   ├── api.md
│   └── formats.md
├── src/
│   ├── main.rs
│   ├── api/                   # HTTP/1.1 + JSON natifs
│   ├── cli/                   # serve, embed, index, search, verify, version
│   ├── binary/                # Safetensors, en-têtes, checksums
│   ├── tensor/                # kernels scalaires et SIMD
│   ├── model/                 # tokenizer, EmbeddingGemma 2
│   ├── media/                 # PNG, JPEG, WAV, MP3 éventuel
│   ├── index/                 # BM25, KNN exact, HNSW éventuel, RRF
│   ├── storage/               # journal, snapshots, transactions
│   ├── jobs/
│   └── error.rs
├── tests/                     # tests Rust std seulement
├── fixtures/                  # golden tests compacts
└── tools/
    └── model-packer/          # Rust std seulement, utilisé au build
```

Créer les modules au fur et à mesure des jalons, pas en avance et vides. Chaque module public documenté avec `///` ou `//!`.

## 5. Boucle de travail

1. Lire le plan, l'ADR concerné et la documentation du module avant de modifier du code.
2. Identifier le jalon courant (M0–M9) et la plus petite étape vérifiable à réaliser.
3. Écrire ou compléter les **golden tests** et les fixtures avant l'implémentation.
4. Implémenter d'abord la voie scalaire de référence, puis optimiser seulement après concordance.
5. Exécuter la vérification complète ci-dessous.
6. Mettre à jour l'ADR/la documentation et consigner les résultats de benchmark si l'étape touche aux performances.
7. Committer avec un message de type `type(scope): résumé` et mentionner le jalon.

Ne jamais sauter l'étape de tests pour « gagner du temps », ni fusionner une optimisation non validée.

### Point de départ — M0

- [ ] Créer `Cargo.toml` sans aucune dépendance et un `src/main.rs` compilable avec une CLI minimale (`version`).
- [ ] Rédiger l'ADR fondateur : `std` seul, formats, données persistantes, politique d'`unsafe`, version du modèle et compatibilité.
- [ ] Consigner les invariants : pas de crash sur entrée hostile, bornes mémoire explicites, pas d'`unwrap()` sur entrée externe, déterminisme des calculs scalaires.
- [ ] Créer l'arborescence `docs/`, `src/`, `tests/`, `fixtures/` et `tools/model-packer/` sans modules vides inutiles.
- [ ] Vérifier le graphe Cargo vide, compiler pour `aarch64-apple-darwin` et contrôler les bibliothèques liées.

## 6. Commandes de vérification

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build --release --target aarch64-apple-darwin

# Contrat « zéro crate » : la sortie ne doit contenir que le crate racine.
cargo tree --edges normal,build,dev

# Bibliothèques liées : uniquement les bibliothèques système macOS.
otool -L target/aarch64-apple-darwin/release/seldon-vault-lite

# Version de déploiement réellement enregistrée dans le Mach-O.
vtool -show-build target/aarch64-apple-darwin/release/seldon-vault-lite
```

`rustup target add aarch64-apple-darwin` est un outil de développement autorisé. Aucun de ces outils ne fait partie du produit livré.

## 7. Règles de code Rust

- **Formatage et lint :** `rustfmt` et `clippy` avec `-D warnings`. Aucun avertissement ignoré sans justification écrite.
- **Nommage :** `snake_case` pour fonctions/variables, `CamelCase` pour types, `SCREAMING_CASE` pour constantes. Pas de préfixe `get_` ; `as_`/`to_`/`into_` selon le coût de conversion.
- **Types :** newtypes pour les identifiants et unités du domaine (`DocumentId`, `Dimension`, `TokenId`), plutôt que des `String`/`usize` nus. Rendre les états invalides impossibles à représenter quand c'est raisonnable.
- **Allocations :** `Vec::with_capacity` quand la taille est connue ; tampons d'activations réutilisables par couche ; pas de copie implicite cachée dans les hot paths ; pas de `clone()` réflexe.
- **Erreurs :** type d'erreur central dans `src/error.rs`, implémentation manuelle de `std::error::Error`, propagation avec `?`. Distinguer erreur utilisateur, entrée corrompue, limite dépassée et erreur interne.
- **Invariants :** valider formes, indices, décalages, longueurs et débordements avant tout accès. Utiliser `checked_*`/`saturating_*` quand l'arithmétique peut déborder.
- **Concurrence :** `std::thread` uniquement ; pool borné, arrêt coopératif, ordre de verrouillage documenté, pas de verrou tenu pendant une opération longue ou un I/O bloquant.
- **SIMD :** détection des capacités CPU à l'exécution ; NEON pour macOS ARM64 ; AVX2/FMA seulement au portage Linux x86-64 ; fallback scalaire obligatoire ; tests différentiels scalaire vs SIMD.
- **Documentation :** expliquer le *pourquoi* (format, contrainte, tolérance), pas seulement le *quoi*. Chaque `unsafe` porte un commentaire `// SAFETY:` vérifiable.

> Les conseils Rust génériques qui supposent des crates (`thiserror`, `anyhow`, `tokio`, `crossbeam`, `parking_lot`, `serde`, `clap`, `axum`, etc.) **ne s'appliquent pas ici** : tout doit être implémenté avec `std`. Les skills Rust présents dans `.dsh/skills/` peuvent aider pour l'ownership, les erreurs, la concurrence ou l'`unsafe`, mais leurs recommandations de dépendances sont à ignorer.

## 8. Modèle et parité numérique

- **Figer** `google/embeddinggemma-2` sur une révision précise et enregistrer les SHA-256 de **tous** les fichiers utilisés dans un manifeste versionné.
- **Inventorier** couches, dimensions, types numériques, tenseurs attendus et noms des poids. Ne pas supposer qu'un « Gemma 4 » générique est compatible : relever les éléments non standards dans les sources officielles.
- **Golden tests** construits hors du produit avec l'implémentation officielle : token IDs, sorties intermédiaires de couches et vecteurs finaux pour français/anglais, code, chaînes vides/Unicode, images variées et extraits WAV.
- **Fixtures** compactes, versionnées, avec provenance et empreintes. **Ne jamais committer les poids géants** dans Git.
- **Tolérances distinctes :** égalité exacte pour la tokenisation et les dimensions ; tolérance absolue/relative pour les tenseurs FP32 ; cosinus et rang de recherche pour les embeddings quantifiés.
- **Tokenisation :** reproduire exactement la variante déclarée par le modèle (normalisation, espaces, UTF-8, tokens spéciaux, padding, masques, limites contextuelles). Toute divergence de token IDs **bloque** l'intégration des couches d'inférence.
- **Prompts :** préfixes `SearchQuery`, `Document`, similarité et classification selon les configurations officielles ; préserver les titres des documents ; distinguer explicitement requête et document indexé.
- **Texte :** 24 couches, fenêtre glissante, pooling et projection 512 → 768 conformes, normalisation L2, dimensions Matryoshka 768/512/256/128 avec **renormalisation après réduction**.
- **Images :** décodeurs PNG (DEFLATE/zlib natif) et JPEG écrits en Rust ; plafonds stricts sur dimensions, pixels, mémoire décompressée et temps ; prétraitement officiel reproduit (couleur, redimensionnement, patches, positions, budget de tokens).
- **Audio :** WAV/RIFF PCM avec conversions et rééchantillonnage natifs ; prétraitement officiel reproduit après inspection du processeur officiel ; MP3 natif seulement s'il est requis, avec les mêmes limites et fixtures. **Pas de Whisper, pas de transcription.**
- **Intermodalité :** texte, image et audio doivent partager le même espace d'embedding et être comparables.

## 9. Stockage, index et API

### Stockage

- Modèle explicite : documents, chunks, vecteurs, modalités, sources, ACL/filtres, métadonnées ; IDs stables et versions de schéma.
- Format portable : enregistrements cadrés, longueurs bornées, checksums, version.
- Journal append-only : commit atomique, `fsync` aux points critiques, replay déterministe, récupération après arrêt brutal.
- Snapshots, compactage sûr, verrouillage des writers, lectures concurrentes.
- Tests de crash injecté à **chaque** étape d'écriture ; reprise identique ; absence de doublons et de pertes.

### Index

- Index lexical inversé avec BM25, normalisation et filtres.
- Recherche vectorielle **exacte** (cosinus/dot product normalisé) comme oracle de correction.
- HNSW natif seulement après mesures de volume et de latence, comparé à l'oracle exact.
- Fusion hybride RRF (lexical + vectoriel), pagination, suppressions cohérentes et scores déterministes.

### API HTTP et CLI

- Serveur `std::net::TcpListener` HTTP/1.1 : parsing strict, connexions bornées, timeouts, `Content-Length` validé, taille maximale des corps.
- JSON produit par le parseur interne ; codes HTTP et erreurs documentés.
- Routes : `GET /health`, `GET /ready`, `GET /v1/models`, `POST /v1/embeddings`, `POST /v1/documents`, `DELETE /v1/documents/{id}`, `POST /v1/search`, `GET /v1/jobs/{id}`.
- File d'ingestion bornée, limitation de concurrence, annulations, arrêt gracieux, observabilité locale.
- Liaison par défaut à `127.0.0.1`. Pour le LAN : configuration explicite, jeton d'accès, limites strictes. Sans TLS externe, **ne pas exposer directement à Internet**.
- CLI natif : `serve`, `embed`, `index`, `search`, `verify`, `version`. Aucun processus auxiliaire.

## 10. Tests et définition de « terminé »

Tests obligatoires avant diffusion :

- **Correction :** parité token IDs, calculs intermédiaires et vecteurs finaux face aux golden tests.
- **Multimodal :** texte → image pertinente ; texte → extrait audio ; image ↔ image.
- **Dimensions :** 128/256/512/768, troncature suivie de normalisation ; même espace pour requêtes et documents.
- **Persistance :** ingestion, modification, suppression, crash, replay, snapshot, redémarrage, restauration.
- **Robustesse :** images/audio corrompus, données arbitraires, tailles limites, JSON invalide, annulation, manque de mémoire.
- **Sécurité :** fuzzing interne avec un générateur de corpus Rust maison ; revue des parseurs et des zones SIMD `unsafe` ; tests d'exposition LAN.
- **Performance :** p50/p95/p99, RAM résidente, démarrage à froid, débit, scalabilité des vecteurs, CPU saturé.
- **Distribution :** macOS Apple Silicon uniquement en phase 1 ; aucune crate tierce ; aucune bibliothèque applicative dynamique non autorisée ; modèle inclus ; aucun réseau au runtime ; test sur un Mac propre sans Homebrew, Python ni outil d'inférence.

**Terminé** signifie : un seul binaire par architecture, validé d'abord sur macOS Apple Silicon, utilisable hors ligne ; embeddings réels texte/image/audio conformes aux références ; recherche hybride persistante ; API documentée et sécurisée pour usage local ; reprise après incident ; performances mesurées ; tests reproductibles ; source lisible et licences respectées. Le portage Linux vient ensuite, sans cible macOS Intel.

## 11. Performance et packaging

- **Profiler avant d'optimiser** : tokenisation, prétraitement, matmul, attention, chargement, requêtes simultanées.
- Prioriser NEON via `core::arch` sur macOS ARM64, avec chemin scalaire conservé. AVX2/FMA seulement sur Linux x86-64. **Pas d'Accelerate ni de Metal** (FFI interdit).
- Détection des capacités CPU et sélection du kernel à l'exécution, sans instruction illégale sur les anciens processeurs.
- Blocking/cache locality, parallélisme borné, réutilisation des tampons.
- Quantification Q8 éventuelle avec accumulations/activations sûres et contrôle de la dérive des embeddings ; ce n'est pas un acquis automatique.
- À chaque optimisation : vérifier les golden tests et le classement de recherche **avant** de comparer les temps. Archiver les benchmarks avec CPU, compilateur, révision du modèle, mémoire et paramètres.
- Pendant le développement, garder les fichiers du modèle **hors du binaire**. Le packer final est écrit en Rust `std`, fige poids/tokenizer/processeurs/manifeste dans un conteneur versionné et vérifié, puis est embarqué via `include_bytes!`.
- Tester les limites réelles du linker Mach-O, du démarrage, de la taille et de la RAM sur Mac Apple Silicon. Les poids BF16 seuls sont de l'ordre de 1,5 Go ; les activations et buffers augmentent la consommation.
- Vérifier `otool -L` et les scénarios Gatekeeper, sans imposer de compte développeur pour l'usage local.
- Seules les **données utilisateur, les journaux et les snapshots** sont des fichiers persistants externes. Un binaire unique ne signifie pas zéro écriture disque.

## 12. Jalons

| Jalon | Résultat observable | Débloque |
|---|---|---|
| **M0** | Dépôt Rust `std` uniquement + ADR + jeux de référence + compilation macOS ARM64 | Développement vérifiable sur Apple Silicon |
| **M1** | Safetensors/JSON/tokenisation + tenseurs FP32 | Première inférence |
| **M2** | Embeddings texte conformes | Recherche sémantique texte |
| **M3** | Index/stockage persistants + recherche hybride | Produit utile en texte |
| **M4** | Images PNG/JPEG et embeddings visuels conformes | Recherche multimodale image |
| **M5** | WAV et embeddings audio conformes | Recherche multimodale audio |
| **M6** | API HTTP, jobs et durcissement | Service local complet |
| **M7** | SIMD NEON et packaging macOS ARM64 vérifiés | Distribution macOS Apple Silicon |
| **M8** | Tests end-to-end sur Mac propre et documentation finale | Version macOS 1.0 |
| **M9** | Portage Linux x86-64/ARM64 MUSL, tests et documentation | Version Linux |

Estimation indicative : plusieurs mois au minimum pour une personne seule ; 9 à 18 mois ou davantage sont plausibles. Les phases les plus risquées : parité des encodeurs image/audio, quantification, parsers de médias, résistance aux pannes.

## 13. Risques à garder visibles

1. **Zéro crate** accroît fortement la charge de maintenance et la surface de bugs (HTTP, JSON, PNG/JPEG, audio, index, persistance) : une revue de sécurité sérieuse est impérative.
2. **Modèle spécialisé** : une évolution majeure des poids ou du processeur exige une mise à jour du moteur. Figer le modèle est une fonctionnalité assumée.
3. **FP16 interdit** pour les activations : commencer en FP32, puis étudier BF16 et la quantification avec validation.
4. **Binaire unique ≠ zéro disque** : les documents indexés et le WAL doivent survivre aux redémarrages.
5. **Aucune dépendance applicative ≠ indépendance OS/CPU** : réseau, fichiers, threads et mémoire passent par la plateforme.
6. **Performance à prouver** : Rust n'est pas automatiquement plus rapide que llama.cpp ; la métrique est la performance à qualité comparable.

## 14. Checklist avant commit

- [ ] Aucune dépendance ajoutée (vérifier `cargo tree`).
- [ ] `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test` et build `aarch64-apple-darwin` passent.
- [ ] Golden tests et fixtures à jour ; aucune régression de parité.
- [ ] Aucun `unsafe` sans commentaire `// SAFETY:` et sans test du chemin scalaire.
- [ ] Aucun poids lourd, secret ou fichier de modèle commité.
- [ ] Documentation/ADR mise à jour si le comportement ou une décision change.
- [ ] Message de commit au format `type(scope): résumé`, avec le jalon concerné.

## 15. Références

- [Google — fiche officielle EmbeddingGemma 2](https://github.com/google-gemma/docs/blob/main/docs/models/embeddinggemma/model-card-v2.md)
- [Google — poids originaux EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)
- [Dalailalama — EmbeddingGemma 2 en Rust](https://github.com/Dalailalama/embeddinggemma2-rust) (à étudier, jamais comme dépendance)
- [Rust — documentation officielle `std`](https://doc.rust-lang.org/std/)
- [Rust — cibles officielles macOS](https://doc.rust-lang.org/rustc/platform-support/apple-darwin.html)
