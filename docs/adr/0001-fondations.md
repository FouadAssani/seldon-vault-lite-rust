# ADR 0001 — Fondations du moteur Seldon Vault Lite

- **Statut** : accepté
- **Date** : 2026-11-11
- **Jalon** : M0
- **Décideur** : mainteneur du projet
- **Remplace** : aucun
- **Remplacé par** : aucun

## Contexte

Seldon Vault Lite doit devenir un moteur local d'embeddings, d'indexation et
de recherche multimodale, distribué sous la forme d'un exécutable autonome par
plateforme. Le projet impose un contrat inhabituel : **zéro crate tierce**,
**code produit 100 % Rust**, aucun moteur d'inférence externe, aucune base de
données externe et aucun réseau requis à l'exécution. La première cible est
macOS Apple Silicon (`aarch64-apple-darwin`) ; macOS Intel est exclu.

Le jalon M0 doit rendre le développement vérifiable : un dépôt Rust qui compile
pour la cible initiale, une CLI minimale, un type d'erreur central, des tests
standards, une ADR fondatrice et des critères de référence documentés.

## Décisions

### D1 — Rust `std` uniquement, aucune crate tierce

Le `Cargo.toml` ne déclare **aucune** section `[dependencies]`,
`[dev-dependencies]` ni `[build-dependencies]`. Seuls `std` et les
intrinsics de `core::arch` sont autorisés. Les outils de comparaison ou
d'implémentation de référence restent **en laboratoire uniquement** et ne font
pas partie du produit.

Conséquences : HTTP, JSON, PNG, JPEG, WAV, Safetensors, index, stockage et
parseurs doivent être écrits en Rust. Toute proposition ajoutant une crate est
rejetée, y compris pour les tests.

Vérification : `cargo tree --edges normal,build,dev` ne doit afficher que la
crate racine.

### D2 — Cible macOS Apple Silicon et plancher macOS 11

La phase 1 ne compile et ne valide que `aarch64-apple-darwin`. La cible
`x86_64-apple-darwin` est exclue : aucune compilation, aucun binaire
universel, aucune promesse de support Intel. Les cibles Linux
(`x86_64-unknown-linux-musl` puis `aarch64-unknown-linux-musl`) sont
reportées au jalon M9.

Le plancher de déploiement est **macOS 11.0**, valeur imposée par
`.cargo/config.toml` via `MACOSX_DEPLOYMENT_TARGET` avec `force = true`.
Le `MACOSX_DEPLOYMENT_TARGET` réellement enregistré dans le Mach-O est
vérifié avec `vtool -show-build` ; une valeur différente de 11.0 est un
échec de validation.

Le fichier `rust-toolchain.toml` sélectionne la chaîne stable avec
`clippy`, `rustfmt` et la cible `aarch64-apple-darwin`. La version validée
pour M0 est **Rust 1.99.1** ; elle est consignée ici et dans les résultats de
vérification afin de rendre les mesures reproductibles.

### D3 — Formats et frontières de confiance

Les formats sont traités comme des **entrées hostiles** jusqu'à validation
complète :

- **JSON** : parseur strict écrit en Rust, conforme pour les objets, tableaux,
  nombres, chaînes, échappements et Unicode. Profondeur, longueur de chaînes
  et taille d'entrée plafonnées. Aucun découpage naïf de chaînes.
- **Safetensors** : lecture seule, en-tête JSON borné, offsets et longueurs
  vérifiés, chevauchements rejetés, aucune exécution de métadonnées.
- **Conteneur interne des poids** : magie, version, sections cadrées,
  longueurs bornées, checksums et manifeste. Format défini au jalon M1 et
  versionné ; il ne remplace pas les poids originaux tant que la parité n'est
  pas démontrée.
- **Médias** : PNG (DEFLATE/zlib natif), JPEG et WAV/RIFF PCM décodés en Rust,
  avec plafonds sur dimensions, pixels, mémoire décompressée et durée.
  Aucun processus externe, aucune bibliothèque applicative dynamique.
- **Texte** : UTF-8 strict. Aucune conversion avec perte silencieuse sur les
  entrées du modèle.

Chaque parseur doit rejeter les troncatures, les tailles extrêmes, les
décalages invalides et les débordements arithmétiques avant tout accès.

### D4 — Données persistantes hors du binaire

Le binaire contient le moteur et, à terme, les poids du modèle. Les
**données utilisateur, les journaux et les snapshots** sont les seuls fichiers
persistants externes. Ils vivent dans un répertoire de travail explicite.

Le stockage sera un journal append-only avec enregistrements cadrés, longueurs
bornées, checksums et version de schéma. Les commits seront atomiques et
`fsync` sera utilisé aux points critiques. Le replay doit être déterministe et
la reprise après arrêt brutal doit produire le même état, sans doublon ni
perte. Des snapshots et un compactage sûr seront ajoutés au jalon M3 ; aucune
base de données externe n'est autorisée.

### D5 — Politique d'`unsafe`

`unsafe` est **interdit par défaut**, en particulier dans les parseurs, le
stockage, le code métier et la CLI.

Il n'est autorisé que pour les kernels SIMD `core::arch` (NEON sur macOS
ARM64, puis AVX2/FMA sur Linux x86-64), avec :

1. un commentaire `// SAFETY:` qui justifie chaque invariant ;
2. un wrapper sûr qui empêche tout appel invalide ;
3. un chemin scalaire équivalent conservé ;
4. des tests différentiels scalaire vs SIMD.

Aucun FFI, aucun `asm!`, aucune assembleur externe, aucune bibliothèque
applicative dynamique. Chaque bloc `unsafe` est relu ligne par ligne.

### D6 — Modèle figé et parité numérique

Le modèle cible est **`google/embeddinggemma-2`**, figé sur une révision
précise. Un manifeste versionné enregistre les SHA-256 de **tous** les fichiers
utilisés (configuration, tokenizer, processeurs, poids, prompts). Les poids
géants ne sont jamais commités dans Git.

Contraintes numériques :

- poids BF16 acceptés, activations en **FP32** ;
- **FP16 interdit pour les activations** ;
- dimensions Matryoshka 768, 512, 256 et 128, avec **renormalisation L2 après
  troncature** ;
- prompts `SearchQuery` et `Document` selon les configurations officielles ;
- aucune hypothèse de compatibilité avec un « Gemma » générique : les
  éléments non standards sont relevés dans les sources officielles.

Le jalon M0 ne charge aucun poids et ne revendique aucune modalité. Les
premiers golden tests réels sont produits au jalon M1/M2 avec l'implémentation
officielle, hors du produit.

### D7 — Invariants de robustesse et de déterminisme

Ces invariants s'appliquent à tout le code produit :

- **Pas de panique sur entrée hostile** : aucun `unwrap()` ni `expect()` sur
  une entrée externe. Les erreurs sont typées et propagées avec `?`.
  `expect()` est réservé aux invariants démontrés ; `unwrap()` est toléré
  dans les tests.
- **Bornes mémoire explicites** : toute allocation issue d'une entrée externe
  est plafonnée (dimensions, pixels, taille décompressée, corps HTTP, file
  d'ingestion, pool de threads).
- **Arithmétique vérifiée** : utilisation de `checked_*`/`saturating_*`
  lorsque des débordements sont possibles ; validation des formes, indices,
  décalages et longueurs avant tout accès.
- **Déterminisme** : le chemin scalaire FP32 est la référence de correction.
  Toute optimisation doit rester dans les tolérances et les golden tests sont
  rejoués après chaque optimisation.
- **Pas de conversion implicite avec perte** : les entrées modèle sont
  validées avant conversion.

La taxonomie d'erreurs distinguera progressivement : erreur d'usage, entrée
corrompue, limite dépassée et erreur interne.

### D8 — CLI et codes de sortie

La CLI du jalon M0 expose `version` et `help`, avec les alias
`--version`/`-V` et `--help`/`-h`. Les commandes `serve`, `embed`,
`index`, `search` et `verify` seront ajoutées aux jalons suivants ; aucun
stub ne simule un comportement inexistant.

Contrat observable :

- les données vont sur `stdout`, les diagnostics sur `stderr` ;
- code de sortie `0` : succès ;
- code de sortie `1` : erreur d'exécution (E/S) ;
- code de sortie `2` : erreur d'usage (commande inconnue, argument superflu,
  argument non UTF-8) ;
- un tuyau fermé (`EPIPE`) est traité comme un succès, conformément aux
  conventions des outils Unix ;
- aucun panique sur argument non UTF-8 : l'erreur est typée.

### D9 — Vérification reproductible

Les commandes suivantes constituent la porte de qualité du dépôt :

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build --release --target aarch64-apple-darwin
cargo tree --edges normal,build,dev
otool -L target/aarch64-apple-darwin/release/seldon-vault-lite
vtool -show-build target/aarch64-apple-darwin/release/seldon-vault-lite
```

Le graphe Cargo ne doit contenir que la crate racine. `otool -L` ne doit
montrer que les bibliothèques système macOS. `vtool -show-build` doit montrer
une cible de déploiement de 11.0. Aucune de ces vérifications ne doit être
sautée avant un commit. Le script `scripts/verify.sh` exécute cette porte de
qualité et vérifie automatiquement le graphe Cargo et `minos 11.0`.

## Conséquences

### Positives

- Le contrat « zéro crate » est vérifiable dès le premier commit.
- La cible et le plancher macOS sont explicites et contrôlables.
- La CLI et les erreurs sont testables sans lancer de processus.
- Les invariants de robustesse sont posés avant l'arrivée des parseurs.
- Les critères de parité et de fixtures sont documentés avant l'inférence.

### Coûts et risques

- Tout doit être réimplémenté : HTTP, JSON, médias, stockage, index. La surface
  de bugs et de sécurité augmente fortement ; une revue dédiée est nécessaire.
- Le modèle est figé : une évolution majeure des poids ou du processeur exigera
  une mise à jour du moteur.
- Les performances ne sont pas acquises : elles devront être mesurées et
  comparées à qualité comparable, pas seulement supposées.
- Le binaire unique n'implique pas l'absence d'écriture disque : le WAL, les
  snapshots et les données utilisateur persistent.

### Suivi

- **ADR 0002 (M1)** : format interne des poids et du conteneur versionné.
- **ADR 0003 (M1/M3)** : format du journal, snapshots et transactions.
- **ADR 0004 (M6)** : contrat HTTP, authentification LAN et limites.
- La licence du code source sera tranchée avant toute distribution ; les poids
  restent soumis à la licence officielle du modèle.
