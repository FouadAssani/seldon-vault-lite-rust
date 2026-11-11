# Seldon Vault Lite — Plan de réalisation 100 % Rust, sans dépendance tierce

**Version :** 1.0 — 11 novembre 2026  
**Objectif :** construire, depuis les formats et algorithmes de base, un moteur d'embeddings, d'indexation et de recherche multimodale, entièrement local et CPU, spécialisé pour `google/embeddinggemma-2`, distribué sous forme d'un exécutable autonome par plateforme. **Première cible : macOS Apple Silicon uniquement (ARM64) ; Linux x86-64 et ARM64 ensuite. Aucun support macOS Intel.**

> Ce document est une feuille de route de développement, pas une affirmation que la compatibilité numérique, les performances ou l'autonomie du binaire sont déjà acquises.

## 0. Le contrat du défi

### Ce que « sans dépendance » signifie ici

1. **Code produit 100 % Rust** : aucun C, C++, Go, Python, assembleur externe, FFI, `unsafe` non justifié ou bibliothèque applicative dynamique.
2. **Zéro crate tierce** : le `Cargo.toml` de production ne déclare aucune dépendance, y compris en `dev-dependencies` et `build-dependencies`. `std` et les intrinsics de `core::arch` sont autorisés.
3. **Zéro moteur d'inférence externe** : ni llama.cpp, ni Candle, ni ONNX Runtime, ni PyTorch, ni serveur d'inférence.
4. **Zéro base de données externe** : ni SQLite ni serveur DB ; stockage, journal, index vectoriel et recherche textuelle sont développés en Rust.
5. **Zéro service ou réseau requis à l'exécution** : le produit fonctionne hors ligne, sans clé ni téléchargement.
6. **Un exécutable CPU par couple OS/architecture** contenant le moteur, les ressources de traitement et les poids du modèle. Les données indexées restent persistées dans un répertoire de travail : « un seul binaire » ne signifie pas « aucun fichier de données ».
7. **Aucune promesse de portabilité binaire universelle** : livrer d'abord `aarch64-apple-darwin` (Mac Apple Silicon seulement), puis `x86_64-unknown-linux-musl` et `aarch64-unknown-linux-musl`. La cible `x86_64-apple-darwin` est explicitement exclue. Un exécutable macOS ne peut pas être entièrement détaché des bibliothèques système fournies par macOS ; cela ne constitue pas une dépendance applicative à installer.
8. **Outils de développement autorisés** : compilateur Rust/Cargo, Git, outils de débogage, scripts de comparaison et implémentation de référence *en laboratoire uniquement*. Aucun ne fait partie du produit livré.

### Périmètre fonctionnel définitif

- Texte et code : embeddings de requêtes et de documents.
- Images : prise en charge PNG et JPEG, avec prétraitement et encodeur visuel natifs.
- Audio : prise en charge WAV PCM en premier, puis MP3 si la matrice des formats finale l'exige ; encodage audio **natif**, sans conversion en transcription.
- Embeddings intermodaux comparables : texte ↔ image, texte ↔ audio et médias ↔ médias.
- Dimensions : 768, 512, 256 et 128, avec renormalisation après troncature.
- Documents, métadonnées, filtres, indexation persistante et recherche hybride.
- API HTTP locale, contrôle des ressources, diagnostic, fermeture propre et reprise après interruption.
- Aucun LLM génératif, aucune transcription, aucun GPU et aucune orchestration externe.

### Règles de sortie

Une étape n'est terminée que si ses tests réussissent, que ses résultats sont reproductibles et qu'aucune régression du périmètre déjà validé n'est constatée. Ne jamais déclarer une modalité « prise en charge » si elle renvoie seulement un vecteur calculé à partir d'un nom de fichier ou d'un marqueur textuel.

## 1. Préparer le référentiel et les critères de qualité

1. Créer le dépôt `seldon-vault-lite`, la branche de développement et l'arborescence Rust.
2. Créer un `Cargo.toml` **sans** `[dependencies]`, `[dev-dependencies]` et `[build-dependencies]` remplis.
3. Définir **macOS Apple Silicon comme plateforme initiale unique** avec `aarch64-apple-darwin`. Exclure toute compilation ou validation macOS Intel ; reporter `x86_64-unknown-linux-musl` puis `aarch64-unknown-linux-musl` à la phase de portage Linux.
4. Définir la version minimale de macOS pour Apple Silicon (macOS 11 ou version ultérieure selon les capacités utilisées), puis vérifier le `MACOSX_DEPLOYMENT_TARGET` réel du binaire compilé.
5. Écrire un ADR pour les choix : `std` seulement, formats, données persistantes, politique d'`unsafe`, versions de modèle et compatibilité.
6. Fixer les invariants : pas de crash sur entrée hostile, bornes mémoire explicites, gestion des erreurs sans `unwrap()` sur entrée externe, déterminisme des calculs scalaires.
7. Ajouter des commandes de vérification reproductibles : `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test`, `cargo build --release` et inspection de la fermeture des dépendances Cargo.

**Validation :** application CLI minimale compilable, tests unitaires standards, graphe Cargo vide et pas de liaison à une bibliothèque applicative additionnelle.

## 2. Figer le modèle et produire une référence de vérité

1. Figer le dépôt officiel `google/embeddinggemma-2` sur une révision précise et enregistrer les SHA-256 de **tous** les fichiers utilisés.
2. Étudier les paramètres exacts (`config.json`, `tokenizer.json`/modèle de tokenisation, configuration du processeur, poids Safetensors, prompts et règles multimodales).
3. Écrire un inventaire des couches, dimensions, types numériques, tenseurs attendus et noms des poids. Ne pas supposer qu'un modèle « Gemma 4 » générique est compatible.
4. Construire hors du produit des *golden tests* avec l'implémentation officielle : tokens, sorties intermédiaires de couches et vecteurs finals, pour textes français/anglais, code, chaînes vides/Unicode, images variées et extraits WAV.
5. Enregistrer les entrées et sorties attendues sous forme de fixtures versionnées de petite taille, avec provenance et empreintes ; ne pas committer les poids géants dans Git.
6. Définir les tolérances distinctes : comparaison exacte pour la tokenisation et les dimensions, tolérance absolue/relative pour les tenseurs FP32, cosinus et rang de recherche pour les embeddings quantifiés.

**Validation :** manifeste du modèle figé, banc de référence documenté et corpus « golden » couvrant chaque modalité. Cette étape précède l'optimisation.

## 3. Développer les formats et primitives binaires en Rust

1. Créer un lecteur d'octets borné avec vérification systématique des tailles, décalages, dépassements et endianness.
2. Implémenter un parseur JSON strict limité aux fonctionnalités nécessaires, mais conforme pour les objets, tableaux, nombres, chaînes, échappements et caractères Unicode ; aucun parseur naïf par découpage de chaînes.
3. Implémenter un lecteur Safetensors : en-tête JSON, métadonnées, vues sur données, noms, formes, strides et vérification des chevauchements/offsets.
4. Ajouter les conversions BF16 ↔ FP32 et la lecture des éléments F32/BF16 ; **ne pas utiliser FP16 comme précision d'activation**.
5. Prévoir un chargeur multi-fichiers/shards et une vérification des empreintes avant utilisation.
6. Définir le format interne versionné des poids embarqués, mais ne pas l'imposer à l'inférence tant que la parité avec les poids originaux n'est pas démontrée.

**Validation :** tests de corruption, troncature, tailles extrêmes, caractères invalides, ordre des champs et accès mémoire hors limites. Aucune exécution de code depuis les métadonnées.

## 4. Construire le noyau de calcul CPU

1. Définir `Tensor`, `Shape`, `DType`, `TensorView` et des allocations bornées, sans copies implicites cachées.
2. Implémenter une voie scalaire FP32 de référence : addition, multiplication, matmul, produits scalaires, softmax stable, RMSNorm, GELU/GeGLU, RoPE, attention et normalisation L2.
3. Ajouter la validation des formes et prévenir les `NaN`, divisions par zéro, accumulations instables et dépassements d'indices.
4. Concevoir un tampon d'activations réutilisable afin de ne pas allouer à chaque couche.
5. Ajouter un ordonnanceur CPU en Rust avec `std::thread`, work queue bornée, arrêt coopératif et taille du pool configurable.
6. Comparer les résultats des opérations aux fixtures du modèle de référence ; optimiser seulement après concordance.

**Validation :** chaque opération a des tests numériques, des entrées pathologiques et des benchmarks ; le mode scalaire est la référence de correction.

## 5. Reproduire exactement la tokenisation et les instructions

1. Identifier l'algorithme réellement déclaré dans les fichiers du modèle et implémenter **cette** variante, avec sa normalisation et ses règles d'espaces.
2. Implémenter UTF-8, tokens spéciaux, padding, masques et limites contextuelles selon les configurations officielles.
3. Reproduire les préfixes de tâches `SearchQuery`, `Document`, similarité, classification et autres usages retenus.
4. Préserver les titres des documents dans le prompt prévu ; distinguer explicitement une requête d'un document indexé.
5. Comparer les identifiants de tokens à la référence, y compris français accentué, emoji, code et entrées longues.

**Validation :** égalité exacte des token IDs sur le corpus de référence. Toute divergence bloque l'intégration des couches d'inférence.

## 6. Implémenter EmbeddingGemma 2 texte

1. Charger les poids du modèle textuel et vérifier noms, formes et types de tous les tenseurs.
2. Implémenter l'architecture exacte : embeddings, entrées par couche, mécanismes d'attention locale/globale, RoPE, normalisations, portes et projections.
3. Respecter les 24 couches, le comportement de fenêtre glissante et les détails spécifiques à l'architecture Gemma 4/EmbeddingGemma 2, notamment les éléments non standards relevés dans les sources officielles.
4. Réaliser le pooling et la projection 512 → 768 conformément au modèle.
5. Ajouter la normalisation L2 et les dimensions Matryoshka 768/512/256/128, **avec renormalisation après réduction**.
6. Supporter les requêtes simultanées de façon bornée et mesurer le compromis latence/débit/mémoire.

**Validation :** correspondance des sorties de couches et des embeddings de référence, puis classement cohérent d'un corpus texte/code. Aucun résultat approché non validé n'est accepté comme « compatible ».

## 7. Implémenter la voie image entièrement en Rust

1. Développer des décodeurs PNG (avec DEFLATE/zlib natif) et JPEG limité d'abord aux variantes annoncées puis élargi selon les fixtures.
2. Imposer des plafonds sur dimensions, nombre de pixels, mémoire décompressée et temps de traitement, afin d'éviter les bombes de décompression.
3. Reproduire le prétraitement officiel : conversion couleur, redimensionnement, placement des patches, positions et budgets de tokens visuels.
4. Implémenter l'encodeur d'image, les projections et son raccordement à l'espace d'embedding commun.
5. Tester plusieurs images différentes, les rotations et rapports d'aspect, puis la recherche **texte → image**.

**Validation :** différences attendues entre images distinctes, compatibilité avec les vecteurs de référence et résultats intermodaux. Aucun recours à un programme externe pour décoder les images.

## 8. Implémenter la voie audio entièrement en Rust

1. Implémenter WAV/RIFF PCM (formats et fréquences explicitement annoncés). Prévoir les conversions et le rééchantillonnage dans le produit.
2. Implémenter exactement le prétraitement officiel (fenêtrage, transformée spectrale, représentation de fréquences, canaux, masques et découpage), **après inspection du processeur officiel**.
3. Implémenter l'encodeur audio et sa projection dans le même espace que le texte et l'image.
4. Tester silence, sons courts, sons longs, voix française, musique et sons d'environnement ; comparer à la référence.
5. Si MP3 est un format final requis, ajouter un décodeur MP3 natif après stabilisation du WAV et lui appliquer les mêmes limites/fixtures. Ne jamais annoncer « tous les audios ».

**Validation :** vecteurs audio conformes aux golden tests et recherche texte → segment audio fonctionnelle, sans Whisper ni transcription.

## 9. Optimiser le moteur sans sacrifier la correction

1. Profiler le CPU sur de vrais textes, images et sons : temps de tokenisation, prétraitement, matmul, attention, chargement et requêtes simultanées.
2. Prioriser les kernels macOS Apple Silicon AArch64 via NEON avec `core::arch`, tout en conservant un chemin scalaire. Les kernels AVX2/FMA ne seront ajoutés qu'au portage Linux x86-64. Ne pas dépendre du framework Accelerate ni de Metal, puisque le défi impose du Rust CPU sans FFI.
3. Ajouter la détection des capacités CPU et la sélection du kernel à l'exécution, sans instruction illégale sur les anciens processeurs.
4. Expérimenter blocking/cache locality, parallélisme et réutilisation des tampons.
5. Concevoir si nécessaire une quantification propre des poids (Q8 d'abord), avec **accumulations/activations sûres** et contrôle de la dérive des embeddings. La quantification n'est pas un acquis automatique.
6. Vérifier à chaque optimisation les golden tests et le classement de recherche, avant de comparer les temps au mode scalaire et à une référence externe.

**Validation :** gain mesuré sur les machines cibles, zéro régression numérique hors seuil ; résultats de benchmark archivés avec CPU, compilateur, révision du modèle, mémoire et paramètres.

## 10. Développer le stockage persistant natif

1. Définir un modèle de documents, chunks, vecteurs, modalités, sources, ACL/filtres et métadonnées ; IDs stables et versions de schéma explicites.
2. Définir un format de stockage portable avec enregistrements cadrés, longueurs bornées, checksums et version.
3. Développer un journal append-only avec commit atomique, `fsync` aux points critiques, replay déterministe et récupération après arrêt brutal.
4. Ajouter snapshots, compactage sûr, verrouillage des writers et lectures concurrentes.
5. Ajouter un index lexical inversé avec BM25, normalisation, filtres et positions si nécessaires.
6. Démarrer par une recherche vectorielle exacte (cosinus/dot product normalisé), qui servira d'oracle de correction ; introduire HNSW natif seulement après mesures de volume/latence.
7. Implémenter la fusion hybride RRF (lexical + vectoriel), la pagination et les suppressions cohérentes.

**Validation :** tests de crash injecté à chaque étape d'écriture, reprise identique, absence de doublons/pertes, filtres et score déterministes ; comparaison HNSW avec l'oracle exact lorsqu'il existe.

## 11. Développer l'API HTTP et le CLI en Rust `std`

1. Créer un serveur `std::net::TcpListener` HTTP/1.1 avec parsing strict, connexions bornées, timeouts, `Content-Length` validé et taille maximale des corps.
2. Implémenter le JSON du contrat API avec le parseur interne ; codes HTTP et réponses d'erreur documentés.
3. Exposer :
   - `GET /health` — processus vivant ;
   - `GET /ready` — poids chargés, stockage prêt ;
   - `GET /v1/models` — version/empreinte, modalités et dimensions ;
   - `POST /v1/embeddings` — texte et média via schéma documenté ;
   - `POST /v1/documents` — ingestion et métadonnées ;
   - `DELETE /v1/documents/{id}` — suppression complète ;
   - `POST /v1/search` — recherche hybride et filtres ;
   - `GET /v1/jobs/{id}` — statut d'une ingestion.
4. Implémenter une file bornée d'ingestion, la limitation de concurrence, les annulations, le shutdown gracieux et l'observabilité locale.
5. Lier par défaut à `127.0.0.1` ; pour le LAN, exiger une configuration explicite, un jeton d'accès et de strictes limites de requêtes. Sans TLS externe, ne pas exposer directement à Internet.
6. Ajouter un CLI natif (`serve`, `embed`, `index`, `search`, `verify`, `version`) ; ne pas lancer de processus auxiliaire.

**Validation :** tests d'intégration de toutes les routes, entrées malformées, JSON Unicode, connexions lentes, requêtes volumineuses et concurrence ; aucune panique sur erreur réseau.

## 12. Fabriquer l'exécutable unique autonome

1. Pendant le développement, garder les fichiers du modèle hors du binaire pour permettre des itérations rapides.
2. Ajouter un packer **écrit lui aussi en Rust `std`** qui fige les poids, tokenizer, processeurs et manifeste dans un conteneur interne versionné et vérifié.
3. Embarquer ce conteneur au moment de la compilation finale (`include_bytes!` ou mécanisme Rust équivalent), sans téléchargement au premier démarrage.
4. Tester les limites réelles du linker Mach-O, du démarrage, de la taille et de la RAM sur **Mac Apple Silicon uniquement**. Les poids BF16 seuls sont de l'ordre de 1,5 Go ; les activations, indices et buffers augmentent la consommation mémoire.
5. Produire et tester **en premier** le binaire macOS `aarch64-apple-darwin`. Vérifier ses bibliothèques liées et qu'aucune bibliothèque applicative tierce n'est requise. Contrôler également la signature/les restrictions Gatekeeper dans les scénarios de distribution, sans imposer de compte développeur pour les usages locaux.
6. Ensuite seulement, porter vers Linux x86-64 et ARM64 MUSL statiques et vérifier l'absence de bibliothèques dynamiques non autorisées.
7. Conserver **uniquement** les données utilisateur, journaux et snapshots comme fichiers persistants externes.

**Validation :** copier le seul exécutable sur une machine propre et hors ligne, démarrer le service, indexer du texte/une image/un WAV, rechercher, arrêter et redémarrer sans perte de données ni fichiers de modèle externes.

## 13. Campagne de tests complète avant diffusion

### Tests obligatoires

- **Correction :** parité token IDs, calculs intermédiaires et vecteurs finaux face aux golden tests.
- **Multimodal :** requête texte renvoyant une image pertinente ; requête texte renvoyant un extrait audio ; recherche image↔image.
- **Dimensions :** 128/256/512/768, troncature suivie de normalisation ; même espace pour requêtes et documents.
- **Persistance :** ingestion, modification, suppression, crash, replay, snapshot, redémarrage et restauration.
- **Robustesse :** images/audio corrompus, données arbitraires, limites de tailles, JSON invalide, annulation et manque de mémoire.
- **Sécurité :** fuzzing interne à partir d'un générateur de corpus Rust maison ; analyse des parsers et des zones SIMD `unsafe` ; tests d'exposition LAN.
- **Performance :** p50/p95/p99, RAM résidente, démarrage à froid, débit, scalabilité des vecteurs et CPU saturé.
- **Distribution :** sur macOS Apple Silicon seulement dans la première phase, aucune crate tierce, absence de bibliothèque applicative dynamique non autorisée, modèle inclus, aucun réseau au runtime. Tester sur un Mac Apple Silicon propre sans Homebrew, Python ou outil d'inférence installé. Tester ensuite chaque architecture Linux séparément.

### Définition de « terminé »

Un seul binaire par architecture, **validé d'abord uniquement sur macOS Apple Silicon**, utilisable hors ligne ; embeddings **réels** texte/image/audio conformes aux références ; recherche hybride persistante ; API documentée et sécurisée pour usage local ; reprise après incident ; performances mesurées ; tests reproductibles ; source lisible et licences respectées. Le portage Linux x86-64/ARM64 vient ensuite, sans cible macOS Intel.

## 14. Ordre de livraison — jalons sans raccourci

| Jalon | Résultat observable | Débloque |
|---|---|---|
| M0 | Dépôt Rust `std` uniquement + ADR + jeux de référence + compilation macOS ARM64 | Développement vérifiable sur Apple Silicon |
| M1 | Safetensors/JSON/tokenisation + tenseurs FP32 | Première inférence |
| M2 | Embeddings texte conformes | Recherche sémantique texte |
| M3 | Index/stockage persistants + recherche hybride | Produit utile en texte |
| M4 | Images PNG/JPEG et embeddings visuels conformes | Recherche multimodale image |
| M5 | WAV et embeddings audio conformes | Recherche multimodale audio |
| M6 | API HTTP, jobs et durcissement | Service local complet |
| M7 | SIMD NEON et packaging macOS ARM64 vérifiés | Distribution macOS Apple Silicon |
| M8 | Tests end-to-end sur Mac Apple Silicon propre et documentation finale | Version macOS 1.0 |
| M9 | Portage Linux x86-64/ARM64 MUSL, tests et documentation | Version Linux |

**Estimation indicative :** pour une personne seule, plusieurs mois au minimum ; 9 à 18 mois ou davantage sont plausibles pour un moteur multimodal robuste, optimisé et vérifié, avec décodeurs et stockage écrits intégralement. Les phases les plus risquées sont la parité des encodeurs image/audio, la quantification, les parsers de médias et la résistance aux pannes.

## 15. Arborescence finale suggérée

```text
seldon-vault-lite/
├── Cargo.toml                 # zéro dépendance
├── Cargo.lock
├── README.md
├── docs/
│   ├── adr/
│   ├── architecture.md
│   ├── model-parity.md
│   ├── api.md
│   └── formats.md
├── src/
│   ├── main.rs
│   ├── api/                  # HTTP/1.1 + JSON natifs
│   ├── cli/
│   ├── binary/               # Safetensors, entêtes, checksums
│   ├── tensor/               # kernels scalaires et SIMD
│   ├── model/                # tokenizer, Gemma 4/EmbeddingGemma 2
│   ├── media/                # PNG, JPEG, WAV, MP3 éventuel
│   ├── index/                # BM25, KNN, HNSW éventuel, RRF
│   ├── storage/              # journal, snapshots, transactions
│   ├── jobs/
│   └── error.rs
├── tests/                    # tests Rust std seulement
├── fixtures/                 # golden tests compacts
└── tools/
    └── model-packer/         # Rust std seulement, utilisé au build
```

## 16. Risques à garder visibles

1. **Zéro crate** accroît beaucoup la charge de maintenance et la surface de bugs dans HTTP, JSON, PNG/JPEG, audio, index et persistance : une revue de sécurité sérieuse devient impérative.
2. **Modèle spécialisé** signifie qu'une évolution majeure de la structure des poids ou du processeur exige une mise à jour du moteur ; figer le modèle est une fonctionnalité, pas une faiblesse cachée.
3. **FP16 interdit** pour les activations selon la documentation officielle ; commencer par FP32, puis étudier BF16 et quantification des poids avec validation.
4. **Un seul fichier exécutable ≠ zéro écriture disque** : les documents indexés et le WAL doivent survivre aux redémarrages.
5. **Aucune dépendance applicative ≠ indépendance de l'OS ou du processeur** : réseau, fichiers, threads et mémoire passent nécessairement par les services de la plateforme.
6. **Performance à prouver** : Rust n'est pas automatiquement plus rapide que llama.cpp ; la métrique doit être la performance à qualité comparable.

## Références techniques de départ

- [Google — fiche officielle EmbeddingGemma 2](https://github.com/google-gemma/docs/blob/main/docs/models/embeddinggemma/model-card-v2.md) : architecture, formats, dimensions MRL, règles de précision et préfixes.
- [Google — poids originaux EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) : configuration, processeurs et checkpoints.
- [Dalailalama — EmbeddingGemma 2 en Rust](https://github.com/Dalailalama/embeddinggemma2-rust) : implémentation indépendante à étudier, sans en faire une dépendance du produit.
- [Rust — documentation officielle `std`](https://doc.rust-lang.org/std/) : primitives autorisées.
- [Rust — cibles officielles macOS](https://doc.rust-lang.org/rustc/platform-support/apple-darwin.html) : Apple Silicon et versions minimales.

**Principe directeur :** commencer par prouver l'exactitude du texte, ensuite étendre aux médias, puis seulement optimiser et intégrer les poids dans l'exécutable. Chaque étape doit être une démonstration reproductible, pas seulement une compilation qui réussit.
