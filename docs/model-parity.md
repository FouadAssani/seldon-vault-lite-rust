# Parité numérique et jeux de référence

## Statut

Au jalon M0, ce document fixe les **critères** de parité et la politique des
fixtures. Aucun poids du modèle n'est encore figé ni chargé, et aucun golden
test réel n'existe. Les valeurs de tolérance ci-dessous sont des cibles
initiales ; elles seront confirmées ou ajustées avec l'implémentation
officielle au jalon M2, et tout ajustement sera documenté avec sa justification.

## Principe

L'implémentation officielle de `google/embeddinggemma-2` sert de référence
**en laboratoire uniquement**. Elle ne fait pas partie du produit, n'est pas
une dépendance Cargo et n'est jamais lancée par le binaire livré.

Les golden tests sont produits hors du produit, puis figés dans `fixtures/`
sous forme compacte. Le produit ne fait que lire ces fixtures et comparer ses
sorties.

## Manifeste du modèle

Avant toute intégration, un manifeste versionné doit enregistrer :

- le dépôt et la **révision exacte** du modèle ;
- la liste des fichiers utilisés (config, tokenizer, processeurs, poids,
  prompts) ;
- la taille et le **SHA-256** de chaque fichier ;
- la provenance des fixtures et l'outil qui les a générées ;
- la version du schéma du manifeste.

Les poids géants ne sont jamais commités. Le manifeste, lui, est versionné.

## Fixtures

Règles :

- taille compacte ; privilégier des extraits et des tenseurs réduits ;
- provenance explicite : modèle, révision, script de génération, date ;
- empreinte SHA-256 de chaque fichier de fixture ;
- aucune donnée personnelle, aucun secret ;
- pas de poids complets, pas de fichier de plusieurs gigaoctets ;
- nommage stable et versionné.

Le fichier `fixtures/README.md` décrit la politique ; les fixtures réelles
seront ajoutées aux jalons M1 (formats), M2 (texte) et M4/M5 (médias).

## Couverture obligatoire

Les golden tests doivent couvrir au minimum :

- français accentué, anglais, code source ;
- chaîne vide, Unicode, emoji, caractères de contrôle ;
- entrées longues et limites contextuelles ;
- images variées : couleurs, niveaux de gris, rapports d'aspect, rotations ;
- extraits WAV : silence, son court, son long, voix, musique, environnement ;
- dimensions 768, 512, 256 et 128 ;
- requêtes et documents indexés, avec les préfixes officiels.

## Tolérances

| Objet | Comparaison | Tolérance initiale |
|---|---|---|
| Token IDs | égalité exacte | aucune divergence acceptée |
| Dimensions et formes | égalité exacte | aucune divergence acceptée |
| Tenseurs FP32 intermédiaires | absolue + relative | `atol <= 1e-5`, `rtol <= 1e-4` pour les valeurs de magnitude supérieure à `1e-3` |
| Embeddings FP32 finaux | cosinus + absolue | `cos >= 0.9999`, `atol <= 1e-4` après normalisation L2 |
| Embeddings quantifiés (futur) | cosinus + rappel | `cos >= 0.999`, `recall@10 >= 0.99` face à l'oracle FP32 |
| Classement de recherche FP32 | rang | top-1 identique, `recall@10 = 1.0` sur le corpus de référence |
| Classement de recherche quantifié | rang | `recall@10 >= 0.99` face à l'oracle FP32 |

Les NaN et les infinis sont interdits dans les sorties attendues. Une
tolérance ne doit jamais masquer une erreur de tokenisation, de dimension ou de
prompt.

## Règle bloquante

Toute divergence de **token IDs** bloque l'intégration des couches
d'inférence. Une couche dont les sorties dépassent les tolérances ne peut pas
être déclarée compatible, même si le classement final « semble » correct.

## Déterminisme

Le chemin scalaire FP32 est la référence. Les kernels SIMD et les
optimisations doivent être comparés au scalaire avant toute mesure de
performance. Les golden tests sont rejoués après chaque optimisation ; un gain
qui casse la parité est rejeté.

## Prompts et rôles

Les préfixes officiels `SearchQuery` et `Document` doivent être appliqués
selon la configuration du modèle. Une requête et un document indexé ne sont
jamais confondus. Les titres de documents sont préservés lorsque le prompt
officiel le prévoit.

## Limites

Les fixtures ne prouvent pas la compatibilité avec toutes les entrées
possibles. Elles prouvent la conformité sur un corpus documenté et
reproductible. Les tests de robustesse (corruption, troncature, tailles
extrêmes) sont distincts des tests de parité.
