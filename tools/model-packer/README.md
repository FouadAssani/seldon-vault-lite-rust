# Model packer (prévu au jalon M12)

Ce répertoire accueillera le packer Rust `std` qui figera les poids, le
tokenizer, les processeurs et le manifeste dans un conteneur interne versionné
et vérifié.

## Contraintes

- Rust `std` uniquement, aucune crate tierce.
- Aucun processus externe, aucun FFI.
- Le conteneur porte une magie, une version, des sections cadrées, des
  longueurs bornées et des checksums.
- Le packer est un outil de développement ; il ne fait pas partie du binaire
  livré au runtime.
- Les poids ne sont jamais commités dans Git.

## État au jalon M0

Aucun code n'est encore présent. Le packer sera écrit lorsque le format des
poids et la parité numérique seront stabilisés (M1/M2), et utilisé pour le
packaging final (M12). Créer un stub maintenant donnerait une fausse impression
de fonctionnalité.
