# Fixtures et golden tests

Ce répertoire contiendra les golden tests compacts et versionnés du projet.

## État au jalon M0

Aucune fixture réelle n'est encore présente. Les poids du modèle ne sont pas
figés et l'implémentation officielle n'a pas encore produit de référence. Ne
pas fabriquer de fixtures « plausibles » : elles doivent provenir de
l'implémentation officielle et être vérifiables.

## Règles

- Chaque fixture est compacte et versionnée.
- Chaque fichier a une empreinte SHA-256 et une provenance explicite.
- Aucun poids complet ni fichier de plusieurs gigaoctets.
- Aucun secret, aucune donnée personnelle.
- Les fixtures de tokenisation, de tenseurs et d'embeddings sont séparées.
- Les fichiers binaires sont petits et reproductibles.
- Les fixtures sont générées hors du produit, jamais par le binaire livré.

## Structure prévue

```text
fixtures/
├── README.md
├── manifest.json          # provenance, révision du modèle, SHA-256
├── tokenizer/             # token IDs attendus (M1/M2)
├── tensors/               # sorties intermédiaires FP32 (M1/M2)
├── embeddings/            # vecteurs finaux, dimensions MRL (M2)
├── images/                # petites images PNG/JPEG (M4)
└── audio/                 # extraits WAV courts (M5)
```

Le manifeste et les fixtures seront ajoutés au fur et à mesure des jalons. Le
format exact du manifeste sera défini au jalon M1.
