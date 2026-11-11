# Fixtures and golden tests

This directory will contain the project's compact, versioned golden tests.

## State at milestone M0

No real fixture is present yet. Model weights are not frozen and the official implementation has not yet produced a reference. Do not fabricate "plausible" fixtures: they must come from the official implementation and be verifiable.

## Rules

- Each fixture is compact and versioned.
- Each file has a SHA-256 fingerprint and explicit provenance.
- No complete weights or multi-gigabyte file.
- No secrets, no personal data.
- Tokenization, tensor, and embedding fixtures are separate.
- Binary files are small and reproducible.
- Fixtures are generated outside the product, never by the delivered binary.

## Planned structure

```text
fixtures/
├── README.md
├── manifest.json          # provenance, model revision, SHA-256
├── tokenizer/             # expected token IDs (M1/M2)
├── tensors/               # intermediate FP32 outputs (M1/M2)
├── embeddings/            # final vectors, MRL dimensions (M2)
├── images/                # small PNG/JPEG images (M4)
└── audio/                 # short WAV excerpts (M5)
```

The manifest and fixtures will be added as milestones progress. The exact manifest format will be defined at milestone M1.
