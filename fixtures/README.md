# Fixtures and golden tests

This directory contains compact, versioned fixtures used by reproducible
tests. Model weights and multi-gigabyte files are never committed.

## State at milestone M1

One format fixture is present:

- `formats/mini.safetensors`: a 281-byte deterministic Safetensors
  file with F32, BF16, and zero-sized tensors. It proves format handling only;
  it is not a model parity fixture.

No tokenizer, embedding, image, or audio golden test exists yet. They require
the frozen model and the official implementation. Do not fabricate "plausible"
fixtures: they must come from a documented reference and be verifiable.

## Rules

- Each fixture is compact and versioned.
- Each file has a SHA-256 fingerprint and explicit provenance.
- No complete weights or multi-gigabyte file.
- No secrets, no personal data.
- Tokenization, tensor, and embedding fixtures are separate.
- Binary files are small and reproducible.
- Fixtures are generated outside the product, never by the delivered binary.

## Structure

```text
fixtures/
├── README.md
├── formats/
│   ├── README.md
│   └── mini.safetensors
├── tokenizer/             # expected token IDs (M1/M2, pending model freeze)
├── tensors/               # intermediate FP32 outputs (M1/M2, pending)
├── embeddings/            # final vectors, MRL dimensions (M2, pending)
├── images/                # small PNG/JPEG images (M4, pending)
└── audio/                 # short WAV excerpts (M5, pending)
```

## Format fixture provenance

`formats/mini.safetensors` was generated outside the product with a
short Python script using only `json`, `struct`, and
`hashlib`. The generation recipe is documented in
`formats/README.md`.

SHA-256:
`caca3b030faeb4ded52275d54c2fca0f2ee0d6b1aeeb8ab1819e0785b1f89ca7`

## Model manifest

The model manifest will record the repository revision, every used file, its
size and SHA-256, the fixture provenance, and a schema version. It will be
added when the model is frozen, before any inference claim.
