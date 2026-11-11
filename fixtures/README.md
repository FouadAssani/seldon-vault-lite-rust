# Fixtures and golden tests

This directory contains compact, versioned fixtures used by reproducible
tests. Model weights and multi-gigabyte files are never committed.

## State at milestone M1

Four fixtures are present:

- `formats/mini.safetensors`: a 281-byte deterministic Safetensors
  file with F32, BF16, and zero-sized tensors. It proves format handling only;
  it is not a model parity fixture.
- `tokenizer/golden.json`: token ids and decoded text produced by the
  official `tokenizers 0.23.3` library for a documented corpus.
- `tokenizer/mini_tokenizer.json`: a 20.9 KiB reduced tokenizer that
  reproduces the full tokenizer exactly on the golden corpus, with original
  token ids preserved. It is not a general tokenizer.
- `model/manifest.json`: the frozen model revision, every file's size
  and SHA-256, the text configuration, the prompt map, and fixture fingerprints.

No embedding, image, or audio golden test exists yet. Do not fabricate
"plausible" fixtures: they must come from a documented reference and be
verifiable.

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
├── model/
│   ├── README.md
│   └── manifest.json
├── tokenizer/
│   ├── README.md
│   ├── golden.json
│   └── mini_tokenizer.json
├── tensors/               # intermediate FP32 outputs (M2, pending)
├── embeddings/            # final vectors, MRL dimensions (M2, pending)
├── images/                # small PNG/JPEG images (M4, pending)
└── audio/                 # short WAV excerpts (M5, pending)
```

## Fingerprints

| Fixture | SHA-256 |
|---|---|
| `formats/mini.safetensors` | `caca3b030faeb4ded52275d54c2fca0f2ee0d6b1aeeb8ab1819e0785b1f89ca7` |
| `tokenizer/golden.json` | recorded in `model/manifest.json` |
| `tokenizer/mini_tokenizer.json` | `2f9c9e6a563da8ff9142f620a3dc4400b8a0cc8147a878abeb6bb76aced4fe3b` |
| `model/manifest.json` | self-describing; fixture hashes are inside |

## Model manifest

The model manifest records the repository revision
`f9972ac22e696a105c482202646979e33e4b5a1e`, every used file, its
size and SHA-256, the text configuration, the prompt map, and a schema version.

Model files are development assets outside Git, in
`.model-cache/<revision>/`. The weights' SHA-256 is the Hugging Face
LFS object id; the 1.5 GB file is not committed.
