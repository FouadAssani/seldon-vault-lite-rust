# Fixtures and golden tests

This directory contains compact, versioned fixtures used by reproducible
tests. Model weights and multi-gigabyte files are never committed.

## State at milestone M4 (image path)

The M1/M2/M3 fixtures remain, and five PNG inputs, five JPEG files, five
reference PNGs, and two image reference fixtures have been added. They are
described in media/README.md and model/README.md.

- `formats/mini.safetensors`: a 281-byte deterministic Safetensors file with
  F32, BF16, and zero-sized tensors. It proves format handling only; it is not
  a model parity fixture.
- `tokenizer/golden.json`: token ids and decoded text produced by the
  official `tokenizers 0.23.3` library for a documented corpus.
- `tokenizer/mini_tokenizer.json`: a 20.9 KiB reduced tokenizer that
  reproduces the full tokenizer exactly on the golden corpus, with original
  token ids preserved. It is not a general tokenizer.
- `model/manifest.json`: the frozen model revision, every file's size and
  SHA-256, the text configuration, the prompt map, and fixture fingerprints.
- `model/text_golden.json`: eight official text embeddings at 768, 512, 256,
  and 128 dimensions, plus a layer probe with the embedding output, all 24
  layer outputs, and the final projected state.
- `model/tiny_text/config.json` and
  `model/tiny_text/model.safetensors`: a complete 3-layer EmbeddingGemma 2
  text model generated with the official implementation (56 tensors, 10,315
  parameters).
- `model/tiny_text/golden.json`: the official layer outputs and normalized
  embeddings for the tiny model.
- `media/preprocess_golden.json`: official resized bytes, padded patch
  values, positions, and SHA-256 fingerprints for nine images at the 70 or
  280 token budget, including rotations and non-square aspect ratios.
- `model/image_golden.json`: patch-embedder and 16 vision layer probes,
  projected vision soft tokens, and the final normalized image embedding for
  seven images, including a rotation and a non-square aspect ratio.

The PNG and JPEG inputs prove decoder behavior. The preprocessing and image
embedding goldens come from the official reference tensor path and are
validated by the Rust tests. The JPEG files were generated with macOS sips and
libjpeg-turbo cjpeg, and the reference PNGs were rendered with sips; these
tools are laboratory-only and are never run by the product. Audio fixtures do
not exist yet. Do not fabricate "plausible" fixtures: they must come from a
documented reference and be verifiable.

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
│   ├── manifest.json
│   ├── text_golden.json
│   ├── image_golden.json
│   └── tiny_text/
│       ├── config.json
│       ├── golden.json
│       └── model.safetensors
├── tokenizer/
│   ├── README.md
│   ├── golden.json
│   └── mini_tokenizer.json
├── media/
│   ├── README.md
│   ├── preprocess_golden.json
│   ├── rgb_2x2.png
│   ├── rgba_2x2.png
│   ├── gray_2x2.png
│   ├── gradient_16x16.png
│   ├── quadrants_16x16.png
│   ├── wide_32x16.png
│   ├── tall_16x32.png
│   ├── quadrants_16x16_rot90.png
│   ├── quadrants_16x16_rot180.png
│   ├── rgb_2x2.jpg
│   ├── gray_2x2.jpg
│   ├── gradient_16x16.jpg
│   ├── quadrants_16x16.jpg
│   ├── quadrants_16x16_444.jpg
│   ├── rgb_2x2_ref.png
│   ├── gray_2x2_ref.png
│   ├── gradient_16x16_ref.png
│   ├── quadrants_16x16_ref.png
│   └── quadrants_16x16_444_ref.png
└── audio/                 # short WAV excerpts (M5, pending)
```

## Fingerprints

| Fixture | SHA-256 |
|---|---|
| `formats/mini.safetensors` | `caca3b030faeb4ded52275d54c2fca0f2ee0d6b1aeeb8ab1819e0785b1f89ca7` |
| `tokenizer/golden.json` | recorded in `model/manifest.json` |
| `tokenizer/mini_tokenizer.json` | `2f9c9e6a563da8ff9142f620a3dc4400b8a0cc8147a878abeb6bb76aced4fe3b` |
| `model/text_golden.json` | `7bcb905b84cefcedffe14cb84ae3532cee3a2107e16a5c0ba482c2cbdc0efb98` |
| `model/tiny_text/config.json` | `8aa35d31bdee5f1ff3284d3ed08b64377c1e4675a471d7f2d89ef4a89a9566f1` |
| `model/tiny_text/model.safetensors` | `2daec622d6d3be44a339158678939f9b739a3b21c8be1dce8f6f7c38fb90caa6` |
| `model/tiny_text/golden.json` | `28a3800304ac2d0ee765df72f6f681952b9dce5e1086eb12d7005df28421bf94` |
| `model/image_golden.json` | `f7aa04357c0b8a3393d83958f88901b3b6b29d77f2ef2499568a4fd8970a8482` |
| `media/preprocess_golden.json` | `5f4d8628d7d4dd7c173b0a33bc67315db929955be70fb0ac56b876b3be84c98c` |
| `media/rgb_2x2.png` | `edbfc60195995fc33f65a2d3cfaa65fffa3c0c68246af114c7076fec71e12c89` |
| `media/rgba_2x2.png` | `fa6cf6c3de0c8a91ab7bb255e4b7b77d2397cd4f21732b7168dd6ec375bd0db8` |
| `media/gray_2x2.png` | `e1777c3fabfd5ed4697f373347c367393cc219b80c7d6e4c609a41afb24be198` |
| `media/gradient_16x16.png` | `87b57cb2838c25880a9695ddc42063c44d32c31b8087fb05c8c74f8e90bc6428` |
| `media/quadrants_16x16.png` | `f1b08c73bd0e72947997255aed0ed0839fdd6fe7cbd78e02c5ba69e74da41409` |
| `media/wide_32x16.png` | `46b1e8c417e4f743626e1a2fb76754aed019b495bb06646e634237b084cd1910` |
| `media/tall_16x32.png` | `d818b602d8bf45b1d8e6f5ba945799d1fb853287fed6ffcea2ff1ed8c8ac70f0` |
| `media/quadrants_16x16_rot90.png` | `daf2e54eae79cdbf0bbe3db759c5364f59a7bea151a8c9f39eec2568d8afd091` |
| `media/quadrants_16x16_rot180.png` | `f10c64c2718cbe9725f19a34b24854fec74a253e1fc614260acabddd42f6fcb3` |
| `media/rgb_2x2.jpg` | `af0636595e0389ddc8256659b7d2eef2f57c3790d384064ec6e38a01eaa88819` |
| `media/gray_2x2.jpg` | `7be63114cb8f9d9e75ab96ebf2111760a45190ff3df2a3aa78f9623a586f053a` |
| `media/gradient_16x16.jpg` | `9d2d323f1f189077859da9eda1b4a77f61df016df866086f4a34921a484d37a7` |
| `media/quadrants_16x16.jpg` | `675e3a5ee9111d2fbd7a3ced0630016cb32c062cfef7221ef6c303816fa02d4c` |
| `media/quadrants_16x16_444.jpg` | `70183b02d9a1d50b5b88e1a2cc604b6085b176d1b6773e092236b8e48b7bec99` |
| `media/rgb_2x2_ref.png` | `0fc0cfb64f28b800593351b05e90bc6590babf5ebbc96cea6d836f9312319d6c` |
| `media/gray_2x2_ref.png` | `18fb01108b3eef02406d6b8ea5a293cdc8d5d73631dbf474f2396ae6e2baf473` |
| `media/gradient_16x16_ref.png` | `ef42a5c090545b940480d5ec4e5738466a6ee0f390047008953100e793a4ba7a` |
| `media/quadrants_16x16_ref.png` | `034f884b1cc8d40ee7985b41ba010eda08d529ccefd1fd54652d3445c8dea2a8` |
| `media/quadrants_16x16_444_ref.png` | `108d2c8a4d2be8f302ef0733f01184cb7ab75a397da7f0ba9945b0ce0f83d7fc` |
| `model/manifest.json` | self-describing; fixture hashes are inside |

## Model manifest

The model manifest records the repository revision
`f9972ac22e696a105c482202646979e33e4b5a1e`, every used file, its size and
SHA-256, the text configuration, the prompt map, and a schema version.

Model files are development assets outside Git, in
`.model-cache/<revision>/`. The weights' SHA-256 is the Hugging Face LFS
object id; the 1.5 GB file is not committed.
