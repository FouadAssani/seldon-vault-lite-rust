# Numerical parity and reference datasets

## Status

At milestone M1, the format and tensor core is implemented and tested, the model is frozen at revision `f9972ac22e696a105c482202646979e33e4b5a1e`, and tokenization parity is verified on a golden corpus. A format fixture, a golden tokenizer fixture, a reduced tokenizer fixture, and the model manifest exist. No model weights are loaded and no embedding or layer-output golden test exists yet. Inference is the next milestone. The tolerance values below are initial targets; they will be confirmed or adjusted with the official implementation at milestone M2, and any adjustment will be documented with its justification.

## Principle

The official `google/embeddinggemma-2` implementation serves as a reference **in the laboratory only**. It is not part of the product, is not a Cargo dependency, and is never launched by the delivered binary.

Golden tests are produced outside the product, then frozen in `fixtures/` in compact form. The product only reads these fixtures and compares its outputs.

## Model manifest

The model is frozen at revision `f9972ac22e696a105c482202646979e33e4b5a1e`. The manifest is `fixtures/model/manifest.json` (schema version 1). It records:

- the repository, exact revision, source URL, freeze date, and license;
- every file used (config, tokenizer, processors, prompts) with its size and SHA-256;
- the text configuration and tokenizer inventory;
- the provenance of the fixtures and the tool that generated them;
- the manifest schema version.

Giant weights are never committed. The weights' SHA-256 is the Hugging Face LFS object id; the file is a development asset outside Git. Model files live in `.model-cache/<revision>/`, which is ignored by Git.

## Fixtures

Rules:

- compact size; prefer excerpts and reduced tensors;
- explicit provenance: model, revision, generation script, date;
- SHA-256 fingerprint of each fixture file;
- no personal data, no secrets;
- no complete weights, no multi-gigabyte file;
- stable, versioned naming.

The `fixtures/README.md` file describes the policy. At M1:

- `fixtures/formats/mini.safetensors` proves format handling;
- `fixtures/tokenizer/golden.json` holds token ids produced by the official tokenizer;
- `fixtures/tokenizer/mini_tokenizer.json` is a reduced tokenizer that reproduces the full tokenizer on the golden corpus;
- `fixtures/model/manifest.json` records the frozen model and fixture fingerprints.

Text tensor and embedding fixtures will be added at M2; media fixtures at M4/M5.

## Tokenization parity

Token ids must match the official tokenizer exactly; any divergence blocks inference integration.

- Golden ids were generated outside the product with `tokenizers 0.23.3` and the frozen `tokenizer.json`.
- The corpus covers empty input, English, accented French, code, emoji, Greek, Japanese, Arabic, leading and multiple spaces, newlines and tabs, special tokens, rare Unicode byte fallback, a long text, and a repeated-character stress case.
- The reduced tokenizer fixture preserves original token ids and reproduces the full tokenizer exactly on the corpus.
- When `.model-cache` is present, the parity test verifies the full tokenizer's SHA-256 against the manifest and replays every case.
- A development check compared the reference BPE prototype with the official tokenizer on 300 random Unicode strings with zero divergences.

## Mandatory coverage

Golden tests must cover at least:

- accented French, English, source code;
- empty string, Unicode, emoji, control characters;
- long inputs and context limits;
- varied images: colors, grayscale, aspect ratios, rotations;
- WAV excerpts: silence, short sound, long sound, speech, music, environment;
- dimensions 768, 512, 256, and 128;
- queries and indexed documents, with the official prefixes.

## Tolerances

| Object | Comparison | Initial tolerance |
|---|---|---|
| Token IDs | exact equality | no divergence accepted |
| Dimensions and shapes | exact equality | no divergence accepted |
| Intermediate FP32 tensors | absolute + relative | `atol <= 1e-5`, `rtol <= 1e-4` for values of magnitude greater than `1e-3` |
| Final FP32 embeddings | cosine + absolute | `cos >= 0.9999`, `atol <= 1e-4` after L2 normalization |
| Quantized embeddings (future) | cosine + recall | `cos >= 0.999`, `recall@10 >= 0.99` against the FP32 oracle |
| FP32 search ranking | rank | identical top-1, `recall@10 = 1.0` on the reference corpus |
| Quantized search ranking | rank | `recall@10 >= 0.99` against the FP32 oracle |

NaN and infinities are forbidden in expected outputs. A tolerance must never hide a tokenization, dimension, or prompt error.

## Blocking rule

Any **token ID** divergence blocks the integration of inference layers. A layer whose outputs exceed tolerances cannot be declared compatible, even if the final ranking "seems" correct.

## Determinism

The FP32 scalar path is the reference. SIMD kernels and optimizations must be compared with the scalar path before any performance measurement. Golden tests are replayed after each optimization; a gain that breaks parity is rejected.

## Prompts and roles

The official `SearchQuery` and `Document` prefixes are copied from `config_sentence_transformers.json` into `model::prompt`. A query and an indexed document are never confused; `Prompt::is_query` makes the distinction explicit. A test compares every constant with the frozen manifest. Document titles are preserved when the official prompt provides for it; the current document prefix uses the official default title placeholder.

## Limits

Fixtures do not prove compatibility with all possible inputs. They prove compliance on a documented and reproducible corpus. Robustness tests (corruption, truncation, extreme sizes) are distinct from parity tests.
