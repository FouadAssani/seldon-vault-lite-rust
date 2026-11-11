# Numerical parity and reference datasets

## Status

At milestone M1, the format and tensor core is implemented and tested, and one format fixture exists. No model weights are frozen or loaded yet, and no model parity golden test exists. Tokenization and inference remain gated on the frozen model manifest. The tolerance values below are initial targets; they will be confirmed or adjusted with the official implementation at milestone M2, and any adjustment will be documented with its justification.

## Principle

The official `google/embeddinggemma-2` implementation serves as a reference **in the laboratory only**. It is not part of the product, is not a Cargo dependency, and is never launched by the delivered binary.

Golden tests are produced outside the product, then frozen in `fixtures/` in compact form. The product only reads these fixtures and compares its outputs.

## Model manifest

Before any integration, a versioned manifest must record:

- the repository and **exact revision** of the model;
- the list of files used (config, tokenizer, processors, weights, prompts);
- the size and **SHA-256** of each file;
- the provenance of the fixtures and the tool that generated them;
- the manifest schema version.

Giant weights are never committed. The manifest itself is versioned.

## Fixtures

Rules:

- compact size; prefer excerpts and reduced tensors;
- explicit provenance: model, revision, generation script, date;
- SHA-256 fingerprint of each fixture file;
- no personal data, no secrets;
- no complete weights, no multi-gigabyte file;
- stable, versioned naming.

The `fixtures/README.md` file describes the policy. A format fixture is present at M1 (`fixtures/formats/mini.safetensors`); model parity fixtures will be added after the model freeze, then text fixtures at M2 and media fixtures at M4/M5.

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

The official `SearchQuery` and `Document` prefixes must be applied according to the model configuration. A query and an indexed document are never confused. Document titles are preserved when the official prompt provides for it.

## Limits

Fixtures do not prove compatibility with all possible inputs. They prove compliance on a documented and reproducible corpus. Robustness tests (corruption, truncation, extreme sizes) are distinct from parity tests.
