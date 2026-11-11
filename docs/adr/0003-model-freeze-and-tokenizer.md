# ADR 0003 — Frozen model, tokenizer parity, and prompt prefixes

- **Status**: accepted
- **Date**: 2026-11-11
- **Milestone**: M1
- **Decision maker**: project maintainer
- **Supersedes**: ADR 0002 D7 (tokenization gating)
- **Superseded by**: none

## Context

ADR 0002 deferred tokenization until the official model files were available.
The model repository is now reachable and must be frozen at a precise revision
before any inference work. Tokenization is the first blocking parity gate: a
single divergent token id invalidates every downstream embedding.

M1 therefore adds:

1. a versioned model manifest with SHA-256 for every used file;
2. an exact Rust implementation of the model's tokenizer;
3. golden token ids produced by the official tokenizer, outside the product;
4. official prompt prefixes with an explicit query/document distinction.

## Decisions

### D1 — Frozen model revision and manifest

The model is frozen at revision:

`f9972ac22e696a105c482202646979e33e4b5a1e`

The manifest is `fixtures/model/manifest.json`. It records:

- model id, revision, source URL, freeze date, and license;
- every file in the repository tree with size and SHA-256;
- the role of each file;
- the model's text configuration and tokenizer inventory;
- the official prompt map;
- the SHA-256 of the committed fixtures.

Weights are **not committed**. The 1.5 GB `model.safetensors` file is a
development asset. Its SHA-256 is the Hugging Face LFS object id recorded in
the manifest; the file is not needed to validate tokenization.

Development files live in `.model-cache/<revision>/`, which is
ignored by Git. The final packer will embed the tokenizer and weights into the
versioned container.

### D2 — Tokenizer algorithm

The model's `tokenizer.json` declares a HuggingFace BPE model:

- 262,144 vocabulary entries and 514,906 merges;
- `byte_fallback = true`, `fuse_unk = true`,
  `ignore_merges = false`;
- normalizer: replace a space with the SentencePiece marker `▁`;
- pre-tokenizer: split on a space with `MergedWithPrevious`;
- post-processor: prepend `<bos>` and append `<eos>`;
- decoder: replace `▁` with a space, apply byte fallback, then fuse.

The implementation follows the official pipeline:

1. match added tokens in raw text, longest first, before normalization;
2. normalize each remaining segment by replacing spaces with `▁`;
3. start BPE from single characters, using `<0xXX>` byte tokens for
   characters absent from the vocabulary;
4. repeatedly merge the adjacent pair with the lowest merge rank, merging all
   non-overlapping occurrences of that pair;
5. apply the post-processor for `add_special_tokens = true`;
6. decode with the declared Replace, ByteFallback, and Fuse stages.

The implementation is strict. A tokenizer document whose model type,
normalizer, pre-tokenizer, post-processor, or decoder differs is rejected as
unsupported. It is better to refuse an unknown pipeline than to tokenize it
silently and produce incompatible embeddings.

### D3 — Bounds and failure behaviour

- tokenizer.json is capped at 64 MiB;
- vocabulary at 1,000,000 entries, merges at 1,000,000, added tokens at 10,000;
- encode input is capped at 1 MiB and output at 262,144 tokens;
- all vocabulary and merge ids are validated at load time;
- unknown ids during decode return `Error::Corrupt`;
- invalid UTF-8 after byte fallback returns `Error::Corrupt`;
- no `unwrap` or `expect` on external data;
- no `unsafe`.

The BPE merge loop is the scalar reference. It is deterministic and is the
oracle for any later optimization.

### D4 — Golden fixtures and reduced tokenizer

Two committed fixtures make tokenization reproducible:

- `fixtures/tokenizer/golden.json`: token ids and decoded text for
  French, English, code, accents, emoji, Greek, Japanese, Arabic, leading and
  multiple spaces, newlines and tabs, special tokens, rare Unicode byte
  fallback, a long text, and a repeated-character stress case. The ids come
  from the official `tokenizers 0.23.3` library, used only in the
  laboratory.
- `fixtures/tokenizer/mini_tokenizer.json`: a 20.9 KiB reduced
  tokenizer containing the vocabulary and merge subset needed by the golden
  corpus, preserving original token ids. It reproduces the full tokenizer
  exactly on that corpus and is not a general-purpose tokenizer.

The full 32 MiB `tokenizer.json` is not committed. The parity test
loads it from `.model-cache` when present, verifies its SHA-256 against
the manifest, and compares every golden case. In a clean checkout without the
model cache, the test reports that it is skipping the full-file comparison; the
reduced fixture test still runs.

### D5 — Prompt prefixes

The official prefixes are copied from
`config_sentence_transformers.json`:

- SearchQuery: `task: search result | query: `
- Document: `title: none | text: `
- Classification: `task: classification | query: `
- Clustering: `task: clustering | query: `
- CodeRetrieval: `task: code retrieval | query: `
- FactChecking: `task: fact checking | query: `
- SentenceSimilarity: `task: sentence similarity | query: `
- QuestionAnswering: `task: question answering | query: `

A query and an indexed document are distinct roles. The code exposes a
`Prompt` enum and `apply`, and a test compares every constant
with the frozen manifest. Document titles will be interpolated at the document
ingestion milestone; the current document prefix uses the official default
title placeholder.

### D6 — Model inventory

The manifest records the text configuration:

- 24 layers, hidden size 512, embedding dimension 768;
- 4 attention heads, 2 key/value heads, head dimension 256;
- intermediate size 2048, sliding window 512, max positions 262,144;
- vocabulary 262,144;
- RMSNorm epsilon `1e-6`;
- activation `gelu_pytorch_tanh`;
- full-attention layers at indices 5, 11, 17, and 23, with head dimension 512
  and one key/value head;
- RoPE theta 1,000,000 for full attention and 10,000 for sliding attention.

The model also declares image, audio, and video configurations. M1 does not
implement those modalities and does not claim them.

### D7 — Verification

The quality gate remains:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build --release --target aarch64-apple-darwin
cargo tree --edges normal,build,dev
otool -L target/aarch64-apple-darwin/release/seldon-vault-lite
vtool -show-build target/aarch64-apple-darwin/release/seldon-vault-lite
```

Tokenization-specific evidence:

- the reduced fixture test compares plain ids, ids with special tokens, and
  decoded text for every golden case;
- the full-file test verifies the tokenizer SHA-256 and the same cases when the
  model cache is present;
- the manifest test verifies revision, configuration, prompts, and fixture
  fingerprints;
- a development check compared the reference BPE prototype with the official
  tokenizer on 300 random Unicode strings with zero divergences.

## Consequences

### Positive

- The model and tokenizer are frozen and auditable.
- Token ids are exact on a documented corpus, unblocking inference work.
- The reduced fixture keeps continuous verification compact.
- Prompts cannot be confused between query and document roles.

### Costs and risks

- The full tokenizer is not in Git; a clean checkout cannot run the full-file
  parity test. The reduced fixture covers the committed corpus, but a general
  tokenization change still requires the model cache.
- The BPE loop is scalar and unoptimized; long inputs are capped and will be
  profiled before any acceleration.
- The tokenizer is specialized to this model revision. A tokenizer change
  requires a new manifest, new golden ids, and a new reduced fixture.
- No embedding, layer output, or final-vector parity exists yet; inference is
  the next milestone.

### Follow-up

- **M2**: text embeddings, layer-output parity, pooling, and Matryoshka
  dimensions.
- **ADR 0004 (M1/M3)**: journal format, snapshots, and transactions.
- **ADR 0005 (M6)**: HTTP contract, LAN authentication, and limits.
