# Tokenizer and prompts

This document describes the exact text input contract implemented at milestone
M1. ADR 0003 is the decision record; this document is the operational
reference.

## Frozen model

The tokenizer belongs to `google/embeddinggemma-2`, frozen at revision:

`f9972ac22e696a105c482202646979e33e4b5a1e`

The model manifest is `fixtures/model/manifest.json`. It records the
revision, every file's size and SHA-256, the text configuration, the prompt map,
and the fixture fingerprints.

Model files are development assets and live outside Git in
`.model-cache/<revision>/`. Weights are never committed.

## Pipeline

The model tokenizer is a HuggingFace BPE with byte fallback:

1. **Added tokens.** Special tokens such as `<bos>`, `<eos>`,
   `<|image|>`, and `<|audio|>` are matched in raw text,
   longest first, before normalization.
2. **Normalization.** Each remaining segment replaces every space with the
   SentencePiece marker `▁`.
3. **Initial symbols.** Each character becomes its vocabulary token. A
   character absent from the vocabulary becomes one `<0xXX>` byte
   token per UTF-8 byte.
4. **BPE merges.** The adjacent pair with the lowest merge rank is merged; all
   non-overlapping occurrences of that pair are merged in one pass. The loop
   stops when no adjacent pair has a merge rank.
5. **Post-processing.** `<bos>` is prepended and `<eos>` is
   appended when special tokens are requested.

The pre-tokenizer declared by the model splits on spaces with
`MergedWithPrevious`. Because normalization already replaces every
space, the tokenizer operates on the normalized segment directly.

## Loading

`Tokenizer::load(path)` reads a bounded file and calls
`Tokenizer::from_json_str`. `Tokenizer::from_json` accepts an
already parsed JSON value.

The loader validates the pipeline instead of guessing:

- model type must be `BPE`;
- byte fallback and fuse_unk must be enabled;
- ignore_merges must be disabled;
- continuing_subword_prefix, end_of_word_suffix, and dropout must be null;
- the normalizer must replace a space with `▁`;
- the pre-tokenizer must split on a space with `MergedWithPrevious`;
- the post-processor must be `<bos> A <eos>`;
- the decoder must be Replace(`▁` to space), ByteFallback, Fuse;
- every merge part and result must exist in the vocabulary;
- every byte token must exist;
- every added token id must exist.

An unknown pipeline returns `Error::Unsupported`. An inconsistent
document returns `Error::Corrupt`.

## API

```rust
let tokenizer = Tokenizer::load(path)?;
let ids = tokenizer.encode("Bonjour, le monde !", false)?;
let ids_with_special = tokenizer.encode("Bonjour, le monde !", true)?;
let text = tokenizer.decode(&ids, true)?;
```

Useful accessors:

| Method | Meaning |
|---|---|
| `token_id(token)` | vocabulary lookup |
| `token(id)` | reverse lookup |
| `vocab_size()` | number of vocabulary entries |
| `max_token_id()` | largest id present |
| `bos_id()`, `eos_id()` | special ids |
| `unk_id()`, `pad_id()` | unknown and padding ids |
| `is_special_id(id)` | added special-token predicate |

## Limits

| Limit | Value |
|---|---|
| tokenizer.json file | 64 MiB |
| vocabulary entries | 1,000,000 |
| merges | 1,000,000 |
| added tokens | 10,000 |
| encode input | 1 MiB |
| encoded tokens | 262,144 |

The 262,144-token cap matches the model's declared maximum positions. Larger
inputs are rejected with `Error::LimitExceeded` before the model is
called.

## Golden corpus and reduced fixture

`fixtures/tokenizer/golden.json` contains token ids and decoded text
produced by the official `tokenizers 0.23.3` library, outside the
product. It covers:

- empty input;
- English and French, including accents and diacritics;
- Rust and Python code;
- emoji, Greek, Japanese, and Arabic;
- leading and multiple spaces;
- newlines and tabs;
- special tokens, including multimodal markers;
- rare Unicode scalar values that exercise byte fallback;
- a long text and a repeated-character stress case.

`fixtures/tokenizer/mini_tokenizer.json` is a 20.9 KiB reduced
tokenizer containing the vocabulary and merge subset required by the corpus,
with original token ids preserved. It reproduces the full tokenizer exactly on
the corpus. It is not a general tokenizer and must not be used for arbitrary
text outside the tests.

The full 32 MiB `tokenizer.json` is not committed. The parity test
loads it from `.model-cache` when present, verifies its SHA-256 against
the manifest, and compares every golden case. Without the cache, the full-file
comparison is skipped and the reduced-fixture test still runs.

## Prompts

The official prompt prefixes are exposed by `model::prompt`:

- SearchQuery: `task: search result | query: `
- Document: `title: none | text: `
- Classification: `task: classification | query: `
- Clustering: `task: clustering | query: `
- CodeRetrieval: `task: code retrieval | query: `
- FactChecking: `task: fact checking | query: `
- SentenceSimilarity: `task: sentence similarity | query: `
- QuestionAnswering: `task: question answering | query: `

`Prompt::is_query` distinguishes a query from a document.
`prompt::apply` prepends the prefix without copying the text twice.

## Limitations

- The tokenizer is specialized to this model revision.
- The reduced fixture is not a general tokenizer.
- The full-file parity test requires the development model cache.
- The BPE loop is the scalar reference and has not been optimized.
- No embedding or layer-output parity exists yet; inference is the next
  milestone.
- Image, audio, and video tokens are recognized as added tokens but their
  modalities are not implemented at M1.
