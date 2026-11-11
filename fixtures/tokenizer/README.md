# Tokenizer fixtures

## Files

| File | Size | SHA-256 |
|---|---|---|
| `golden.json` | 23,319 bytes | `10324582de2ca345117f8edbb82984df1434493be09d91b1e9bf4b2082b3ce42` |
| `mini_tokenizer.json` | 20,923 bytes | `2f9c9e6a563da8ff9142f620a3dc4400b8a0cc8147a878abeb6bb76aced4fe3b` |

## Golden corpus

`golden.json` was generated outside the product with the official
`tokenizers 0.23.3` library and the frozen
`tokenizer.json` (SHA-256
`4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4`).

It contains, for each case:

- `name`, `text`;
- `ids` without special tokens;
- `ids_with_special` with `<bos>` and `<eos>`;
- `decoded` text after decoding with special tokens skipped.

Coverage:

- empty input;
- English and French, including accents and diacritics;
- Rust and Python code;
- emoji, Greek, Japanese, and Arabic;
- leading and multiple spaces;
- newlines and tabs;
- special tokens, including multimodal markers;
- rare Unicode scalar values that exercise byte fallback;
- a long text and a repeated-character stress case.

## Reduced tokenizer

`mini_tokenizer.json` is a reduced copy of the full tokenizer:

- the vocabulary contains only tokens that are substrings of the normalized
  corpus, all single characters used by the corpus, all 256 byte tokens, all
  special tokens, and the space marker;
- the merge list preserves the original order and contains only merges whose
  parts and result are in the selected vocabulary;
- original token ids are preserved, so ids are sparse;
- the normalizer, pre-tokenizer, post-processor, decoder, and added tokens are
  copied unchanged.

This fixture reproduces the full tokenizer exactly on the golden corpus. It is
not a general tokenizer and must not be used for arbitrary text.

## Reproducing

The full tokenizer file is a development asset, not committed:

```bash
REV=f9972ac22e696a105c482202646979e33e4b5a1e
mkdir -p .model-cache/$REV
curl -sSL "https://huggingface.co/google/embeddinggemma-2/resolve/$REV/tokenizer.json" \
  -o .model-cache/$REV/tokenizer.json
```

`.model-cache/` is ignored by Git. Python and `tokenizers`
are laboratory tools only; they are not part of the product and are not
invoked by the binary or the test suite.

The parity test in `tests/tokenizer.rs` loads the full file when it is
present, verifies its SHA-256 against `fixtures/model/manifest.json`,
and replays every golden case. Without the file, the test reports that it is
skipping the full-file comparison; the reduced-fixture test still runs.

## Why two fixtures

The full tokenizer is 32 MiB and cannot be committed as a compact fixture. The
golden ids are the authoritative output; the reduced tokenizer is the smallest
committed input that reproduces those ids exactly. Together they make the
tokenizer test reproducible without committing model-scale data.
