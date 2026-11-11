# Frozen model manifest

## File

`manifest.json` is the versioned record of the frozen model. It is
generated outside the product and committed. Its current SHA-256 is:

`ceccff1d89a25636fd4f7e684dc25767d02561b3a0acfad2daeb949c6219f4d0`

## Contents

- schema version;
- model id `google/embeddinggemma-2`;
- exact revision `f9972ac22e696a105c482202646979e33e4b5a1e`;
- source URL, freeze date, and license;
- every file in the repository tree with size, SHA-256, role, and a
  `committed: false` marker;
- tokenizer inventory: vocabulary size, merge count, byte fallback,
  normalizer, pre-tokenizer, post-processor, added tokens, and special ids;
- text configuration: layers, hidden size, embedding dimension, heads,
  sliding window, maximum positions, RMSNorm epsilon, activation, layer types,
  per-layer overrides, and RoPE parameters;
- official prompt map;
- SHA-256 of the committed tokenizer fixtures.

## Model files

Model files are development assets and are not committed. They live in:

```text
.model-cache/<revision>/
```

The directory is ignored by Git. Fetch a file with:

```bash
REV=f9972ac22e696a105c482202646979e33e4b5a1e
curl -sSL "https://huggingface.co/google/embeddinggemma-2/resolve/$REV/config.json" \
  -o .model-cache/$REV/config.json
```

The weights file is about 1.5 GB. Its SHA-256 in the manifest is the Hugging
Face LFS object id:

`197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79`

The file is not needed to validate tokenization. The final packer will embed
the tokenizer and weights into the versioned container at the packaging
milestone.

## License

The model repository declares the Apache-2.0 license. The weights remain
subject to the model's official license; no weights are redistributed in this
repository.
