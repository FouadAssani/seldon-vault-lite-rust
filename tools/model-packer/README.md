# Model packer — milestone M7

This directory hosts the Rust `std` packer that freezes the weights, tokenizer,
processors, and manifest into a versioned and verified internal container. It is
a development tool; it is not part of the delivered runtime binary.

## Status

The container format (`src/bundle`), the streaming writer, and the packer
commands are implemented. The packer was run on the frozen revision and
produced 9 sections and 1,525,784,026 bytes; `verify` re-parsed the file,
checked the header CRC-32 and catalog digest, and checked every section
SHA-256. Embedded loading and the Mach-O size/startup measurements remain in
progress.

## Constraints

- Rust `std` only, no third-party crates. The only dependency is the
  first-party `seldon-vault-lite` library, used for the bundle format, SHA-256,
  and strict JSON.
- No external process, no FFI.
- The container carries a magic, a version, a bounded header, a sorted section
  table, per-section SHA-256, a header CRC-32, a catalog digest, and explicit
  caps on header size, section count, path length, section length, and total
  bundle size.
- Weights are never committed to Git.
- The packer validates the frozen manifest hashes before writing.
- The final binary embeds the container with `include_bytes!` (or an
  equivalent Rust mechanism) and loads model files from the embedded bytes.
  Only user data, logs, and snapshots remain external files.

## Interface

    cargo build --manifest-path tools/model-packer/Cargo.toml --release

    model-packer pack --manifest fixtures/model/manifest.json \
        --model-dir .model-cache/<revision> --output <path>
    model-packer verify <path>
    model-packer list <path>
    model-packer version

The `pack` command reads the manifest, checks each selected file's size and
SHA-256, writes the container to a temporary file, calls `sync_all`, renames it
atomically, then reads it back and verifies the structure and every section
digest. The `verify` command parses the container, checks the header CRC-32,
the catalog digest, every section SHA-256, and the section path/offset/length
invariants. The `list` command prints the role, length, and path of every
section.

The frozen revision is
`f9972ac22e696a105c482202646979e33e4b5a1e`. The packer includes the required
`config.json`, `model.safetensors`, and `tokenizer.json`, plus the official
prompt, processor, pooling, normalization, and tokenizer model files when the
manifest lists them.
