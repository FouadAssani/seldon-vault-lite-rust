# Model packer — milestone M7

This directory hosts the Rust `std` packer that freezes the weights, tokenizer,
processors, and manifest into a versioned and verified internal container. It is
a development tool; it is not part of the delivered runtime binary.

## Status

M7 is in progress. The NEON execution path is implemented and tested. The
container format, packer, embedded loading path, and Mach-O size/startup
measurements are the remaining M7 packaging work. This directory does not yet
contain a working packer; creating a stub would give a false impression of
functionality.

## Constraints

- Rust `std` only, no third-party crates.
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

## Planned interface

    model-packer pack --manifest fixtures/model/manifest.json \
        --model-dir .model-cache/<revision> --output <path>
    model-packer verify <path>

The `pack` command reads the manifest, checks each selected file's size and
SHA-256, writes the container, and verifies it before reporting success. The
`verify` command parses the container, checks the header CRC-32, the catalog
digest, every section SHA-256, and the section path/offset/length invariants.
