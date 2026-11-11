# Model packer (planned at milestone M12)

This directory will host the Rust `std` packer that will freeze the weights, tokenizer, processors, and manifest into a versioned and verified internal container.

## Constraints

- Rust `std` only, no third-party crates.
- No external process, no FFI.
- The container carries a magic, a version, framed sections, bounded lengths, and checksums.
- The packer is a development tool; it is not part of the binary delivered at runtime.
- Weights are never committed to Git.

## State at milestone M0

No code is present yet. The packer will be written when the weight format and numerical parity are stabilized (M1/M2), and used for final packaging (M12). Creating a stub now would give a false impression of functionality.
