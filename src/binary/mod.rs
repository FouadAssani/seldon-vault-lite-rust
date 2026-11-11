//! Binary primitives: bounded readers, checksums, numeric conversion, and
//! Safetensors.
//!
//! These modules are the only place where external bytes are interpreted.
//! They validate lengths, offsets, endianness, and limits before any
//! allocation or indexing.

pub mod bf16;
pub mod crc32;
pub mod reader;
pub mod safetensors;
pub mod sha256;

pub use reader::{ByteReader, Endian, read_file_bounded};
pub use safetensors::{
    MAX_HEADER_BYTES, MAX_METADATA_ENTRIES, MAX_TENSORS, SafetensorsFile, TensorEntry,
};
