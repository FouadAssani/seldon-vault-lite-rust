//! CRC-32 checksum shared by storage and media parsers.
//!
//! The product cannot depend on an external checksum crate. This module
//! implements the reflected CRC-32/ISO-HDLC used by Ethernet, zlib, and PNG:
//! polynomial 0xEDB88320, initial value 0xFFFFFFFF, final XOR 0xFFFFFFFF.
//! It detects torn writes, bit flips, and accidental record mixing; it is not
//! a cryptographic hash. SHA-256 remains available for fingerprints.

/// Lookup table for the reflected CRC-32 polynomial.
///
/// The table is generated at compile time so the runtime path is a simple
/// byte loop without any initialization race.
const fn build_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut index = 0usize;
    while index < 256 {
        let mut value = index as u32;
        let mut bit = 0u32;
        while bit < 8 {
            value = if value & 1 != 0 {
                0xEDB8_8320 ^ (value >> 1)
            } else {
                value >> 1
            };
            bit += 1;
        }
        table[index] = value;
        index += 1;
    }
    table
}

/// Compile-time CRC-32 lookup table.
static TABLE: [u32; 256] = build_table();

/// Streaming CRC-32 hasher.
///
/// PNG validates a checksum over the chunk type and data, which are separate
/// slices in the parser. Streaming avoids concatenating them.
#[derive(Debug, Clone)]
pub struct Crc32 {
    crc: u32,
}

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

impl Crc32 {
    /// Creates a hasher in its initial state.
    pub fn new() -> Self {
        Self { crc: 0xFFFF_FFFF }
    }

    /// Adds bytes to the checksum.
    pub fn update(&mut self, data: &[u8]) {
        for byte in data {
            let index = ((self.crc ^ u32::from(*byte)) & 0xFF) as usize;
            self.crc = TABLE[index] ^ (self.crc >> 8);
        }
    }

    /// Finishes the checksum.
    pub fn finalize(self) -> u32 {
        self.crc ^ 0xFFFF_FFFF
    }
}

/// Computes the CRC-32/ISO-HDLC checksum of a byte slice.
pub fn crc32(data: &[u8]) -> u32 {
    let mut hasher = Crc32::new();
    hasher.update(data);
    hasher.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_has_the_standard_value() {
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn check_value_matches_the_standard_vector() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn quick_brown_fox_matches_the_known_value() {
        assert_eq!(
            crc32(b"The quick brown fox jumps over the lazy dog"),
            0x414F_A339
        );
    }

    #[test]
    fn single_bit_changes_are_detected() {
        let original = crc32(b"seldon-vault-lite");
        for bit in 0..8 {
            let mut mutated = *b"seldon-vault-lite";
            mutated[0] ^= 1 << bit;
            assert_ne!(crc32(&mutated), original, "bit {bit}");
        }
    }

    #[test]
    fn streaming_matches_one_shot() {
        let data = b"streaming input split at every possible boundary";
        let expected = crc32(data);
        for split in 0..=data.len() {
            let mut hasher = Crc32::new();
            hasher.update(&data[..split]);
            hasher.update(&data[split..]);
            assert_eq!(hasher.finalize(), expected, "split at {split}");
        }
    }
}
