//! CRC-32 checksum for journal records and snapshots.
//!
//! The journal cannot depend on an external checksum crate. This module
//! implements the reflected CRC-32/ISO-HDLC used by Ethernet and zlib:
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

/// Computes the CRC-32/ISO-HDLC checksum of a byte slice.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for byte in data {
        let index = ((crc ^ u32::from(*byte)) & 0xFF) as usize;
        crc = TABLE[index] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
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
}
