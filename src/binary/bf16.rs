//! BF16 to FP32 conversion and little-endian decoding.
//!
//! BF16 weights are accepted by the model contract, but activations are FP32.
//! The conversion is exact for the value represented by the BF16 bits: BF16 is
//! simply the upper 16 bits of an FP32 value. Encoding uses round-to-nearest,
//! ties-to-even, the same rule as the reference conversion. NaN payloads are
//! quieted so that a hostile weight file cannot smuggle a signalling NaN into
//! the compute path.

use crate::error::{Error, Result};

/// Converts BF16 bits to an FP32 value.
pub const fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Converts an FP32 value to BF16 bits.
///
/// The conversion rounds to nearest, ties to even. NaN is mapped to a quiet
/// NaN while preserving the sign bit. Values above the largest BF16 finite
/// value round to infinity, which is the correct IEEE-754 behaviour.
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        let sign = ((bits >> 16) & 0x8000) as u16;
        return sign | 0x7fc0;
    }
    let least_significant_bit = (bits >> 16) & 1;
    let rounding_bias = 0x7fff + least_significant_bit;
    let rounded = bits.wrapping_add(rounding_bias);
    (rounded >> 16) as u16
}

/// Decodes little-endian BF16 bytes into FP32 values.
///
/// # Errors
///
/// Returns Error::Corrupt when the byte length is not exactly twice the
/// output length, or when the length multiplication overflows.
pub fn decode_bf16(bytes: &[u8], output: &mut [f32]) -> Result<()> {
    let expected = output
        .len()
        .checked_mul(2)
        .ok_or_else(|| Error::corrupt("BF16 byte length overflow"))?;
    if bytes.len() != expected {
        return Err(Error::corrupt(format!(
            "BF16 length mismatch: expected {expected} bytes for {} values, got {}",
            output.len(),
            bytes.len()
        )));
    }
    let (chunks, remainder) = bytes.as_chunks::<2>();
    if !remainder.is_empty() {
        return Err(Error::internal(
            "BF16 length check did not consume all bytes",
        ));
    }
    for (chunk, value) in chunks.iter().zip(output.iter_mut()) {
        *value = bf16_to_f32(u16::from_le_bytes(*chunk));
    }
    Ok(())
}

/// Encodes FP32 values as little-endian BF16 bytes.
///
/// # Errors
///
/// Returns Error::Corrupt when the output length is not exactly twice the
/// input length, or when the length multiplication overflows.
pub fn encode_bf16(values: &[f32], output: &mut [u8]) -> Result<()> {
    let expected = values
        .len()
        .checked_mul(2)
        .ok_or_else(|| Error::corrupt("BF16 byte length overflow"))?;
    if output.len() != expected {
        return Err(Error::corrupt(format!(
            "BF16 output length mismatch: expected {expected} bytes for {} values, got {}",
            values.len(),
            output.len()
        )));
    }
    let (chunks, remainder) = output.as_chunks_mut::<2>();
    if !remainder.is_empty() {
        return Err(Error::internal(
            "BF16 length check did not consume all bytes",
        ));
    }
    for (value, chunk) in values.iter().zip(chunks.iter_mut()) {
        chunk.copy_from_slice(&f32_to_bf16(*value).to_le_bytes());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values_convert_both_ways() {
        let cases = [
            (1.0f32, 0x3f80u16),
            (-1.0, 0xbf80),
            (0.0, 0x0000),
            (-0.0, 0x8000),
            (0.5, 0x3f00),
            (1.5, 0x3fc0),
            (2.0, 0x4000),
            (3.140625, 0x4049),
        ];
        for (value, bits) in cases {
            assert_eq!(f32_to_bf16(value), bits, "encoding {value}");
            assert_eq!(bf16_to_f32(bits), value, "decoding {bits:#06x}");
        }
    }

    #[test]
    fn rounding_is_ties_to_even() {
        // Midpoint between 0x3f80 (even) and 0x3f81: stays at 0x3f80.
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_4000)), 0x3f80);
        // Midpoint between 0x3f81 (odd) and 0x3f82: rounds up to 0x3f82.
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
        // Just below the midpoint rounds down.
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f81_4000)), 0x3f81);
        // Just above the midpoint rounds up.
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f81_c000)), 0x3f82);
    }

    #[test]
    fn special_values_are_preserved() {
        assert_eq!(f32_to_bf16(f32::INFINITY), 0x7f80);
        assert_eq!(f32_to_bf16(f32::NEG_INFINITY), 0xff80);
        assert_eq!(bf16_to_f32(0x7f80), f32::INFINITY);
        assert_eq!(bf16_to_f32(0xff80), f32::NEG_INFINITY);

        assert_eq!(f32_to_bf16(f32::NAN), 0x7fc0);
        assert_eq!(f32_to_bf16(-f32::NAN), 0xffc0);
        assert!(bf16_to_f32(0x7fc0).is_nan());
        assert!(bf16_to_f32(0x7fc1).is_nan());
    }

    #[test]
    fn largest_finite_value_rounds_to_infinity() {
        // f32::MAX is above the BF16 midpoint between the largest finite
        // BF16 value and infinity.
        assert_eq!(f32_to_bf16(f32::MAX), 0x7f80);
        // The largest finite BF16 value is exactly representable.
        assert_eq!(bf16_to_f32(0x7f7f), f32::from_bits(0x7f7f_0000));
        assert_eq!(f32_to_bf16(f32::from_bits(0x7f7f_0000)), 0x7f7f);
    }

    #[test]
    fn decodes_little_endian_bytes() {
        let bytes = [0x80, 0x3f, 0x80, 0xbf, 0x00, 0x40];
        let mut output = [0.0f32; 3];
        decode_bf16(&bytes, &mut output).expect("decoding must succeed");
        assert_eq!(output, [1.0, -1.0, 2.0]);
    }

    #[test]
    fn rejects_length_mismatches() {
        let mut output = [0.0f32; 2];
        let error = decode_bf16(&[0, 0, 0], &mut output).expect_err("short input must fail");
        assert!(matches!(error, Error::Corrupt(_)));
        assert!(error.to_string().contains("length mismatch"));

        let error = decode_bf16(&[0, 0], &mut output).expect_err("short output must fail");
        assert!(matches!(error, Error::Corrupt(_)));

        let mut bytes = [0u8; 3];
        let error = encode_bf16(&[1.0, 2.0], &mut bytes).expect_err("short output must fail");
        assert!(matches!(error, Error::Corrupt(_)));
    }

    #[test]
    fn round_trip_is_exact_for_representable_values() {
        let values = [0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 1.0 / 3.0, 12345.678];
        let mut bytes = vec![0u8; values.len() * 2];
        encode_bf16(&values, &mut bytes).expect("encoding must succeed");
        let mut decoded = vec![0.0f32; values.len()];
        decode_bf16(&bytes, &mut decoded).expect("decoding must succeed");
        for (value, round_trip) in values.iter().zip(decoded.iter()) {
            assert_eq!(*round_trip, bf16_to_f32(f32_to_bf16(*value)));
        }
    }
}
