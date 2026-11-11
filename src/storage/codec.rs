//! Bounded little-endian binary codec used by the journal and snapshots.
//!
//! The storage format is written by the product itself but is still treated as
//! hostile when read: every length is capped, every range uses checked
//! arithmetic, and strings are validated as UTF-8. Encoders also enforce the
//! same cap so a live write cannot create a record that replay would reject.

use crate::error::{Error, Result};

/// Encodes values into a bounded little-endian byte buffer.
#[derive(Debug)]
pub struct Encoder {
    bytes: Vec<u8>,
    max_bytes: usize,
}

impl Encoder {
    /// Creates an encoder that refuses to grow above max_bytes.
    pub fn new(max_bytes: usize) -> Self {
        Self {
            bytes: Vec::new(),
            max_bytes,
        }
    }

    /// Number of bytes written so far.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Returns true when nothing has been written.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Finishes the encoding and returns the buffer.
    pub fn finish(self) -> Vec<u8> {
        self.bytes
    }

    fn reserve(&mut self, additional: usize) -> Result<()> {
        let total = self
            .bytes
            .len()
            .checked_add(additional)
            .ok_or_else(|| Error::limit_exceeded("encoded storage record length overflow"))?;
        if total > self.max_bytes {
            return Err(Error::limit_exceeded(format!(
                "encoded storage record would be {total} bytes, above the {}-byte cap",
                self.max_bytes
            )));
        }
        Ok(())
    }

    /// Writes one byte.
    pub fn write_u8(&mut self, value: u8) -> Result<()> {
        self.reserve(1)?;
        self.bytes.push(value);
        Ok(())
    }

    /// Writes a little-endian 16-bit unsigned integer.
    pub fn write_u16(&mut self, value: u16) -> Result<()> {
        self.write_bytes(&value.to_le_bytes())
    }

    /// Writes a little-endian 32-bit unsigned integer.
    pub fn write_u32(&mut self, value: u32) -> Result<()> {
        self.write_bytes(&value.to_le_bytes())
    }

    /// Writes a little-endian 64-bit unsigned integer.
    pub fn write_u64(&mut self, value: u64) -> Result<()> {
        self.write_bytes(&value.to_le_bytes())
    }

    /// Writes a little-endian IEEE-754 f32.
    pub fn write_f32(&mut self, value: f32) -> Result<()> {
        self.write_bytes(&value.to_le_bytes())
    }

    /// Writes raw bytes without a length prefix.
    pub fn write_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        self.reserve(bytes.len())?;
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }

    /// Writes a length-prefixed UTF-8 string.
    ///
    /// The caller supplies the semantic cap. The encoder rejects strings that
    /// do not fit in a u32 length prefix.
    pub fn write_string(&mut self, value: &str) -> Result<()> {
        let length = u32::try_from(value.len())
            .map_err(|_| Error::limit_exceeded("storage string exceeds the u32 length range"))?;
        self.write_u32(length)?;
        self.write_bytes(value.as_bytes())
    }
}

/// Decodes values from a borrowed byte slice with checked bounds.
#[derive(Debug, Clone)]
pub struct Decoder<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Decoder<'a> {
    /// Creates a decoder over the whole slice.
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    /// Current offset from the start of the slice.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Number of bytes left after the cursor.
    pub fn remaining(&self) -> usize {
        self.bytes.len() - self.position
    }

    /// Returns true when the slice has been fully consumed.
    pub fn is_empty(&self) -> bool {
        self.position == self.bytes.len()
    }

    fn error(&self, message: impl Into<String>) -> Error {
        Error::corrupt(format!(
            "storage record at byte {}: {}",
            self.position,
            message.into()
        ))
    }

    fn read_exact(&mut self, length: usize) -> Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(length)
            .ok_or_else(|| self.error("range overflow"))?;
        if end > self.bytes.len() {
            return Err(self.error(format!(
                "truncated storage record: need {length} bytes at offset {}, only {} remain",
                self.position,
                self.remaining()
            )));
        }
        let slice = &self.bytes[self.position..end];
        self.position = end;
        Ok(slice)
    }

    /// Reads one byte.
    pub fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_exact(1)?[0])
    }

    /// Reads a little-endian 16-bit unsigned integer.
    pub fn read_u16(&mut self) -> Result<u16> {
        let bytes: [u8; 2] = self
            .read_exact(2)?
            .try_into()
            .map_err(|_| self.error("u16 conversion failed"))?;
        Ok(u16::from_le_bytes(bytes))
    }

    /// Reads a little-endian 32-bit unsigned integer.
    pub fn read_u32(&mut self) -> Result<u32> {
        let bytes: [u8; 4] = self
            .read_exact(4)?
            .try_into()
            .map_err(|_| self.error("u32 conversion failed"))?;
        Ok(u32::from_le_bytes(bytes))
    }

    /// Reads a little-endian 64-bit unsigned integer.
    pub fn read_u64(&mut self) -> Result<u64> {
        let bytes: [u8; 8] = self
            .read_exact(8)?
            .try_into()
            .map_err(|_| self.error("u64 conversion failed"))?;
        Ok(u64::from_le_bytes(bytes))
    }

    /// Reads a little-endian IEEE-754 f32.
    pub fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    /// Reads a length-prefixed byte string with an explicit cap.
    pub fn read_bytes(&mut self, max_bytes: usize) -> Result<&'a [u8]> {
        let length = self.read_u32()? as usize;
        if length > max_bytes {
            return Err(Error::limit_exceeded(format!(
                "storage byte string is {length} bytes, above the {max_bytes}-byte cap"
            )));
        }
        self.read_exact(length)
    }

    /// Reads a length-prefixed UTF-8 string with an explicit cap.
    pub fn read_string(&mut self, max_bytes: usize) -> Result<String> {
        let bytes = self.read_bytes(max_bytes)?;
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| self.error("storage string is not valid UTF-8"))
    }

    /// Verifies that the decoder consumed the whole slice.
    pub fn finish(&self) -> Result<()> {
        if self.is_empty() {
            Ok(())
        } else {
            Err(self.error(format!(
                "{} trailing bytes after the storage value",
                self.remaining()
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_scalars_and_strings() {
        let mut encoder = Encoder::new(1024);
        encoder.write_u8(7).expect("u8");
        encoder.write_u16(0x1234).expect("u16");
        encoder.write_u32(0x1234_5678).expect("u32");
        encoder.write_u64(0x0123_4567_89AB_CDEF).expect("u64");
        encoder.write_f32(1.5).expect("f32");
        encoder.write_string("héllo").expect("string");
        let bytes = encoder.finish();

        let mut decoder = Decoder::new(&bytes);
        assert_eq!(decoder.read_u8().expect("u8"), 7);
        assert_eq!(decoder.read_u16().expect("u16"), 0x1234);
        assert_eq!(decoder.read_u32().expect("u32"), 0x1234_5678);
        assert_eq!(decoder.read_u64().expect("u64"), 0x0123_4567_89AB_CDEF);
        assert_eq!(decoder.read_f32().expect("f32"), 1.5);
        assert_eq!(decoder.read_string(32).expect("string"), "héllo");
        decoder.finish().expect("all bytes consumed");
    }

    #[test]
    fn encoder_enforces_its_cap_before_growing() {
        let mut encoder = Encoder::new(4);
        encoder.write_u32(1).expect("fits");
        assert_eq!(encoder.len(), 4);
        let error = encoder.write_u8(1).expect_err("the cap must be enforced");
        assert!(matches!(error, Error::LimitExceeded(_)));
        assert_eq!(encoder.len(), 4);
    }

    #[test]
    fn decoder_rejects_truncated_values() {
        let bytes = [1, 2, 3];
        let mut decoder = Decoder::new(&bytes);
        let error = decoder.read_u32().expect_err("truncated u32 must fail");
        assert!(matches!(error, Error::Corrupt(_)));
        assert_eq!(decoder.position(), 0);
    }

    #[test]
    fn decoder_rejects_oversized_strings_before_reading() {
        let mut encoder = Encoder::new(64);
        encoder.write_u32(10).expect("length");
        let bytes = encoder.finish();
        let mut decoder = Decoder::new(&bytes);
        let error = decoder
            .read_string(4)
            .expect_err("the string cap must be enforced");
        assert!(matches!(error, Error::LimitExceeded(_)));
        assert_eq!(decoder.position(), 4);
    }

    #[test]
    fn decoder_rejects_invalid_utf8() {
        let mut encoder = Encoder::new(16);
        encoder.write_u32(1).expect("length");
        encoder.write_u8(0xFF).expect("byte");
        let bytes = encoder.finish();
        let mut decoder = Decoder::new(&bytes);
        let error = decoder.read_string(8).expect_err("invalid UTF-8 must fail");
        assert!(matches!(error, Error::Corrupt(_)));
    }

    #[test]
    fn finish_rejects_trailing_bytes() {
        let bytes = [1, 2];
        let decoder = Decoder::new(&bytes);
        let error = decoder.finish().expect_err("trailing bytes must fail");
        assert!(matches!(error, Error::Corrupt(_)));
        assert!(error.to_string().contains("trailing"));
    }
}
