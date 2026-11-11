//! Bounded byte reader with checked offsets and endianness.
//!
//! Every length, offset, and endianness conversion goes through checked
//! arithmetic. The reader never allocates: it returns borrowed slices, so a
//! hostile length can only produce a typed error, never an allocation.

use crate::error::{Error, Result};

/// Byte order used by a reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endian {
    /// Least significant byte first (Safetensors, RIFF).
    Little,
    /// Most significant byte first (network byte order, some image formats).
    Big,
}

/// A cursor over a borrowed byte slice.
///
/// The reader is the single entry point for binary parsing in the crate. It
/// deliberately exposes no unchecked accessor.
#[derive(Debug, Clone)]
pub struct ByteReader<'a> {
    bytes: &'a [u8],
    position: usize,
    endian: Endian,
}

impl<'a> ByteReader<'a> {
    /// Creates a little-endian reader over the whole slice.
    pub fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            position: 0,
            endian: Endian::Little,
        }
    }

    /// Creates a reader with an explicit byte order.
    pub fn with_endian(bytes: &'a [u8], endian: Endian) -> Self {
        Self {
            bytes,
            position: 0,
            endian,
        }
    }

    /// Current offset from the start of the slice.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Number of bytes left after the cursor.
    pub fn remaining(&self) -> usize {
        self.bytes.len() - self.position
    }

    /// Returns true when no byte is left.
    pub fn is_empty(&self) -> bool {
        self.position == self.bytes.len()
    }

    /// Byte order used by scalar reads.
    pub fn endian(&self) -> Endian {
        self.endian
    }

    /// Changes the byte order used by subsequent scalar reads.
    pub fn set_endian(&mut self, endian: Endian) {
        self.endian = endian;
    }

    /// Borrows the bytes that have not been consumed yet.
    pub fn remaining_slice(&self) -> &'a [u8] {
        &self.bytes[self.position..]
    }

    /// Consumes exactly length bytes and returns them.
    ///
    /// # Errors
    ///
    /// Returns Error::Corrupt when the requested range overflows or extends
    /// past the end of the slice.
    pub fn read_exact(&mut self, length: usize) -> Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(length)
            .ok_or_else(|| Error::corrupt("byte range overflow"))?;
        if end > self.bytes.len() {
            return Err(Error::corrupt(format!(
                "truncated input: need {length} bytes at offset {}, only {} remain",
                self.position,
                self.remaining()
            )));
        }
        let slice = &self.bytes[self.position..end];
        self.position = end;
        Ok(slice)
    }

    /// Skips exactly length bytes.
    pub fn skip(&mut self, length: usize) -> Result<()> {
        self.read_exact(length).map(|_| ())
    }

    /// Consumes exactly N bytes and returns them as a fixed-size array.
    pub fn read_array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let slice = self.read_exact(N)?;
        let mut array = [0u8; N];
        array.copy_from_slice(slice);
        Ok(array)
    }

    /// Reads one unsigned byte.
    pub fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_array::<1>()?[0])
    }

    /// Reads one signed byte.
    pub fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    /// Reads a 16-bit unsigned integer using the current byte order.
    pub fn read_u16(&mut self) -> Result<u16> {
        let bytes = self.read_array::<2>()?;
        Ok(match self.endian {
            Endian::Little => u16::from_le_bytes(bytes),
            Endian::Big => u16::from_be_bytes(bytes),
        })
    }

    /// Reads a 16-bit signed integer using the current byte order.
    pub fn read_i16(&mut self) -> Result<i16> {
        Ok(self.read_u16()? as i16)
    }

    /// Reads a 32-bit unsigned integer using the current byte order.
    pub fn read_u32(&mut self) -> Result<u32> {
        let bytes = self.read_array::<4>()?;
        Ok(match self.endian {
            Endian::Little => u32::from_le_bytes(bytes),
            Endian::Big => u32::from_be_bytes(bytes),
        })
    }

    /// Reads a 32-bit signed integer using the current byte order.
    pub fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32()? as i32)
    }

    /// Reads a 64-bit unsigned integer using the current byte order.
    pub fn read_u64(&mut self) -> Result<u64> {
        let bytes = self.read_array::<8>()?;
        Ok(match self.endian {
            Endian::Little => u64::from_le_bytes(bytes),
            Endian::Big => u64::from_be_bytes(bytes),
        })
    }

    /// Reads a 64-bit signed integer using the current byte order.
    pub fn read_i64(&mut self) -> Result<i64> {
        Ok(self.read_u64()? as i64)
    }

    /// Reads a 32-bit IEEE-754 float using the current byte order.
    pub fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    /// Reads a 64-bit IEEE-754 float using the current byte order.
    pub fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    /// Reads length bytes into a new vector, enforcing an explicit cap.
    ///
    /// This is the only allocating method of the reader. The cap must be
    /// chosen by the caller from a documented format limit.
    pub fn read_vec(&mut self, length: usize, max_bytes: usize) -> Result<Vec<u8>> {
        if length > max_bytes {
            return Err(Error::limit_exceeded(format!(
                "requested {length} bytes exceeds the {max_bytes}-byte cap"
            )));
        }
        Ok(self.read_exact(length)?.to_vec())
    }
}

/// Reads at most max_bytes from a regular file.
///
/// The file size is checked from metadata before reading, then the read is
/// bounded again with std::io::Read::take so that a growing file cannot
/// bypass the limit. Non-regular files are rejected.
pub fn read_file_bounded(path: &std::path::Path, max_bytes: u64) -> Result<Vec<u8>> {
    use std::fs::File;
    use std::io::Read;

    let metadata = std::fs::metadata(path)?;
    if !metadata.is_file() {
        return Err(Error::corrupt(format!(
            "{} is not a regular file",
            path.display()
        )));
    }
    if metadata.len() > max_bytes {
        return Err(Error::limit_exceeded(format!(
            "{} is {} bytes, above the {max_bytes}-byte cap",
            path.display(),
            metadata.len()
        )));
    }

    let file = File::open(path)?;
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > max_bytes {
        return Err(Error::limit_exceeded(format!(
            "{} grew above the {max_bytes}-byte cap while being read",
            path.display()
        )));
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_little_endian_integers() {
        let bytes = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let mut reader = ByteReader::new(&bytes);
        assert_eq!(reader.read_u8().expect("u8"), 0x01);
        assert_eq!(reader.read_u16().expect("u16"), 0x0302);
        assert_eq!(reader.read_u32().expect("u32"), 0x07060504);
        assert_eq!(reader.remaining(), 1);
        assert_eq!(reader.position(), 7);
    }

    #[test]
    fn reads_big_endian_integers() {
        let bytes = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let mut reader = ByteReader::with_endian(&bytes, Endian::Big);
        assert_eq!(reader.read_u16().expect("u16"), 0x0102);
        assert_eq!(reader.read_u32().expect("u32"), 0x03040506);
        assert_eq!(reader.read_u16().expect("u16"), 0x0708);
        assert!(reader.is_empty());
    }

    #[test]
    fn reads_signed_values() {
        let bytes = [0xff, 0xff, 0xfe, 0xff];
        let mut reader = ByteReader::new(&bytes);
        assert_eq!(reader.read_i16().expect("i16"), -1);
        assert_eq!(reader.read_i16().expect("i16"), -2);
    }

    #[test]
    fn reads_floats() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1.5f32.to_le_bytes());
        bytes.extend_from_slice(&(-2.25f64).to_le_bytes());
        let mut reader = ByteReader::new(&bytes);
        assert_eq!(reader.read_f32().expect("f32"), 1.5);
        assert_eq!(reader.read_f64().expect("f64"), -2.25);
    }

    #[test]
    fn rejects_truncated_reads() {
        let bytes = [1, 2, 3];
        let mut reader = ByteReader::new(&bytes);
        let error = reader.read_u32().expect_err("truncated read must fail");
        assert!(matches!(error, Error::Corrupt(_)));
        assert!(error.to_string().contains("truncated"));
        assert_eq!(reader.position(), 0);
    }

    #[test]
    fn rejects_range_overflow() {
        let bytes = [1, 2, 3];
        let mut reader = ByteReader::new(&bytes);
        reader.read_u8().expect("advance the cursor");
        let error = reader
            .read_exact(usize::MAX)
            .expect_err("overflowing range must fail");
        assert!(matches!(error, Error::Corrupt(_)));
        assert!(error.to_string().contains("overflow"));
    }

    #[test]
    fn zero_length_reads_are_allowed() {
        let bytes = [1, 2, 3];
        let mut reader = ByteReader::new(&bytes);
        assert!(reader.read_exact(0).expect("zero-length read").is_empty());
        assert_eq!(reader.position(), 0);
    }

    #[test]
    fn read_vec_enforces_the_cap() {
        let bytes = [1, 2, 3, 4];
        let mut reader = ByteReader::new(&bytes);
        let error = reader
            .read_vec(4, 3)
            .expect_err("the cap must be enforced before reading");
        assert!(matches!(error, Error::LimitExceeded(_)));
        assert_eq!(reader.position(), 0);
        assert_eq!(reader.read_vec(2, 2).expect("bounded read"), vec![1, 2]);
        assert_eq!(reader.remaining_slice(), &[3, 4]);
    }

    #[test]
    fn skip_and_remaining_slice_are_consistent() {
        let bytes = [1, 2, 3, 4];
        let mut reader = ByteReader::new(&bytes);
        reader.skip(2).expect("skip");
        assert_eq!(reader.remaining_slice(), &[3, 4]);
        assert_eq!(reader.remaining(), 2);
        reader.skip(2).expect("skip");
        assert!(reader.is_empty());
        assert!(reader.remaining_slice().is_empty());
    }

    #[test]
    fn endian_can_be_changed_mid_stream() {
        let bytes = [0x01, 0x02];
        let mut reader = ByteReader::new(&bytes);
        assert_eq!(reader.read_u8().expect("u8"), 1);
        reader.set_endian(Endian::Big);
        assert_eq!(reader.endian(), Endian::Big);
        assert_eq!(reader.read_u8().expect("u8"), 2);
    }
}
