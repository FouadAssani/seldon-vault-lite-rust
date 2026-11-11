//! Bounded zlib and DEFLATE decoder for PNG image data.
//!
//! PNG stores scanlines as a zlib stream: a two-byte header, DEFLATE blocks,
//! and a four-byte Adler-32 trailer. This module implements stored, fixed
//! Huffman, and dynamic Huffman blocks, including LZ77 back-references. It
//! enforces the caller's output cap before every byte, so a decompression bomb
//! cannot allocate beyond the documented image limit.
//!
//! The decoder never panics on hostile input: truncated streams, invalid
//! Huffman codes, impossible distances, bad checksums, and trailing bytes are
//! typed corruption errors.

use crate::error::{Error, Result};

/// Maximum compressed input size accepted by the zlib decoder.
pub const MAX_ZLIB_INPUT_BYTES: usize = 64 * 1024 * 1024;
/// Maximum Huffman code length in DEFLATE.
const MAX_CODE_BITS: usize = 15;

const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

/// Decompresses one zlib stream with an explicit output cap.
///
/// The returned vector never exceeds max_output bytes. A missing or extra
/// trailer byte is corruption; a valid stream must end exactly after its
/// Adler-32 checksum.
pub fn inflate_zlib(input: &[u8], max_output: usize) -> Result<Vec<u8>> {
    if input.len() > MAX_ZLIB_INPUT_BYTES {
        return Err(Error::limit_exceeded(format!(
            "zlib input is {} bytes, above the {MAX_ZLIB_INPUT_BYTES}-byte cap",
            input.len()
        )));
    }
    if input.len() < 6 {
        return Err(Error::corrupt(
            "zlib stream is shorter than its header and trailer",
        ));
    }

    let cmf = input[0];
    let flg = input[1];
    if cmf & 0x0F != 8 {
        return Err(Error::unsupported(format!(
            "zlib compression method {} is not DEFLATE",
            cmf & 0x0F
        )));
    }
    if cmf >> 4 > 7 {
        return Err(Error::corrupt("zlib window size is above 32 KiB"));
    }
    if (u16::from(cmf) << 8 | u16::from(flg)) % 31 != 0 {
        return Err(Error::corrupt("zlib header check bits are invalid"));
    }
    if flg & 0x20 != 0 {
        return Err(Error::unsupported(
            "zlib preset dictionaries are not supported",
        ));
    }

    let mut reader = BitReader::new(&input[2..]);
    let mut output = Vec::new();

    loop {
        let final_block = reader.read_bits(1)? == 1;
        let block_type = reader.read_bits(2)?;
        match block_type {
            0 => decode_stored_block(&mut reader, &mut output, max_output)?,
            1 => {
                let literal = fixed_literal_huffman()?;
                let distance = fixed_distance_huffman()?;
                decode_huffman_block(&mut reader, &mut output, max_output, &literal, &distance)?;
            }
            2 => {
                let (literal, distance) = read_dynamic_huffman(&mut reader)?;
                decode_huffman_block(&mut reader, &mut output, max_output, &literal, &distance)?;
            }
            _ => return Err(Error::corrupt("reserved DEFLATE block type 3")),
        }
        if final_block {
            break;
        }
    }

    reader.align_to_byte();
    let trailer_start = reader.byte_position();
    let trailer_end = trailer_start
        .checked_add(4)
        .ok_or_else(|| Error::corrupt("zlib trailer offset overflow"))?;
    if trailer_end != reader.input.len() {
        return Err(Error::corrupt(if trailer_end > reader.input.len() {
            "zlib stream is truncated before its Adler-32 trailer"
        } else {
            "zlib stream has trailing bytes after its Adler-32 trailer"
        }));
    }
    let trailer = &reader.input[trailer_start..trailer_end];
    let expected = u32::from_be_bytes([trailer[0], trailer[1], trailer[2], trailer[3]]);
    if adler32(&output) != expected {
        return Err(Error::corrupt("zlib Adler-32 checksum mismatch"));
    }

    Ok(output)
}

struct BitReader<'a> {
    input: &'a [u8],
    bit_position: usize,
}

impl<'a> BitReader<'a> {
    fn new(input: &'a [u8]) -> Self {
        Self {
            input,
            bit_position: 0,
        }
    }

    fn read_bit(&mut self) -> Result<u32> {
        let byte_index = self.bit_position / 8;
        let byte = self
            .input
            .get(byte_index)
            .ok_or_else(|| Error::corrupt("truncated DEFLATE bitstream"))?;
        let bit = (byte >> (self.bit_position % 8)) & 1;
        self.bit_position += 1;
        Ok(u32::from(bit))
    }

    fn read_bits(&mut self, count: u32) -> Result<u32> {
        if count > 32 {
            return Err(Error::internal("DEFLATE bit read exceeds 32 bits"));
        }
        let mut value = 0u32;
        for index in 0..count {
            value |= self.read_bit()? << index;
        }
        Ok(value)
    }

    fn align_to_byte(&mut self) {
        self.bit_position = (self.bit_position + 7) & !7;
    }

    fn byte_position(&self) -> usize {
        self.bit_position / 8
    }

    fn read_aligned_bytes(&mut self, length: usize) -> Result<&'a [u8]> {
        if !self.bit_position.is_multiple_of(8) {
            return Err(Error::internal("aligned byte read on a non-byte boundary"));
        }
        let start = self.bit_position / 8;
        let end = start
            .checked_add(length)
            .ok_or_else(|| Error::corrupt("DEFLATE byte range overflow"))?;
        if end > self.input.len() {
            return Err(Error::corrupt(format!(
                "truncated DEFLATE byte range: need {length} bytes at offset {start}"
            )));
        }
        self.bit_position = end * 8;
        Ok(&self.input[start..end])
    }

    fn read_u16_le(&mut self) -> Result<u16> {
        let bytes = self.read_aligned_bytes(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }
}

#[derive(Debug)]
struct Huffman {
    counts: [u16; 16],
    first_code: [u32; 16],
    first_symbol: [u32; 16],
    symbols: Vec<u16>,
}

impl Huffman {
    fn build(lengths: &[u8]) -> Result<Self> {
        let mut counts = [0u16; 16];
        for length in lengths {
            let length = usize::from(*length);
            if length > MAX_CODE_BITS {
                return Err(Error::corrupt(format!(
                    "Huffman code length {length} exceeds {MAX_CODE_BITS}"
                )));
            }
            counts[length] += 1;
        }

        let symbol_count = lengths.iter().filter(|length| **length != 0).count();
        let mut symbols = Vec::with_capacity(symbol_count);
        for length in 1..=MAX_CODE_BITS {
            for (symbol, symbol_length) in lengths.iter().enumerate() {
                if usize::from(*symbol_length) == length {
                    symbols.push(symbol as u16);
                }
            }
        }

        let mut left = 1i32;
        for count in counts.iter().take(MAX_CODE_BITS + 1).skip(1) {
            left <<= 1;
            left -= i32::from(*count);
            if left < 0 {
                return Err(Error::corrupt("oversubscribed Huffman code"));
            }
        }

        let mut first_code = [0u32; 16];
        let mut first_symbol = [0u32; 16];
        let mut code = 0u32;
        let mut symbol_index = 0u32;
        for length in 1..=MAX_CODE_BITS {
            first_code[length] = code;
            first_symbol[length] = symbol_index;
            code = (code + u32::from(counts[length])) << 1;
            symbol_index += u32::from(counts[length]);
        }

        Ok(Self {
            counts,
            first_code,
            first_symbol,
            symbols,
        })
    }

    fn decode(&self, reader: &mut BitReader<'_>) -> Result<u16> {
        if self.symbols.is_empty() {
            return Err(Error::corrupt("empty Huffman table"));
        }
        let mut code = 0u32;
        for length in 1..=MAX_CODE_BITS {
            code = (code << 1) | reader.read_bit()?;
            let count = u32::from(self.counts[length]);
            if count != 0 {
                let first = self.first_code[length];
                if code >= first && code < first + count {
                    let index = self.first_symbol[length] + (code - first);
                    return self
                        .symbols
                        .get(index as usize)
                        .copied()
                        .ok_or_else(|| Error::internal("Huffman symbol index out of range"));
                }
            }
        }
        Err(Error::corrupt("invalid Huffman code"))
    }
}

fn fixed_literal_huffman() -> Result<Huffman> {
    let mut lengths = [0u8; 288];
    lengths[0..144].fill(8);
    lengths[144..256].fill(9);
    lengths[256..280].fill(7);
    lengths[280..288].fill(8);
    Huffman::build(&lengths)
}

fn fixed_distance_huffman() -> Result<Huffman> {
    Huffman::build(&[5u8; 32])
}

fn read_dynamic_huffman(reader: &mut BitReader<'_>) -> Result<(Huffman, Huffman)> {
    let literal_count = reader.read_bits(5)? as usize + 257;
    let distance_count = reader.read_bits(5)? as usize + 1;
    let code_length_count = reader.read_bits(4)? as usize + 4;

    if literal_count > 286 {
        return Err(Error::corrupt("dynamic Huffman literal count exceeds 286"));
    }
    if distance_count > 32 {
        return Err(Error::corrupt("dynamic Huffman distance count exceeds 32"));
    }

    const ORDER: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];
    let mut code_lengths = [0u8; 19];
    for index in 0..code_length_count {
        code_lengths[ORDER[index]] = reader.read_bits(3)? as u8;
    }
    let code_length_huffman = Huffman::build(&code_lengths)?;

    let total = literal_count
        .checked_add(distance_count)
        .ok_or_else(|| Error::corrupt("dynamic Huffman length count overflow"))?;
    let mut lengths = Vec::with_capacity(total);
    while lengths.len() < total {
        let symbol = code_length_huffman.decode(reader)?;
        match symbol {
            0..=15 => lengths.push(symbol as u8),
            16 => {
                let previous = *lengths
                    .last()
                    .ok_or_else(|| Error::corrupt("repeat code 16 has no previous length"))?;
                let repeat = 3 + reader.read_bits(2)? as usize;
                append_repeated(&mut lengths, previous, repeat, total)?;
            }
            17 => {
                let repeat = 3 + reader.read_bits(3)? as usize;
                append_repeated(&mut lengths, 0, repeat, total)?;
            }
            18 => {
                let repeat = 11 + reader.read_bits(7)? as usize;
                append_repeated(&mut lengths, 0, repeat, total)?;
            }
            _ => return Err(Error::corrupt("invalid code-length symbol")),
        }
    }

    if lengths[256] == 0 {
        return Err(Error::corrupt(
            "dynamic Huffman table has no end-of-block symbol",
        ));
    }
    let literal_lengths = &lengths[..literal_count];
    let distance_lengths = &lengths[literal_count..];
    let literal = Huffman::build(literal_lengths)?;
    let distance = Huffman::build(distance_lengths)?;
    Ok((literal, distance))
}

fn append_repeated(lengths: &mut Vec<u8>, value: u8, repeat: usize, total: usize) -> Result<()> {
    let new_length = lengths
        .len()
        .checked_add(repeat)
        .ok_or_else(|| Error::corrupt("dynamic Huffman length overflow"))?;
    if new_length > total {
        return Err(Error::corrupt(
            "dynamic Huffman repeat exceeds the declared length count",
        ));
    }
    lengths.resize(new_length, value);
    Ok(())
}

fn decode_stored_block(
    reader: &mut BitReader<'_>,
    output: &mut Vec<u8>,
    max_output: usize,
) -> Result<()> {
    reader.align_to_byte();
    let length = usize::from(reader.read_u16_le()?);
    let complement = reader.read_u16_le()?;
    if length as u16 != !complement {
        return Err(Error::corrupt(
            "stored DEFLATE block length and complement disagree",
        ));
    }
    let bytes = reader.read_aligned_bytes(length)?;
    let new_length = output
        .len()
        .checked_add(length)
        .ok_or_else(|| Error::limit_exceeded("DEFLATE output length overflow"))?;
    if new_length > max_output {
        return Err(Error::limit_exceeded(format!(
            "DEFLATE output exceeds the {max_output}-byte cap"
        )));
    }
    output.extend_from_slice(bytes);
    Ok(())
}

fn decode_huffman_block(
    reader: &mut BitReader<'_>,
    output: &mut Vec<u8>,
    max_output: usize,
    literal: &Huffman,
    distance: &Huffman,
) -> Result<()> {
    loop {
        let symbol = literal.decode(reader)?;
        match symbol {
            0..=255 => push_byte(output, symbol as u8, max_output)?,
            256 => return Ok(()),
            257..=285 => {
                let index = usize::from(symbol - 257);
                let length = usize::from(LENGTH_BASE[index])
                    + reader.read_bits(u32::from(LENGTH_EXTRA[index]))? as usize;
                let distance_symbol = distance.decode(reader)?;
                if distance_symbol > 29 {
                    return Err(Error::corrupt(format!(
                        "invalid DEFLATE distance symbol {distance_symbol}"
                    )));
                }
                let distance_index = usize::from(distance_symbol);
                let copy_distance = usize::from(DIST_BASE[distance_index])
                    + reader.read_bits(u32::from(DIST_EXTRA[distance_index]))? as usize;
                if copy_distance == 0 || copy_distance > output.len() {
                    return Err(Error::corrupt(format!(
                        "DEFLATE distance {copy_distance} is outside the {} bytes produced",
                        output.len()
                    )));
                }
                let new_length = output
                    .len()
                    .checked_add(length)
                    .ok_or_else(|| Error::limit_exceeded("DEFLATE output length overflow"))?;
                if new_length > max_output {
                    return Err(Error::limit_exceeded(format!(
                        "DEFLATE output exceeds the {max_output}-byte cap"
                    )));
                }
                for _ in 0..length {
                    let byte = output[output.len() - copy_distance];
                    output.push(byte);
                }
            }
            _ => {
                return Err(Error::corrupt(format!(
                    "invalid DEFLATE literal/length symbol {symbol}"
                )));
            }
        }
    }
}

fn push_byte(output: &mut Vec<u8>, byte: u8, max_output: usize) -> Result<()> {
    if output.len() >= max_output {
        return Err(Error::limit_exceeded(format!(
            "DEFLATE output exceeds the {max_output}-byte cap"
        )));
    }
    output.push(byte);
    Ok(())
}

pub(crate) fn adler32(data: &[u8]) -> u32 {
    const MODULUS: u32 = 65_521;
    let mut a = 1u32;
    let mut b = 0u32;
    for byte in data {
        a = (a + u32::from(*byte)) % MODULUS;
        b = (b + a) % MODULUS;
    }
    (b << 16) | a
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(input: &str) -> Vec<u8> {
        assert_eq!(input.len() % 2, 0, "hex length");
        let mut bytes = Vec::with_capacity(input.len() / 2);
        for index in (0..input.len()).step_by(2) {
            bytes.push(u8::from_str_radix(&input[index..index + 2], 16).expect("hex digit"));
        }
        bytes
    }

    #[test]
    fn inflates_a_stored_block() {
        let input = hex("7801011400ebff73746f72656420626c6f636b207061796c6f6164523e07c7");
        assert_eq!(
            inflate_zlib(&input, 1024).expect("stored block"),
            b"stored block payload"
        );
    }

    #[test]
    fn inflates_a_fixed_huffman_block() {
        let input = hex("78014bcbac484d5148cac94fce562848acccc94f4c51481b446200aab63b29");
        assert_eq!(
            inflate_zlib(&input, 1024).expect("fixed block"),
            b"fixed block payload ".repeat(8)
        );
    }

    #[test]
    fn inflates_a_dynamic_huffman_block() {
        let input = hex(
            "78daedcbd10dc0200805c055de004d27710983c4908818c0fd3b43ffb9ff6be6ac901357316c992324d195f301d90ea6e4bc8e3ee44890ec095e922f5ac58a152bfe8f1f8778a5c8",
        );
        let expected = b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(20);
        assert_eq!(
            inflate_zlib(&input, expected.len()).expect("dynamic block"),
            expected
        );
    }

    #[test]
    fn inflates_empty_output() {
        let input = hex("789c030000000001");
        assert!(inflate_zlib(&input, 16).expect("empty stream").is_empty());
    }

    #[test]
    fn rejects_an_adler32_mismatch() {
        let mut input = hex("789ccb48cdc9c90700062c0215");
        let last = input.len() - 1;
        input[last] ^= 0xFF;
        let error = inflate_zlib(&input, 1024).expect_err("bad Adler-32");
        assert!(matches!(error, Error::Corrupt(_)), "{error}");
    }

    #[test]
    fn rejects_truncated_streams() {
        let input = hex("789ccb48cdc9c90700062c0215");
        for cut in 0..input.len() {
            let error = inflate_zlib(&input[..cut], 1024).expect_err("truncated stream");
            assert!(matches!(error, Error::Corrupt(_)), "cut {cut}: {error}");
        }
    }

    #[test]
    fn enforces_the_output_cap() {
        let input = hex("789ccb48cdc9c90700062c0215");
        let error = inflate_zlib(&input, 4).expect_err("output cap");
        assert!(matches!(error, Error::LimitExceeded(_)), "{error}");
    }

    #[test]
    fn rejects_bad_zlib_headers() {
        let mut input = hex("789ccb48cdc9c90700062c0215");
        input[0] = 0x79;
        input[1] = 0x18;
        let error = inflate_zlib(&input, 1024).expect_err("unsupported compression method");
        assert!(matches!(error, Error::Unsupported(_)), "{error}");

        let mut input = hex("789ccb48cdc9c90700062c0215");
        input[1] ^= 1;
        let error = inflate_zlib(&input, 1024).expect_err("bad check bits");
        assert!(matches!(error, Error::Corrupt(_)), "{error}");

        let mut input = hex("789ccb48cdc9c90700062c0215");
        input[1] = 0x20;
        let error = inflate_zlib(&input, 1024).expect_err("preset dictionary");
        assert!(matches!(error, Error::Unsupported(_)), "{error}");
    }

    #[test]
    fn rejects_trailing_bytes() {
        let mut input = hex("789ccb48cdc9c90700062c0215");
        input.push(0);
        let error = inflate_zlib(&input, 1024).expect_err("trailing byte");
        assert!(matches!(error, Error::Corrupt(_)), "{error}");
    }

    #[test]
    fn rejects_inputs_above_the_compressed_cap() {
        let input = vec![0u8; MAX_ZLIB_INPUT_BYTES + 1];
        let error = inflate_zlib(&input, 16).expect_err("input cap");
        assert!(matches!(error, Error::LimitExceeded(_)), "{error}");
    }

    #[test]
    fn adler32_matches_the_standard_value() {
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
    }
}
