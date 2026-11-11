//! Strict PNG decoder for the image path.
//!
//! The decoder supports non-interlaced 8-bit grayscale, RGB, and RGBA PNG
//! images. It validates the signature, every chunk length and CRC, the IHDR
//! fields, dimensions, pixel count, decompressed size, filter bytes, and the
//! zlib stream before producing pixels. Palette, 16-bit, and interlaced images
//! are rejected as unsupported rather than approximated.
//!
//! All allocations are bounded by documented caps. The parser never panics on
//! hostile input.

use std::path::Path;

use crate::binary::crc32::Crc32;
use crate::binary::reader::{ByteReader, Endian, read_file_bounded};
use crate::error::{Error, Result};
use crate::media::deflate::inflate_zlib;
pub use crate::media::image::Image;

/// PNG signature.
pub const PNG_SIGNATURE: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];
/// Maximum PNG file size accepted from disk.
pub const MAX_PNG_FILE_BYTES: u64 = 64 * 1024 * 1024;
/// Maximum image width.
pub const MAX_WIDTH: usize = 16_384;
/// Maximum image height.
pub const MAX_HEIGHT: usize = 16_384;
/// Maximum number of pixels in one image.
pub const MAX_PIXELS: usize = 16_777_216;
/// Maximum decompressed scanline bytes, including filter bytes.
pub const MAX_DECOMPRESSED_BYTES: usize = 64 * 1024 * 1024;
/// Maximum size of one PNG chunk.
const MAX_CHUNK_BYTES: usize = 32 * 1024 * 1024;

/// Decodes a PNG file with the documented file-size cap.
pub fn decode_file(path: &Path) -> Result<Image> {
    let bytes = read_file_bounded(path, MAX_PNG_FILE_BYTES)?;
    decode(&bytes)
}

/// Decodes a PNG byte slice.
pub fn decode(bytes: &[u8]) -> Result<Image> {
    if bytes.len() as u64 > MAX_PNG_FILE_BYTES {
        return Err(Error::limit_exceeded(format!(
            "PNG file is {} bytes, above the {MAX_PNG_FILE_BYTES}-byte cap",
            bytes.len()
        )));
    }
    if bytes.len() < PNG_SIGNATURE.len() || bytes[..8] != PNG_SIGNATURE {
        return Err(Error::corrupt("PNG signature is invalid"));
    }

    let mut reader = ByteReader::with_endian(&bytes[8..], Endian::Big);
    let mut first_chunk = true;
    let mut header = None;
    let mut idat = Vec::new();
    let mut seen_iend = false;

    while !reader.is_empty() {
        if reader.remaining() < 12 {
            return Err(Error::corrupt("truncated PNG chunk header"));
        }
        let length = reader.read_u32()? as usize;
        if length > MAX_CHUNK_BYTES {
            return Err(Error::limit_exceeded(format!(
                "PNG chunk is {length} bytes, above the {MAX_CHUNK_BYTES}-byte cap"
            )));
        }
        let kind_bytes = reader.read_exact(4)?;
        let kind: [u8; 4] = kind_bytes
            .try_into()
            .map_err(|_| Error::corrupt("PNG chunk type is not four bytes"))?;
        let data = reader.read_exact(length)?;
        let stored_crc = reader.read_u32()?;
        let mut hasher = Crc32::new();
        hasher.update(&kind);
        hasher.update(data);
        if hasher.finalize() != stored_crc {
            return Err(Error::corrupt(format!(
                "PNG chunk {} checksum mismatch",
                chunk_name(&kind)
            )));
        }

        if first_chunk {
            if kind != *b"IHDR" {
                return Err(Error::corrupt("PNG first chunk is not IHDR"));
            }
            first_chunk = false;
        }

        if kind == *b"IHDR" {
            if header.is_some() {
                return Err(Error::corrupt("PNG has more than one IHDR chunk"));
            }
            header = Some(parse_ihdr(data)?);
        } else if kind == *b"IDAT" {
            if header.is_none() {
                return Err(Error::corrupt("PNG IDAT appears before IHDR"));
            }
            let new_length = idat
                .len()
                .checked_add(length)
                .ok_or_else(|| Error::limit_exceeded("PNG IDAT length overflow"))?;
            if new_length as u64 > MAX_PNG_FILE_BYTES {
                return Err(Error::limit_exceeded(format!(
                    "PNG IDAT data exceeds the {MAX_PNG_FILE_BYTES}-byte cap"
                )));
            }
            idat.extend_from_slice(data);
        } else if kind == *b"IEND" {
            if length != 0 {
                return Err(Error::corrupt("PNG IEND chunk is not empty"));
            }
            seen_iend = true;
            break;
        } else if kind == *b"PLTE" {
            return Err(Error::unsupported("PNG palette images are not implemented"));
        } else if kind == *b"tRNS" {
            return Err(Error::unsupported(
                "PNG transparency chunks are not implemented",
            ));
        } else if kind[0].is_ascii_uppercase() {
            return Err(Error::unsupported(format!(
                "PNG critical chunk {} is not implemented",
                chunk_name(&kind)
            )));
        }
    }

    if !seen_iend {
        return Err(Error::corrupt("PNG is missing its IEND chunk"));
    }
    if !reader.is_empty() {
        return Err(Error::corrupt("PNG has trailing bytes after IEND"));
    }
    let header = header.ok_or_else(|| Error::corrupt("PNG is missing its IHDR chunk"))?;
    if idat.is_empty() {
        return Err(Error::corrupt("PNG has no IDAT image data"));
    }

    let row_bytes = header
        .width
        .checked_mul(usize::from(header.channels))
        .ok_or_else(|| Error::limit_exceeded("PNG row length overflow"))?;
    let stride = row_bytes
        .checked_add(1)
        .ok_or_else(|| Error::limit_exceeded("PNG stride overflow"))?;
    let expected = header
        .height
        .checked_mul(stride)
        .ok_or_else(|| Error::limit_exceeded("PNG decompressed length overflow"))?;
    if expected > MAX_DECOMPRESSED_BYTES {
        return Err(Error::limit_exceeded(format!(
            "PNG decompressed data is {expected} bytes, above the {MAX_DECOMPRESSED_BYTES}-byte cap"
        )));
    }

    let scanlines = inflate_zlib(&idat, expected)?;
    if scanlines.len() != expected {
        return Err(Error::corrupt(format!(
            "PNG decompressed {} bytes, expected {expected}",
            scanlines.len()
        )));
    }
    let pixels = unfilter(
        &scanlines,
        header.width,
        header.height,
        usize::from(header.channels),
    )?;

    Image::new(header.width, header.height, header.channels, pixels)
}

#[derive(Debug, Clone, Copy)]
struct Ihdr {
    width: usize,
    height: usize,
    channels: u8,
}

fn parse_ihdr(data: &[u8]) -> Result<Ihdr> {
    if data.len() != 13 {
        return Err(Error::corrupt("PNG IHDR chunk is not 13 bytes"));
    }
    let width = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let height = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let bit_depth = data[8];
    let color_type = data[9];
    let compression = data[10];
    let filter = data[11];
    let interlace = data[12];

    if width == 0 || height == 0 {
        return Err(Error::corrupt("PNG dimensions must be positive"));
    }
    if width > MAX_WIDTH || height > MAX_HEIGHT {
        return Err(Error::limit_exceeded(format!(
            "PNG dimensions {width}x{height} exceed the {MAX_WIDTH}x{MAX_HEIGHT} cap"
        )));
    }
    let pixels = width
        .checked_mul(height)
        .ok_or_else(|| Error::limit_exceeded("PNG pixel count overflow"))?;
    if pixels > MAX_PIXELS {
        return Err(Error::limit_exceeded(format!(
            "PNG has {pixels} pixels, above the {MAX_PIXELS}-pixel cap"
        )));
    }
    if bit_depth != 8 {
        return Err(Error::unsupported(format!(
            "PNG bit depth {bit_depth}; only 8-bit samples are implemented"
        )));
    }
    let channels = match color_type {
        0 => 1,
        2 => 3,
        6 => 4,
        3 | 4 => {
            return Err(Error::unsupported(format!(
                "PNG color type {color_type} is not implemented"
            )));
        }
        other => {
            return Err(Error::corrupt(format!("PNG color type {other} is invalid")));
        }
    };
    if compression != 0 {
        return Err(Error::unsupported(format!(
            "PNG compression method {compression} is not implemented"
        )));
    }
    if filter != 0 {
        return Err(Error::unsupported(format!(
            "PNG filter method {filter} is not implemented"
        )));
    }
    if interlace != 0 {
        return Err(Error::unsupported(format!(
            "PNG interlace method {interlace} is not implemented"
        )));
    }

    Ok(Ihdr {
        width,
        height,
        channels,
    })
}

fn unfilter(scanlines: &[u8], width: usize, height: usize, channels: usize) -> Result<Vec<u8>> {
    let row_bytes = width
        .checked_mul(channels)
        .ok_or_else(|| Error::limit_exceeded("PNG row length overflow"))?;
    let stride = row_bytes
        .checked_add(1)
        .ok_or_else(|| Error::limit_exceeded("PNG stride overflow"))?;
    let expected = height
        .checked_mul(stride)
        .ok_or_else(|| Error::limit_exceeded("PNG scanline length overflow"))?;
    if scanlines.len() != expected {
        return Err(Error::internal(
            "PNG scanline length changed after inflation",
        ));
    }

    let mut pixels = Vec::with_capacity(height * row_bytes);
    let mut previous = vec![0u8; row_bytes];
    for y in 0..height {
        let start = y * stride;
        let filter = scanlines[start];
        let mut row = scanlines[start + 1..start + stride].to_vec();
        match filter {
            0 => {}
            1 => {
                for index in channels..row.len() {
                    row[index] = row[index].wrapping_add(row[index - channels]);
                }
            }
            2 => {
                for index in 0..row.len() {
                    row[index] = row[index].wrapping_add(previous[index]);
                }
            }
            3 => {
                for index in 0..row.len() {
                    let left = if index >= channels {
                        row[index - channels]
                    } else {
                        0
                    };
                    let predictor = (u16::from(left) + u16::from(previous[index])) / 2;
                    row[index] = row[index].wrapping_add(predictor as u8);
                }
            }
            4 => {
                for index in 0..row.len() {
                    let left = if index >= channels {
                        row[index - channels]
                    } else {
                        0
                    };
                    let up = previous[index];
                    let up_left = if index >= channels {
                        previous[index - channels]
                    } else {
                        0
                    };
                    row[index] = row[index].wrapping_add(paeth(left, up, up_left));
                }
            }
            other => {
                return Err(Error::corrupt(format!(
                    "PNG scanline {y} uses invalid filter {other}"
                )));
            }
        }
        pixels.extend_from_slice(&row);
        previous.copy_from_slice(&row);
    }
    Ok(pixels)
}

fn paeth(left: u8, up: u8, up_left: u8) -> u8 {
    let p = i16::from(left) + i16::from(up) - i16::from(up_left);
    let pa = (p - i16::from(left)).abs();
    let pb = (p - i16::from(up)).abs();
    let pc = (p - i16::from(up_left)).abs();
    if pa <= pb && pa <= pc {
        left
    } else if pb <= pc {
        up
    } else {
        up_left
    }
}

fn chunk_name(kind: &[u8; 4]) -> String {
    kind.iter()
        .map(|byte| {
            if byte.is_ascii_graphic() {
                char::from(*byte)
            } else {
                '?'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::deflate::adler32;

    fn rgb_fixture() -> Image {
        decode(include_bytes!("../../fixtures/media/rgb_2x2.png")).expect("RGB fixture")
    }

    fn rgba_fixture() -> Image {
        decode(include_bytes!("../../fixtures/media/rgba_2x2.png")).expect("RGBA fixture")
    }

    fn gray_fixture() -> Image {
        decode(include_bytes!("../../fixtures/media/gray_2x2.png")).expect("grayscale fixture")
    }

    fn gradient_fixture() -> Image {
        decode(include_bytes!("../../fixtures/media/gradient_16x16.png")).expect("gradient fixture")
    }

    fn chunk(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(data.len() as u32).to_be_bytes());
        bytes.extend_from_slice(kind);
        bytes.extend_from_slice(data);
        let mut hasher = Crc32::new();
        hasher.update(kind);
        hasher.update(data);
        bytes.extend_from_slice(&hasher.finalize().to_be_bytes());
        bytes
    }

    fn ihdr(width: u32, height: u32, bit_depth: u8, color_type: u8, interlace: u8) -> Vec<u8> {
        let mut data = Vec::with_capacity(13);
        data.extend_from_slice(&width.to_be_bytes());
        data.extend_from_slice(&height.to_be_bytes());
        data.push(bit_depth);
        data.push(color_type);
        data.push(0);
        data.push(0);
        data.push(interlace);
        data
    }

    fn zlib_stored(data: &[u8]) -> Vec<u8> {
        let length = u16::try_from(data.len()).expect("test data fits a stored block");
        let mut bytes = vec![0x78, 0x01, 0x01];
        bytes.extend_from_slice(&length.to_le_bytes());
        bytes.extend_from_slice(&(!length).to_le_bytes());
        bytes.extend_from_slice(data);
        bytes.extend_from_slice(&adler32(data).to_be_bytes());
        bytes
    }

    fn build_png(
        width: u32,
        height: u32,
        bit_depth: u8,
        color_type: u8,
        interlace: u8,
        raw: &[u8],
    ) -> Vec<u8> {
        let mut bytes = PNG_SIGNATURE.to_vec();
        bytes.extend(chunk(
            b"IHDR",
            &ihdr(width, height, bit_depth, color_type, interlace),
        ));
        bytes.extend(chunk(b"IDAT", &zlib_stored(raw)));
        bytes.extend(chunk(b"IEND", b""));
        bytes
    }

    #[test]
    fn decodes_rgb_with_sub_filter() {
        let image = rgb_fixture();
        assert_eq!(image.width(), 2);
        assert_eq!(image.height(), 2);
        assert_eq!(image.channels(), 3);
        assert_eq!(
            image.pixels(),
            &[255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]
        );
        assert_eq!(image.pixel(0, 0), Some(&[255, 0, 0][..]));
        assert_eq!(image.pixel(1, 1), Some(&[255, 255, 255][..]));
        assert_eq!(image.pixel(2, 0), None);
    }

    #[test]
    fn decodes_rgba_with_paeth_filter() {
        let image = rgba_fixture();
        assert_eq!(image.channels(), 4);
        assert_eq!(
            image.pixels(),
            &[
                255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 255, 64, 255, 255, 255, 0
            ]
        );
    }

    #[test]
    fn decodes_grayscale_with_average_and_up_filters() {
        let image = gray_fixture();
        assert_eq!(image.channels(), 1);
        assert_eq!(image.pixels(), &[0, 64, 128, 255]);
    }

    #[test]
    fn decodes_a_dynamic_huffman_gradient() {
        let image = gradient_fixture();
        assert_eq!(image.width(), 16);
        assert_eq!(image.height(), 16);
        assert_eq!(image.channels(), 3);
        assert_eq!(image.pixel(0, 0), Some(&[0, 0, 0][..]));
        assert_eq!(image.pixel(15, 15), Some(&[255, 255, 240][..]));
        assert_eq!(image.pixels().len(), 16 * 16 * 3);
    }

    #[test]
    fn decode_file_reads_a_committed_fixture() {
        let image = decode_file(Path::new("fixtures/media/rgb_2x2.png")).expect("fixture");
        assert_eq!(image.width(), 2);
    }

    #[test]
    fn rejects_a_bad_signature() {
        let mut bytes = include_bytes!("../../fixtures/media/rgb_2x2.png").to_vec();
        bytes[0] ^= 0xFF;
        let error = decode(&bytes).expect_err("bad signature");
        assert!(matches!(error, Error::Corrupt(_)), "{error}");
    }

    #[test]
    fn rejects_every_truncation_of_a_valid_fixture() {
        let bytes = include_bytes!("../../fixtures/media/rgb_2x2.png");
        for cut in 0..bytes.len() {
            let error = decode(&bytes[..cut]).expect_err("truncated PNG");
            assert!(matches!(error, Error::Corrupt(_)), "cut {cut}: {error}");
        }
    }

    #[test]
    fn rejects_a_chunk_crc_mismatch() {
        let mut bytes = include_bytes!("../../fixtures/media/rgb_2x2.png").to_vec();
        let position = bytes
            .windows(4)
            .position(|window| window == b"IDAT")
            .expect("IDAT");
        bytes[position + 4] ^= 0xFF;
        let error = decode(&bytes).expect_err("CRC mismatch");
        assert!(matches!(error, Error::Corrupt(_)), "{error}");
    }

    #[test]
    fn rejects_unsupported_image_features() {
        let cases = [(16u8, 2u8, 0u8), (8, 3, 0), (8, 4, 0), (8, 2, 1)];
        for (bit_depth, color_type, interlace) in cases {
            let bytes = build_png(1, 1, bit_depth, color_type, interlace, &[]);
            let error = decode(&bytes).expect_err("unsupported PNG feature");
            assert!(matches!(error, Error::Unsupported(_)), "{error}");
        }
    }

    #[test]
    fn rejects_unknown_critical_and_transparency_chunks() {
        let mut bytes = PNG_SIGNATURE.to_vec();
        bytes.extend(chunk(b"IHDR", &ihdr(1, 1, 8, 2, 0)));
        bytes.extend(chunk(b"ABCD", b"x"));
        bytes.extend(chunk(b"IDAT", &zlib_stored(&[0, 0, 0, 0])));
        bytes.extend(chunk(b"IEND", b""));
        let error = decode(&bytes).expect_err("unknown critical chunk");
        assert!(matches!(error, Error::Unsupported(_)), "{error}");

        let mut bytes = PNG_SIGNATURE.to_vec();
        bytes.extend(chunk(b"IHDR", &ihdr(1, 1, 8, 2, 0)));
        bytes.extend(chunk(b"tRNS", &[0, 0, 0, 0, 0, 0]));
        bytes.extend(chunk(b"IDAT", &zlib_stored(&[0, 0, 0, 0])));
        bytes.extend(chunk(b"IEND", b""));
        let error = decode(&bytes).expect_err("transparency chunk");
        assert!(matches!(error, Error::Unsupported(_)), "{error}");
    }

    #[test]
    fn rejects_missing_iend_or_idat() {
        let mut bytes = PNG_SIGNATURE.to_vec();
        bytes.extend(chunk(b"IHDR", &ihdr(1, 1, 8, 2, 0)));
        bytes.extend(chunk(b"IDAT", &zlib_stored(&[0, 0, 0, 0])));
        let error = decode(&bytes).expect_err("missing IEND");
        assert!(matches!(error, Error::Corrupt(_)), "{error}");

        let mut bytes = PNG_SIGNATURE.to_vec();
        bytes.extend(chunk(b"IHDR", &ihdr(1, 1, 8, 2, 0)));
        bytes.extend(chunk(b"IEND", b""));
        let error = decode(&bytes).expect_err("missing IDAT");
        assert!(matches!(error, Error::Corrupt(_)), "{error}");
    }

    #[test]
    fn rejects_invalid_filter_bytes() {
        let bytes = build_png(1, 1, 8, 2, 0, &[5, 0, 0, 0]);
        let error = decode(&bytes).expect_err("invalid filter");
        assert!(matches!(error, Error::Corrupt(_)), "{error}");
    }

    #[test]
    fn rejects_a_decompressed_length_mismatch() {
        let bytes = build_png(1, 1, 8, 2, 0, &[]);
        let error = decode(&bytes).expect_err("missing scanlines");
        assert!(matches!(error, Error::Corrupt(_)), "{error}");
    }

    #[test]
    fn enforces_dimension_pixel_and_decompressed_caps() {
        let too_wide = build_png((MAX_WIDTH + 1) as u32, 1, 8, 2, 0, &[]);
        let error = decode(&too_wide).expect_err("width cap");
        assert!(matches!(error, Error::LimitExceeded(_)), "{error}");

        let too_many_pixels = build_png(MAX_WIDTH as u32, 1025, 8, 2, 0, &[]);
        let error = decode(&too_many_pixels).expect_err("pixel cap");
        assert!(matches!(error, Error::LimitExceeded(_)), "{error}");

        let too_much_decompressed = build_png(MAX_WIDTH as u32, 1024, 8, 6, 0, &[]);
        let error = decode(&too_much_decompressed).expect_err("decompressed cap");
        assert!(matches!(error, Error::LimitExceeded(_)), "{error}");
    }
}
