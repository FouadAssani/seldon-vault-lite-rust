//! Strict baseline JPEG decoder for the image path.
//!
//! The decoder supports 8-bit baseline and extended-sequential Huffman JPEG
//! files with one to three components, interleaved scans, 1x1 to 4x4 sampling
//! factors, 8-bit quantization tables, and restart intervals. Progressive,
//! arithmetic-coded, lossless, 12-bit, CMYK, and non-interleaved multi-scan
//! files are rejected as unsupported rather than approximated.
//!
//! Every allocation derived from the file is capped: file size, dimensions,
//! pixel count, MCU count, block count, and decoded memory. Entropy data is
//! read with byte stuffing and restart markers; truncated streams, invalid
//! Huffman codes, impossible runs, bad markers, and trailing bytes are typed
//! corruption errors. The parser never panics on hostile input.
//!
//! Chroma upsampling is bilinear with sample-center alignment. That is
//! deterministic and bounded, but it is not claimed to be bit-identical to a
//! particular reference decoder; official image preprocessing and the vision
//! encoder remain pending, so no image embedding is declared supported.

use std::path::Path;

use crate::binary::reader::read_file_bounded;
use crate::error::{Error, Result};
use crate::media::image::Image;

/// Maximum JPEG file size accepted from disk.
pub const MAX_JPEG_FILE_BYTES: u64 = 64 * 1024 * 1024;
/// Maximum image width.
pub const MAX_WIDTH: usize = 16_384;
/// Maximum image height.
pub const MAX_HEIGHT: usize = 16_384;
/// Maximum number of pixels in one image.
pub const MAX_PIXELS: usize = 16_777_216;
/// Maximum number of frame components.
pub const MAX_COMPONENTS: usize = 3;
/// Maximum number of MCUs in one image.
pub const MAX_MCU_COUNT: usize = 4_194_304;
/// Maximum number of decoded 8x8 blocks across all components.
pub const MAX_BLOCKS: usize = 4_000_000;
/// Maximum decoded memory for component planes plus the output image.
pub const MAX_DECODED_BYTES: usize = 128 * 1024 * 1024;

/// Zigzag index to natural (row-major) index.
const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Decodes a JPEG file with the documented file-size cap.
pub fn decode_file(path: &Path) -> Result<Image> {
    let bytes = read_file_bounded(path, MAX_JPEG_FILE_BYTES)?;
    decode(&bytes)
}

/// Decodes a JPEG byte slice.
pub fn decode(bytes: &[u8]) -> Result<Image> {
    if bytes.len() as u64 > MAX_JPEG_FILE_BYTES {
        return Err(Error::limit_exceeded(format!(
            "JPEG file is {} bytes, above the {MAX_JPEG_FILE_BYTES}-byte cap",
            bytes.len()
        )));
    }
    if bytes.len() < 2 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
        return Err(Error::corrupt("JPEG SOI marker is missing"));
    }

    let mut quant_tables: [Option<[u16; 64]>; 4] = [None; 4];
    let mut dc_tables: [Option<HuffmanTable>; 4] = std::array::from_fn(|_| None);
    let mut ac_tables: [Option<HuffmanTable>; 4] = std::array::from_fn(|_| None);
    let mut frame: Option<Frame> = None;
    let mut adobe_transform: Option<u8> = None;
    let mut restart_interval = 0usize;
    let mut scan_seen = false;
    let mut image: Option<Image> = None;
    let mut position = 2usize;

    while position < bytes.len() {
        let marker = read_marker(bytes, &mut position)?;
        match marker {
            0xD8 => return Err(Error::corrupt("JPEG contains more than one SOI marker")),
            0xD9 => {
                if !scan_seen {
                    return Err(Error::corrupt("JPEG EOI appears before any scan"));
                }
                if position != bytes.len() {
                    return Err(Error::corrupt("JPEG has trailing bytes after EOI"));
                }
                return image.ok_or_else(|| Error::corrupt("JPEG has no decoded image"));
            }
            0x01 => {}
            0xD0..=0xD7 => {
                return Err(Error::corrupt(
                    "JPEG restart marker appears outside entropy data",
                ));
            }
            _ => {
                let data = read_segment(bytes, &mut position)?;
                match marker {
                    0xC0 | 0xC1 => {
                        if frame.is_some() {
                            return Err(Error::corrupt("JPEG has more than one SOF marker"));
                        }
                        let mut parsed = parse_sof(data)?;
                        parsed.adobe_transform = adobe_transform;
                        frame = Some(parsed);
                    }
                    0xC2 => {
                        return Err(Error::unsupported(
                            "progressive JPEG is not implemented; only baseline and extended sequential Huffman scans are supported",
                        ));
                    }
                    0xC3 | 0xC5..=0xC7 | 0xC9..=0xCF => {
                        return Err(Error::unsupported(format!(
                            "JPEG marker 0x{marker:02X} is a lossless, differential, or arithmetic-coded frame and is not implemented"
                        )));
                    }
                    0xC4 => parse_dht(data, &mut dc_tables, &mut ac_tables)?,
                    0xDB => parse_dqt(data, &mut quant_tables)?,
                    0xDD => restart_interval = parse_dri(data)?,
                    0xDA => {
                        if scan_seen {
                            return Err(Error::unsupported(
                                "multiple JPEG scans are not implemented; a single interleaved baseline scan is required",
                            ));
                        }
                        let frame = frame
                            .as_ref()
                            .ok_or_else(|| Error::corrupt("JPEG SOS appears before SOF"))?;
                        let scan = parse_sos(data, frame)?;
                        validate_color_transform(frame)?;
                        for component in &frame.components {
                            if quant_tables[component.quant].is_none() {
                                return Err(Error::corrupt(format!(
                                    "JPEG component {} references missing quantization table {}",
                                    component.id, component.quant
                                )));
                            }
                        }
                        for scan_component in &scan {
                            if dc_tables[scan_component.dc_table].is_none() {
                                return Err(Error::corrupt(format!(
                                    "JPEG scan references missing DC Huffman table {}",
                                    scan_component.dc_table
                                )));
                            }
                            if ac_tables[scan_component.ac_table].is_none() {
                                return Err(Error::corrupt(format!(
                                    "JPEG scan references missing AC Huffman table {}",
                                    scan_component.ac_table
                                )));
                            }
                        }
                        let (decoded, consumed) = decode_scan(
                            &bytes[position..],
                            frame,
                            &scan,
                            &quant_tables,
                            &dc_tables,
                            &ac_tables,
                            restart_interval,
                        )?;
                        position = position
                            .checked_add(consumed)
                            .ok_or_else(|| Error::limit_exceeded("JPEG scan length overflow"))?;
                        image = Some(decoded);
                        scan_seen = true;
                    }
                    0xEE => {
                        if let Some(transform) = parse_app14(data)? {
                            adobe_transform = Some(transform);
                            if let Some(frame) = frame.as_mut() {
                                frame.adobe_transform = Some(transform);
                            }
                        }
                    }
                    0xE0..=0xED | 0xEF | 0xFE => {}
                    _ => {
                        return Err(Error::unsupported(format!(
                            "JPEG marker 0x{marker:02X} is not implemented"
                        )));
                    }
                }
            }
        }
    }

    Err(Error::corrupt("JPEG is missing its EOI marker"))
}

/// Reads the next marker, skipping any fill bytes.
fn read_marker(bytes: &[u8], position: &mut usize) -> Result<u8> {
    if *position >= bytes.len() {
        return Err(Error::corrupt("truncated JPEG marker"));
    }
    if bytes[*position] != 0xFF {
        return Err(Error::corrupt("expected a JPEG marker"));
    }
    while *position < bytes.len() && bytes[*position] == 0xFF {
        *position += 1;
    }
    if *position >= bytes.len() {
        return Err(Error::corrupt("truncated JPEG marker"));
    }
    let marker = bytes[*position];
    *position += 1;
    Ok(marker)
}

/// Reads a marker segment and returns its payload, excluding the length.
fn read_segment<'a>(bytes: &'a [u8], position: &mut usize) -> Result<&'a [u8]> {
    if *position + 2 > bytes.len() {
        return Err(Error::corrupt("truncated JPEG segment length"));
    }
    let length = usize::from(u16::from_be_bytes([bytes[*position], bytes[*position + 1]]));
    if length < 2 {
        return Err(Error::corrupt("JPEG segment length is below two"));
    }
    let end = position
        .checked_add(length)
        .ok_or_else(|| Error::limit_exceeded("JPEG segment length overflow"))?;
    if end > bytes.len() {
        return Err(Error::corrupt("JPEG segment extends past the file"));
    }
    let data = &bytes[*position + 2..end];
    *position = end;
    Ok(data)
}

#[derive(Debug, Clone, Copy)]
struct FrameComponent {
    id: u8,
    h: usize,
    v: usize,
    quant: usize,
    plane_width: usize,
    plane_height: usize,
}

#[derive(Debug)]
struct Frame {
    width: usize,
    height: usize,
    components: Vec<FrameComponent>,
    max_h: usize,
    max_v: usize,
    mcu_cols: usize,
    mcu_rows: usize,
    adobe_transform: Option<u8>,
}

fn parse_sof(data: &[u8]) -> Result<Frame> {
    if data.len() < 6 {
        return Err(Error::corrupt("JPEG SOF segment is too short"));
    }
    let precision = data[0];
    let height = usize::from(u16::from_be_bytes([data[1], data[2]]));
    let width = usize::from(u16::from_be_bytes([data[3], data[4]]));
    let component_count = usize::from(data[5]);
    if data.len() != 6 + 3 * component_count {
        return Err(Error::corrupt(
            "JPEG SOF component count does not match its length",
        ));
    }
    if precision != 8 {
        return Err(Error::unsupported(format!(
            "JPEG sample precision {precision}; only 8-bit samples are implemented"
        )));
    }
    if width == 0 || height == 0 {
        return Err(Error::corrupt("JPEG dimensions must be positive"));
    }
    if width > MAX_WIDTH || height > MAX_HEIGHT {
        return Err(Error::limit_exceeded(format!(
            "JPEG dimensions {width}x{height} exceed the {MAX_WIDTH}x{MAX_HEIGHT} cap"
        )));
    }
    if component_count == 0 || component_count > MAX_COMPONENTS {
        return Err(Error::unsupported(format!(
            "JPEG has {component_count} components; only 1 to {MAX_COMPONENTS} are implemented"
        )));
    }
    let pixels = width
        .checked_mul(height)
        .ok_or_else(|| Error::limit_exceeded("JPEG pixel count overflow"))?;
    if pixels > MAX_PIXELS {
        return Err(Error::limit_exceeded(format!(
            "JPEG has {pixels} pixels, above the {MAX_PIXELS}-pixel cap"
        )));
    }

    let mut components = Vec::with_capacity(component_count);
    for chunk in data[6..].as_chunks::<3>().0 {
        let id = chunk[0];
        let sampling = chunk[1];
        let h = usize::from(sampling >> 4);
        let v = usize::from(sampling & 0x0F);
        let quant = usize::from(chunk[2]);
        if !(1..=4).contains(&h) || !(1..=4).contains(&v) {
            return Err(Error::corrupt(format!(
                "JPEG component {id} has invalid sampling factors {h}x{v}"
            )));
        }
        if h * v > 10 {
            return Err(Error::unsupported(format!(
                "JPEG component {id} requests {} blocks per MCU, above the 10-block limit",
                h * v
            )));
        }
        if quant > 3 {
            return Err(Error::corrupt(format!(
                "JPEG component {id} references quantization table {quant}"
            )));
        }
        if components
            .iter()
            .any(|candidate: &FrameComponent| candidate.id == id)
        {
            return Err(Error::corrupt(format!(
                "JPEG has more than one component with id {id}"
            )));
        }
        components.push(FrameComponent {
            id,
            h,
            v,
            quant,
            plane_width: 0,
            plane_height: 0,
        });
    }

    let max_h = components
        .iter()
        .map(|component| component.h)
        .max()
        .unwrap_or(1);
    let max_v = components
        .iter()
        .map(|component| component.v)
        .max()
        .unwrap_or(1);
    let mcu_width = max_h
        .checked_mul(8)
        .ok_or_else(|| Error::limit_exceeded("JPEG MCU width overflow"))?;
    let mcu_height = max_v
        .checked_mul(8)
        .ok_or_else(|| Error::limit_exceeded("JPEG MCU height overflow"))?;
    let mcu_cols = width.div_ceil(mcu_width);
    let mcu_rows = height.div_ceil(mcu_height);
    let mcu_count = mcu_cols
        .checked_mul(mcu_rows)
        .ok_or_else(|| Error::limit_exceeded("JPEG MCU count overflow"))?;
    if mcu_count > MAX_MCU_COUNT {
        return Err(Error::limit_exceeded(format!(
            "JPEG has {mcu_count} MCUs, above the {MAX_MCU_COUNT}-MCU cap"
        )));
    }

    let mut total_blocks = 0usize;
    let mut plane_bytes = 0usize;
    for component in &mut components {
        let blocks_x = mcu_cols
            .checked_mul(component.h)
            .ok_or_else(|| Error::limit_exceeded("JPEG block column count overflow"))?;
        let blocks_y = mcu_rows
            .checked_mul(component.v)
            .ok_or_else(|| Error::limit_exceeded("JPEG block row count overflow"))?;
        component.plane_width = blocks_x
            .checked_mul(8)
            .ok_or_else(|| Error::limit_exceeded("JPEG component width overflow"))?;
        component.plane_height = blocks_y
            .checked_mul(8)
            .ok_or_else(|| Error::limit_exceeded("JPEG component height overflow"))?;
        let blocks = blocks_x
            .checked_mul(blocks_y)
            .ok_or_else(|| Error::limit_exceeded("JPEG block count overflow"))?;
        total_blocks = total_blocks
            .checked_add(blocks)
            .ok_or_else(|| Error::limit_exceeded("JPEG total block count overflow"))?;
        let bytes = component
            .plane_width
            .checked_mul(component.plane_height)
            .ok_or_else(|| Error::limit_exceeded("JPEG component plane size overflow"))?;
        plane_bytes = plane_bytes
            .checked_add(bytes)
            .ok_or_else(|| Error::limit_exceeded("JPEG plane memory overflow"))?;
    }
    if total_blocks > MAX_BLOCKS {
        return Err(Error::limit_exceeded(format!(
            "JPEG has {total_blocks} blocks, above the {MAX_BLOCKS}-block cap"
        )));
    }
    let channels = match component_count {
        1 => 1usize,
        3 => 3usize,
        _ => {
            return Err(Error::unsupported(format!(
                "JPEG has {component_count} components; only grayscale and three-component images are implemented"
            )));
        }
    };
    let output_bytes = pixels
        .checked_mul(channels)
        .ok_or_else(|| Error::limit_exceeded("JPEG output size overflow"))?;
    let decoded_bytes = plane_bytes
        .checked_add(output_bytes)
        .ok_or_else(|| Error::limit_exceeded("JPEG decoded memory overflow"))?;
    if decoded_bytes > MAX_DECODED_BYTES {
        return Err(Error::limit_exceeded(format!(
            "JPEG needs {decoded_bytes} decoded bytes, above the {MAX_DECODED_BYTES}-byte cap"
        )));
    }

    Ok(Frame {
        width,
        height,
        components,
        max_h,
        max_v,
        mcu_cols,
        mcu_rows,
        adobe_transform: None,
    })
}

fn parse_dqt(data: &[u8], tables: &mut [Option<[u16; 64]>; 4]) -> Result<()> {
    let mut position = 0usize;
    while position < data.len() {
        let info = data[position];
        position += 1;
        let precision = info >> 4;
        let identifier = usize::from(info & 0x0F);
        if precision > 1 || identifier > 3 {
            return Err(Error::corrupt(format!(
                "JPEG quantization table header 0x{info:02X} is invalid"
            )));
        }
        if precision != 0 {
            return Err(Error::unsupported(
                "16-bit JPEG quantization tables are not implemented",
            ));
        }
        if position + 64 > data.len() {
            return Err(Error::corrupt("truncated JPEG quantization table"));
        }
        let mut values = [0u16; 64];
        for (zigzag, &value) in data[position..position + 64].iter().enumerate() {
            if value == 0 {
                return Err(Error::corrupt(
                    "JPEG quantization table contains a zero value",
                ));
            }
            values[ZIGZAG[zigzag]] = u16::from(value);
        }
        position += 64;
        tables[identifier] = Some(values);
    }
    Ok(())
}

fn parse_dri(data: &[u8]) -> Result<usize> {
    if data.len() != 2 {
        return Err(Error::corrupt("JPEG DRI segment is not two bytes"));
    }
    Ok(usize::from(u16::from_be_bytes([data[0], data[1]])))
}

fn parse_app14(data: &[u8]) -> Result<Option<u8>> {
    if data.len() < 5 || &data[..5] != b"Adobe" {
        return Ok(None);
    }
    if data.len() < 12 {
        return Err(Error::corrupt("truncated Adobe APP14 segment"));
    }
    Ok(Some(data[11]))
}

fn validate_color_transform(frame: &Frame) -> Result<()> {
    if frame.components.len() != 3 {
        return Ok(());
    }
    match frame.adobe_transform {
        None | Some(1) => Ok(()),
        Some(0) => {
            if rgb_component_indices(frame).is_some() {
                Ok(())
            } else {
                Err(Error::unsupported(
                    "Adobe RGB JPEG with numeric component ids is not implemented",
                ))
            }
        }
        Some(2) => Err(Error::unsupported(
            "Adobe YCCK/CMYK JPEG is not implemented",
        )),
        Some(other) => Err(Error::unsupported(format!(
            "Adobe JPEG color transform {other} is not implemented"
        ))),
    }
}

fn parse_dht(
    data: &[u8],
    dc_tables: &mut [Option<HuffmanTable>; 4],
    ac_tables: &mut [Option<HuffmanTable>; 4],
) -> Result<()> {
    let mut position = 0usize;
    while position < data.len() {
        let info = data[position];
        position += 1;
        let class = info >> 4;
        let identifier = usize::from(info & 0x0F);
        if class > 1 || identifier > 3 {
            return Err(Error::corrupt(format!(
                "JPEG Huffman table header 0x{info:02X} is invalid"
            )));
        }
        if position + 16 > data.len() {
            return Err(Error::corrupt("truncated JPEG Huffman counts"));
        }
        let mut counts = [0u8; 17];
        let mut total = 0usize;
        for (offset, &count) in data[position..position + 16].iter().enumerate() {
            counts[offset + 1] = count;
            total = total
                .checked_add(usize::from(count))
                .ok_or_else(|| Error::limit_exceeded("JPEG Huffman symbol count overflow"))?;
        }
        position += 16;
        if total == 0 || total > 256 {
            return Err(Error::corrupt(format!(
                "JPEG Huffman table has {total} symbols"
            )));
        }
        if position + total > data.len() {
            return Err(Error::corrupt("truncated JPEG Huffman symbols"));
        }
        let symbols = data[position..position + total].to_vec();
        position += total;
        let table = HuffmanTable::build(counts, symbols)?;
        if class == 0 {
            dc_tables[identifier] = Some(table);
        } else {
            ac_tables[identifier] = Some(table);
        }
    }
    Ok(())
}

#[derive(Debug)]
struct HuffmanTable {
    symbols: Vec<u8>,
    min_code: [i32; 17],
    max_code: [i32; 17],
    value_index: [usize; 17],
}

impl HuffmanTable {
    fn build(counts: [u8; 17], symbols: Vec<u8>) -> Result<Self> {
        let mut min_code = [0i32; 17];
        let mut max_code = [0i32; 17];
        let mut value_index = [0usize; 17];
        let mut code: u32 = 0;
        let mut index = 0usize;
        for (length, &count) in counts.iter().enumerate().skip(1) {
            let count = u32::from(count);
            if code
                .checked_add(count)
                .is_none_or(|next| next > (1u32 << length))
            {
                return Err(Error::corrupt(
                    "JPEG Huffman code lengths exceed the available code space",
                ));
            }
            min_code[length] = code as i32;
            max_code[length] = (code + count) as i32 - 1;
            value_index[length] = index;
            index += count as usize;
            code = (code + count) << 1;
        }
        if index != symbols.len() {
            return Err(Error::corrupt(
                "JPEG Huffman symbol count does not match its code lengths",
            ));
        }
        Ok(Self {
            symbols,
            min_code,
            max_code,
            value_index,
        })
    }

    fn decode(&self, reader: &mut EntropyReader<'_>) -> Result<u8> {
        let mut code: u32 = 0;
        for length in 1..=16 {
            code = (code << 1) | reader.read_bit()?;
            let min = self.min_code[length];
            let max = self.max_code[length];
            if (code as i32) >= min && (code as i32) <= max {
                let offset = (code as i32 - min) as usize;
                let index = self.value_index[length] + offset;
                return self.symbols.get(index).copied().ok_or_else(|| {
                    Error::corrupt("JPEG Huffman table symbol index is out of bounds")
                });
            }
        }
        Err(Error::corrupt(
            "JPEG entropy data contains an invalid Huffman code",
        ))
    }
}

#[derive(Debug, Clone, Copy)]
struct ScanComponent {
    component_index: usize,
    dc_table: usize,
    ac_table: usize,
}

fn parse_sos(data: &[u8], frame: &Frame) -> Result<Vec<ScanComponent>> {
    if data.len() < 4 {
        return Err(Error::corrupt("JPEG SOS segment is too short"));
    }
    let count = usize::from(data[0]);
    if data.len() != 1 + 2 * count + 3 {
        return Err(Error::corrupt(
            "JPEG SOS component count does not match its length",
        ));
    }
    if count == 0 || count > MAX_COMPONENTS {
        return Err(Error::corrupt(format!("JPEG scan has {count} components")));
    }
    if count != frame.components.len() {
        return Err(Error::unsupported(
            "non-interleaved JPEG scans are not implemented; every frame component must appear in one interleaved scan",
        ));
    }
    let mut seen = vec![false; frame.components.len()];
    let mut scan = Vec::with_capacity(count);
    for index in 0..count {
        let id = data[1 + 2 * index];
        let tables = data[2 + 2 * index];
        let dc_table = usize::from(tables >> 4);
        let ac_table = usize::from(tables & 0x0F);
        if dc_table > 3 || ac_table > 3 {
            return Err(Error::corrupt(format!(
                "JPEG scan component {id} references Huffman table above 3"
            )));
        }
        let component_index = frame
            .components
            .iter()
            .position(|component| component.id == id)
            .ok_or_else(|| {
                Error::corrupt(format!("JPEG scan references unknown component {id}"))
            })?;
        if seen[component_index] {
            return Err(Error::corrupt(format!(
                "JPEG scan contains component {id} more than once"
            )));
        }
        seen[component_index] = true;
        scan.push(ScanComponent {
            component_index,
            dc_table,
            ac_table,
        });
    }
    if seen.iter().any(|seen| !seen) {
        return Err(Error::corrupt(
            "JPEG scan does not contain every frame component",
        ));
    }
    let spectral_start = data[1 + 2 * count];
    let spectral_end = data[2 + 2 * count];
    let approximation = data[3 + 2 * count];
    if spectral_start != 0 || spectral_end != 63 || approximation != 0 {
        return Err(Error::unsupported(
            "only baseline sequential JPEG scans with Ss=0, Se=63, and Ah=Al=0 are implemented",
        ));
    }
    Ok(scan)
}

struct EntropyReader<'a> {
    data: &'a [u8],
    position: usize,
    current: u8,
    bits_left: u8,
}

impl<'a> EntropyReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            position: 0,
            current: 0,
            bits_left: 0,
        }
    }

    fn read_bit(&mut self) -> Result<u32> {
        if self.bits_left == 0 {
            self.fill()?;
        }
        self.bits_left -= 1;
        Ok(u32::from((self.current >> self.bits_left) & 1))
    }

    fn read_bits(&mut self, count: u32) -> Result<u32> {
        let mut value = 0u32;
        for _ in 0..count {
            value = (value << 1) | self.read_bit()?;
        }
        Ok(value)
    }

    fn fill(&mut self) -> Result<()> {
        if self.position >= self.data.len() {
            return Err(Error::corrupt("truncated JPEG entropy data"));
        }
        let byte = self.data[self.position];
        self.position += 1;
        if byte == 0xFF {
            if self.position >= self.data.len() {
                return Err(Error::corrupt("truncated JPEG byte stuffing"));
            }
            let next = self.data[self.position];
            if next == 0x00 {
                self.position += 1;
            } else {
                return Err(Error::corrupt(
                    "JPEG entropy data ends before all blocks are decoded",
                ));
            }
        }
        self.current = byte;
        self.bits_left = 8;
        Ok(())
    }

    fn align(&mut self) {
        self.bits_left = 0;
    }

    fn position(&self) -> usize {
        self.position
    }

    fn consume_restart(&mut self, expected: u8) -> Result<()> {
        self.align();
        while self.position < self.data.len() && self.data[self.position] == 0xFF {
            self.position += 1;
        }
        if self.position >= self.data.len() {
            return Err(Error::corrupt("truncated JPEG restart marker"));
        }
        let marker = self.data[self.position];
        self.position += 1;
        if marker != 0xD0 + expected {
            return Err(Error::corrupt(format!(
                "expected JPEG restart marker RST{}, found 0x{marker:02X}",
                expected
            )));
        }
        Ok(())
    }
}

fn decode_scan(
    entropy: &[u8],
    frame: &Frame,
    scan: &[ScanComponent],
    quant_tables: &[Option<[u16; 64]>; 4],
    dc_tables: &[Option<HuffmanTable>; 4],
    ac_tables: &[Option<HuffmanTable>; 4],
    restart_interval: usize,
) -> Result<(Image, usize)> {
    let basis = build_idct_basis();
    let mut planes = Vec::with_capacity(frame.components.len());
    for component in &frame.components {
        let size = component
            .plane_width
            .checked_mul(component.plane_height)
            .ok_or_else(|| Error::limit_exceeded("JPEG component plane size overflow"))?;
        planes.push(vec![0u8; size]);
    }
    let mut dc_predictors = vec![0i32; frame.components.len()];
    let mut reader = EntropyReader::new(entropy);
    let mut restart_index = 0u8;
    let mut mcu_index = 0usize;

    for mcu_y in 0..frame.mcu_rows {
        for mcu_x in 0..frame.mcu_cols {
            if restart_interval != 0 && mcu_index != 0 && mcu_index.is_multiple_of(restart_interval)
            {
                reader.consume_restart(restart_index)?;
                restart_index = (restart_index + 1) & 7;
                dc_predictors.fill(0);
            }
            for scan_component in scan {
                let component = &frame.components[scan_component.component_index];
                let quant = quant_tables[component.quant].as_ref().ok_or_else(|| {
                    Error::internal("JPEG quantization table disappeared after validation")
                })?;
                let dc_table = dc_tables[scan_component.dc_table].as_ref().ok_or_else(|| {
                    Error::internal("JPEG DC Huffman table disappeared after validation")
                })?;
                let ac_table = ac_tables[scan_component.ac_table].as_ref().ok_or_else(|| {
                    Error::internal("JPEG AC Huffman table disappeared after validation")
                })?;
                for block_y in 0..component.v {
                    for block_x in 0..component.h {
                        let x = mcu_x * component.h + block_x;
                        let y = mcu_y * component.v + block_y;
                        let mut coefficients = [0i32; 64];
                        decode_block(
                            &mut reader,
                            dc_table,
                            ac_table,
                            quant,
                            &mut dc_predictors[scan_component.component_index],
                            &mut coefficients,
                        )?;
                        let mut block = [0u8; 64];
                        inverse_dct(&coefficients, &basis, &mut block);
                        store_block(
                            &mut planes[scan_component.component_index],
                            component.plane_width,
                            x,
                            y,
                            &block,
                        );
                    }
                }
            }
            mcu_index += 1;
        }
    }

    reader.align();
    let consumed = reader.position();
    let image = compose_image(frame, &planes)?;
    Ok((image, consumed))
}

fn decode_block(
    reader: &mut EntropyReader<'_>,
    dc_table: &HuffmanTable,
    ac_table: &HuffmanTable,
    quant: &[u16; 64],
    dc_predictor: &mut i32,
    coefficients: &mut [i32; 64],
) -> Result<()> {
    coefficients.fill(0);
    let dc_size = u32::from(dc_table.decode(reader)?);
    if dc_size > 11 {
        return Err(Error::corrupt(format!(
            "JPEG DC coefficient category {dc_size} is above 11"
        )));
    }
    let dc_difference = if dc_size == 0 {
        0
    } else {
        extend_sign(reader.read_bits(dc_size)?, dc_size)
    };
    *dc_predictor = dc_predictor
        .checked_add(dc_difference)
        .ok_or_else(|| Error::corrupt("JPEG DC predictor overflow"))?;
    coefficients[0] = dc_predictor
        .checked_mul(i32::from(quant[0]))
        .ok_or_else(|| Error::corrupt("JPEG DC coefficient overflow"))?;

    let mut zigzag = 1usize;
    while zigzag < 64 {
        let symbol = ac_table.decode(reader)?;
        let run = usize::from(symbol >> 4);
        let size = u32::from(symbol & 0x0F);
        if size == 0 {
            if run == 0 {
                break;
            }
            if run == 15 {
                zigzag = zigzag
                    .checked_add(16)
                    .ok_or_else(|| Error::corrupt("JPEG AC run overflow"))?;
                if zigzag > 64 {
                    return Err(Error::corrupt("JPEG AC zero run exceeds 63 coefficients"));
                }
                continue;
            }
            return Err(Error::corrupt(format!(
                "JPEG AC symbol 0x{symbol:02X} is invalid"
            )));
        }
        if size > 10 {
            return Err(Error::corrupt(format!(
                "JPEG AC coefficient size {size} is above 10"
            )));
        }
        zigzag = zigzag
            .checked_add(run)
            .ok_or_else(|| Error::corrupt("JPEG AC run overflow"))?;
        if zigzag >= 64 {
            return Err(Error::corrupt("JPEG AC run exceeds 63 coefficients"));
        }
        let value = extend_sign(reader.read_bits(size)?, size);
        let natural = ZIGZAG[zigzag];
        coefficients[natural] = value
            .checked_mul(i32::from(quant[natural]))
            .ok_or_else(|| Error::corrupt("JPEG AC coefficient overflow"))?;
        zigzag += 1;
    }
    Ok(())
}

fn extend_sign(value: u32, size: u32) -> i32 {
    let threshold = 1u32 << (size - 1);
    let value = value as i32;
    if value < threshold as i32 {
        value - ((1i32 << size) - 1)
    } else {
        value
    }
}

fn build_idct_basis() -> [[f32; 8]; 8] {
    let mut basis = [[0.0f32; 8]; 8];
    for (x, basis_row) in basis.iter_mut().enumerate() {
        for (u, value) in basis_row.iter_mut().enumerate() {
            let scale = if u == 0 {
                std::f32::consts::FRAC_1_SQRT_2
            } else {
                1.0
            };
            let angle = ((2 * x + 1) as f32 * u as f32 * std::f32::consts::PI) / 16.0;
            *value = scale * angle.cos();
        }
    }
    basis
}

fn inverse_dct(coefficients: &[i32; 64], basis: &[[f32; 8]; 8], output: &mut [u8; 64]) {
    let mut intermediate = [[0.0f32; 8]; 8];
    for (v, coefficient_row) in coefficients.as_chunks::<8>().0.iter().enumerate() {
        for (x, basis_row) in basis.iter().enumerate() {
            let sum = coefficient_row
                .iter()
                .zip(basis_row)
                .map(|(coefficient, basis)| *coefficient as f32 * basis)
                .sum();
            intermediate[v][x] = sum;
        }
    }
    for (y, basis_row) in basis.iter().enumerate() {
        for (x, sample) in output[y * 8..y * 8 + 8].iter_mut().enumerate() {
            let sum = intermediate
                .iter()
                .zip(basis_row)
                .map(|(row, basis)| row[x] * basis)
                .sum::<f32>();
            *sample = clamp_sample(0.25 * sum + 128.0);
        }
    }
}

fn clamp_sample(value: f32) -> u8 {
    let rounded = value.round();
    if rounded <= 0.0 {
        0
    } else if rounded >= 255.0 {
        255
    } else {
        rounded as u8
    }
}

fn store_block(plane: &mut [u8], stride: usize, block_x: usize, block_y: usize, block: &[u8; 64]) {
    for row in 0..8 {
        let destination = (block_y * 8 + row) * stride + block_x * 8;
        plane[destination..destination + 8].copy_from_slice(&block[row * 8..row * 8 + 8]);
    }
}

fn compose_image(frame: &Frame, planes: &[Vec<u8>]) -> Result<Image> {
    let channels = match frame.components.len() {
        1 => 1usize,
        3 => 3usize,
        _ => {
            return Err(Error::internal(
                "JPEG frame component count changed after validation",
            ));
        }
    };
    let expected = frame
        .width
        .checked_mul(frame.height)
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or_else(|| Error::limit_exceeded("JPEG output size overflow"))?;
    let mut pixels = Vec::with_capacity(expected);
    let rgb_indices = if channels == 3 {
        match frame.adobe_transform {
            Some(0) => rgb_component_indices(frame),
            Some(1) => None,
            _ => rgb_component_indices(frame),
        }
    } else {
        None
    };

    for y in 0..frame.height {
        for x in 0..frame.width {
            if channels == 1 {
                let component = &frame.components[0];
                let sample =
                    sample_component(&planes[0], component, frame.max_h, frame.max_v, x, y);
                pixels.push(sample);
            } else if let Some([red, green, blue]) = rgb_indices {
                let r = sample_component(
                    &planes[red],
                    &frame.components[red],
                    frame.max_h,
                    frame.max_v,
                    x,
                    y,
                );
                let g = sample_component(
                    &planes[green],
                    &frame.components[green],
                    frame.max_h,
                    frame.max_v,
                    x,
                    y,
                );
                let b = sample_component(
                    &planes[blue],
                    &frame.components[blue],
                    frame.max_h,
                    frame.max_v,
                    x,
                    y,
                );
                pixels.extend_from_slice(&[r, g, b]);
            } else {
                let y_sample = sample_component(
                    &planes[0],
                    &frame.components[0],
                    frame.max_h,
                    frame.max_v,
                    x,
                    y,
                );
                let cb_sample = sample_component(
                    &planes[1],
                    &frame.components[1],
                    frame.max_h,
                    frame.max_v,
                    x,
                    y,
                );
                let cr_sample = sample_component(
                    &planes[2],
                    &frame.components[2],
                    frame.max_h,
                    frame.max_v,
                    x,
                    y,
                );
                let (r, g, b) = ycbcr_to_rgb(y_sample, cb_sample, cr_sample);
                pixels.extend_from_slice(&[r, g, b]);
            }
        }
    }

    Image::new(frame.width, frame.height, channels as u8, pixels)
}

fn rgb_component_indices(frame: &Frame) -> Option<[usize; 3]> {
    let mut red = None;
    let mut green = None;
    let mut blue = None;
    for (index, component) in frame.components.iter().enumerate() {
        match component.id {
            b'R' | b'r' => red = Some(index),
            b'G' | b'g' => green = Some(index),
            b'B' | b'b' => blue = Some(index),
            _ => {}
        }
    }
    Some([red?, green?, blue?])
}

fn sample_component(
    plane: &[u8],
    component: &FrameComponent,
    max_h: usize,
    max_v: usize,
    x: usize,
    y: usize,
) -> u8 {
    let scale_x = component.h as f32 / max_h as f32;
    let scale_y = component.v as f32 / max_v as f32;
    let source_x = (x as f32 + 0.5) * scale_x - 0.5;
    let source_y = (y as f32 + 0.5) * scale_y - 0.5;
    let x0 = source_x
        .floor()
        .clamp(0.0, (component.plane_width - 1) as f32) as usize;
    let y0 = source_y
        .floor()
        .clamp(0.0, (component.plane_height - 1) as f32) as usize;
    let x1 = (x0 + 1).min(component.plane_width - 1);
    let y1 = (y0 + 1).min(component.plane_height - 1);
    let fx = (source_x - x0 as f32).clamp(0.0, 1.0);
    let fy = (source_y - y0 as f32).clamp(0.0, 1.0);
    let top_left = f32::from(plane[y0 * component.plane_width + x0]);
    let top_right = f32::from(plane[y0 * component.plane_width + x1]);
    let bottom_left = f32::from(plane[y1 * component.plane_width + x0]);
    let bottom_right = f32::from(plane[y1 * component.plane_width + x1]);
    let top = top_left * (1.0 - fx) + top_right * fx;
    let bottom = bottom_left * (1.0 - fx) + bottom_right * fx;
    clamp_sample(top * (1.0 - fy) + bottom * fy)
}

fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    let y = f32::from(y);
    let cb = f32::from(cb) - 128.0;
    let cr = f32::from(cr) - 128.0;
    let red = y + 1.402 * cr;
    let green = y - 0.344_136 * cb - 0.714_136 * cr;
    let blue = y + 1.772 * cb;
    (clamp_sample(red), clamp_sample(green), clamp_sample(blue))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::png;

    fn segment(marker: u8, payload: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0xFF, marker];
        let length = u16::try_from(payload.len() + 2).expect("test segment fits in u16");
        bytes.extend_from_slice(&length.to_be_bytes());
        bytes.extend_from_slice(payload);
        bytes
    }

    fn sof(width: u16, height: u16, components: &[(u8, u8, u8)]) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.push(8);
        payload.extend_from_slice(&height.to_be_bytes());
        payload.extend_from_slice(&width.to_be_bytes());
        payload.push(u8::try_from(components.len()).expect("test component count fits in u8"));
        for &(id, sampling, quant) in components {
            payload.push(id);
            payload.push(sampling);
            payload.push(quant);
        }
        segment(0xC0, &payload)
    }

    fn dqt() -> Vec<u8> {
        let mut payload = vec![0u8];
        payload.extend_from_slice(&[1u8; 64]);
        segment(0xDB, &payload)
    }

    fn dht(class: u8, identifier: u8, counts: [u8; 16], symbols: &[u8]) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.push((class << 4) | identifier);
        payload.extend_from_slice(&counts);
        payload.extend_from_slice(symbols);
        segment(0xC4, &payload)
    }

    fn sos(
        components: &[(u8, u8)],
        spectral_start: u8,
        spectral_end: u8,
        approximation: u8,
    ) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.push(u8::try_from(components.len()).expect("test scan count fits in u8"));
        for &(id, tables) in components {
            payload.push(id);
            payload.push(tables);
        }
        payload.push(spectral_start);
        payload.push(spectral_end);
        payload.push(approximation);
        segment(0xDA, &payload)
    }

    fn jpeg(parts: &[Vec<u8>], entropy: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0xFF, 0xD8];
        for part in parts {
            bytes.extend_from_slice(part);
        }
        bytes.extend_from_slice(entropy);
        bytes.extend_from_slice(&[0xFF, 0xD9]);
        bytes
    }

    fn dc_only_jpeg(
        width: u16,
        height: u16,
        entropy: &[u8],
        dc_counts: [u8; 16],
        dc_symbols: &[u8],
        restart_interval: Option<u16>,
    ) -> Vec<u8> {
        let mut parts = vec![
            sof(width, height, &[(1, 0x11, 0)]),
            dqt(),
            dht(0, 0, dc_counts, dc_symbols),
            dht(
                1,
                0,
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                &[0x00],
            ),
        ];
        if let Some(interval) = restart_interval {
            parts.push(segment(0xDD, &interval.to_be_bytes()));
        }
        parts.push(sos(&[(1, 0x00)], 0, 63, 0));
        jpeg(&parts, entropy)
    }

    fn reference(name: &str) -> Image {
        let bytes: &[u8] = match name {
            "gray_2x2" => include_bytes!("../../fixtures/media/gray_2x2_ref.png"),
            "rgb_2x2" => include_bytes!("../../fixtures/media/rgb_2x2_ref.png"),
            "gradient_16x16" => include_bytes!("../../fixtures/media/gradient_16x16_ref.png"),
            "quadrants_16x16" => include_bytes!("../../fixtures/media/quadrants_16x16_ref.png"),
            "quadrants_16x16_444" => {
                include_bytes!("../../fixtures/media/quadrants_16x16_444_ref.png")
            }
            _ => panic!("unknown reference fixture {name}"),
        };
        png::decode(bytes).expect("reference PNG must decode")
    }

    fn assert_close(actual: &Image, expected: &Image, tolerance: i32, label: &str) {
        assert_eq!(actual.width(), expected.width(), "{label}: width");
        assert_eq!(actual.height(), expected.height(), "{label}: height");
        assert_eq!(actual.channels(), expected.channels(), "{label}: channels");
        for (index, (actual, expected)) in actual
            .pixels()
            .iter()
            .zip(expected.pixels().iter())
            .enumerate()
        {
            let difference = i32::from(*actual) - i32::from(*expected);
            assert!(
                difference.abs() <= tolerance,
                "{label}: byte {index} differs by {difference} (actual {actual}, expected {expected})"
            );
        }
    }

    #[test]
    fn decodes_a_committed_grayscale_fixture() {
        let image = decode(include_bytes!("../../fixtures/media/gray_2x2.jpg"))
            .expect("grayscale JPEG fixture");
        assert_eq!(image.width(), 2);
        assert_eq!(image.height(), 2);
        assert_eq!(image.channels(), 1);
        assert_close(&image, &reference("gray_2x2"), 2, "gray_2x2");
    }

    #[test]
    fn decodes_a_committed_rgb_fixture() {
        let image =
            decode(include_bytes!("../../fixtures/media/rgb_2x2.jpg")).expect("RGB JPEG fixture");
        assert_eq!(image.width(), 2);
        assert_eq!(image.height(), 2);
        assert_eq!(image.channels(), 3);
        assert_close(&image, &reference("rgb_2x2"), 4, "rgb_2x2");
    }

    #[test]
    fn decodes_a_committed_420_gradient_fixture() {
        let image = decode(include_bytes!("../../fixtures/media/gradient_16x16.jpg"))
            .expect("gradient JPEG fixture");
        assert_eq!(image.width(), 16);
        assert_eq!(image.height(), 16);
        assert_eq!(image.channels(), 3);
        assert_close(&image, &reference("gradient_16x16"), 24, "gradient_16x16");
    }

    #[test]
    fn decodes_a_committed_420_color_fixture() {
        let image = decode(include_bytes!("../../fixtures/media/quadrants_16x16.jpg"))
            .expect("quadrant JPEG fixture");
        assert_eq!(image.channels(), 3);
        assert_close(&image, &reference("quadrants_16x16"), 24, "quadrants_16x16");
    }

    #[test]
    fn decodes_a_committed_444_color_fixture() {
        let image = decode(include_bytes!(
            "../../fixtures/media/quadrants_16x16_444.jpg"
        ))
        .expect("4:4:4 JPEG fixture");
        assert_eq!(image.channels(), 3);
        assert_close(
            &image,
            &reference("quadrants_16x16_444"),
            3,
            "quadrants_16x16_444",
        );
    }

    #[test]
    fn decode_file_reads_a_committed_fixture() {
        let image = decode_file(Path::new("fixtures/media/quadrants_16x16_444.jpg"))
            .expect("committed JPEG fixture");
        assert_eq!(image.width(), 16);
        assert_eq!(image.height(), 16);
    }

    #[test]
    fn decodes_a_dc_only_block_to_a_constant_image() {
        let bytes = dc_only_jpeg(
            8,
            8,
            &[0x00],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            &[0],
            None,
        );
        let image = decode(&bytes).expect("DC-only JPEG");
        assert_eq!(image.channels(), 1);
        assert_eq!(image.pixels(), &[128u8; 64]);
    }

    #[test]
    fn decodes_a_nonzero_dc_coefficient_exactly() {
        let bytes = dc_only_jpeg(
            8,
            8,
            &[0x60],
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            &[0, 4],
            None,
        );
        let image = decode(&bytes).expect("nonzero DC-only JPEG");
        assert_eq!(image.pixels(), &[129u8; 64]);
    }

    #[test]
    fn decodes_restart_intervals_and_resets_predictors() {
        let entropy = [0x00, 0xFF, 0xD0, 0x00, 0xFF, 0xD1, 0x00, 0xFF, 0xD2, 0x00];
        let bytes = dc_only_jpeg(
            16,
            16,
            &entropy,
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            &[0],
            Some(1),
        );
        let image = decode(&bytes).expect("JPEG with restart markers");
        assert_eq!(image.pixels(), &[128u8; 256]);
    }

    #[test]
    fn rejects_a_wrong_restart_marker() {
        let entropy = [0x00, 0xFF, 0xD3, 0x00, 0xFF, 0xD1, 0x00, 0xFF, 0xD2, 0x00];
        let bytes = dc_only_jpeg(
            16,
            16,
            &entropy,
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            &[0],
            Some(1),
        );
        let error = decode(&bytes).expect_err("wrong restart marker must fail");
        assert!(matches!(error, Error::Corrupt(_)), "{error}");
    }

    #[test]
    fn rejects_a_missing_restart_marker() {
        let entropy = [0x00, 0x00, 0x00, 0x00];
        let bytes = dc_only_jpeg(
            16,
            16,
            &entropy,
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            &[0],
            Some(1),
        );
        let error = decode(&bytes).expect_err("missing restart marker must fail");
        assert!(matches!(error, Error::Corrupt(_)), "{error}");
    }

    #[test]
    fn rejects_a_bad_soi_and_trailing_bytes() {
        let mut bytes = include_bytes!("../../fixtures/media/quadrants_16x16_444.jpg").to_vec();
        bytes[0] ^= 0xFF;
        let error = decode(&bytes).expect_err("bad SOI");
        assert!(matches!(error, Error::Corrupt(_)), "{error}");

        let mut bytes = include_bytes!("../../fixtures/media/quadrants_16x16_444.jpg").to_vec();
        bytes.push(0);
        let error = decode(&bytes).expect_err("trailing byte");
        assert!(matches!(error, Error::Corrupt(_)), "{error}");
    }

    #[test]
    fn rejects_every_truncation_of_a_valid_fixture() {
        let bytes = include_bytes!("../../fixtures/media/quadrants_16x16_444.jpg");
        for cut in 0..bytes.len() {
            let error = decode(&bytes[..cut]).expect_err("truncated JPEG");
            assert!(matches!(error, Error::Corrupt(_)), "cut {cut}: {error}");
        }
    }

    #[test]
    fn rejects_progressive_and_unsupported_frame_markers() {
        let mut bytes = dc_only_jpeg(
            8,
            8,
            &[0x00],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            &[0],
            None,
        );
        let sof_position = bytes
            .windows(2)
            .position(|window| window == [0xFF, 0xC0])
            .expect("SOF marker");
        bytes[sof_position + 1] = 0xC2;
        let error = decode(&bytes).expect_err("progressive JPEG");
        assert!(matches!(error, Error::Unsupported(_)), "{error}");

        let bytes = jpeg(&[segment(0xC3, &[8, 0, 8, 0, 8, 1, 1, 0x11, 0])], &[]);
        let error = decode(&bytes).expect_err("lossless JPEG");
        assert!(matches!(error, Error::Unsupported(_)), "{error}");
    }

    #[test]
    fn rejects_non_eight_bit_precision_and_too_many_components() {
        let mut bytes = dc_only_jpeg(
            8,
            8,
            &[0x00],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            &[0],
            None,
        );
        let sof_position = bytes
            .windows(2)
            .position(|window| window == [0xFF, 0xC0])
            .expect("SOF marker");
        bytes[sof_position + 4] = 12;
        let error = decode(&bytes).expect_err("12-bit JPEG");
        assert!(matches!(error, Error::Unsupported(_)), "{error}");

        let bytes = jpeg(
            &[sof(
                8,
                8,
                &[(1, 0x11, 0), (2, 0x11, 1), (3, 0x11, 2), (4, 0x11, 3)],
            )],
            &[],
        );
        let error = decode(&bytes).expect_err("four-component JPEG");
        assert!(matches!(error, Error::Unsupported(_)), "{error}");
    }

    #[test]
    fn rejects_16_bit_quantization_and_invalid_huffman_tables() {
        let mut payload = vec![0x10u8];
        payload.extend_from_slice(&[1u8; 128]);
        let bytes = jpeg(&[segment(0xDB, &payload)], &[]);
        let error = decode(&bytes).expect_err("16-bit quantization");
        assert!(matches!(error, Error::Unsupported(_)), "{error}");

        let bytes = jpeg(
            &[dht(
                0,
                0,
                [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                &[0, 1, 2],
            )],
            &[],
        );
        let error = decode(&bytes).expect_err("invalid Huffman code space");
        assert!(matches!(error, Error::Corrupt(_)), "{error}");
    }

    #[test]
    fn rejects_missing_tables_and_invalid_scan_parameters() {
        let bytes = jpeg(
            &[sof(8, 8, &[(1, 0x11, 0)]), sos(&[(1, 0x00)], 0, 63, 0)],
            &[0x00],
        );
        let error = decode(&bytes).expect_err("missing quantization table");
        assert!(matches!(error, Error::Corrupt(_)), "{error}");

        let bytes = jpeg(
            &[
                sof(8, 8, &[(1, 0x11, 0)]),
                dqt(),
                dht(0, 0, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[0]),
                dht(1, 0, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[0]),
                sos(&[(1, 0x00)], 1, 63, 0),
            ],
            &[0x00],
        );
        let error = decode(&bytes).expect_err("non-baseline spectral selection");
        assert!(matches!(error, Error::Unsupported(_)), "{error}");

        let bytes = jpeg(
            &[
                sof(8, 8, &[(1, 0x22, 0), (2, 0x11, 1), (3, 0x11, 2)]),
                dqt(),
                dht(0, 0, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[0]),
                dht(1, 0, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[0]),
                sos(&[(1, 0x00)], 0, 63, 0),
            ],
            &[0x00],
        );
        let error = decode(&bytes).expect_err("non-interleaved scan");
        assert!(matches!(error, Error::Unsupported(_)), "{error}");
    }

    #[test]
    fn enforces_dimension_and_pixel_caps() {
        let bytes = jpeg(&[sof((MAX_WIDTH + 1) as u16, 1, &[(1, 0x11, 0)])], &[]);
        let error = decode(&bytes).expect_err("width cap");
        assert!(matches!(error, Error::LimitExceeded(_)), "{error}");

        let bytes = jpeg(&[sof(1, (MAX_HEIGHT + 1) as u16, &[(1, 0x11, 0)])], &[]);
        let error = decode(&bytes).expect_err("height cap");
        assert!(matches!(error, Error::LimitExceeded(_)), "{error}");

        let bytes = jpeg(&[sof(4096, 4097, &[(1, 0x11, 0)])], &[]);
        let error = decode(&bytes).expect_err("pixel cap");
        assert!(matches!(error, Error::LimitExceeded(_)), "{error}");
    }

    #[test]
    fn rejects_missing_eoi_and_missing_scan_data() {
        let mut bytes = dc_only_jpeg(
            8,
            8,
            &[0x00],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            &[0],
            None,
        );
        bytes.truncate(bytes.len() - 2);
        let error = decode(&bytes).expect_err("missing EOI");
        assert!(matches!(error, Error::Corrupt(_)), "{error}");

        let bytes = dc_only_jpeg(
            8,
            8,
            &[],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            &[0],
            None,
        );
        let error = decode(&bytes).expect_err("missing entropy data");
        assert!(matches!(error, Error::Corrupt(_)), "{error}");
    }

    #[test]
    fn accepts_an_adobe_ycbcr_transform() {
        let mut bytes = include_bytes!("../../fixtures/media/quadrants_16x16_444.jpg").to_vec();
        let app14 = segment(0xEE, b"Adobe\x00\x64\x00\x00\x00\x00\x01");
        bytes.splice(2..2, app14);
        let image = decode(&bytes).expect("Adobe YCbCr JPEG");
        assert_eq!(image.width(), 16);
        assert_eq!(image.channels(), 3);
    }

    #[test]
    fn rejects_unsupported_adobe_color_transforms() {
        let parts = vec![
            segment(0xEE, b"Adobe\x00\x64\x00\x00\x00\x00\x02"),
            sof(8, 8, &[(1, 0x11, 0), (2, 0x11, 1), (3, 0x11, 2)]),
            sos(&[(1, 0x00), (2, 0x11), (3, 0x11)], 0, 63, 0),
        ];
        let bytes = jpeg(&parts, &[0x00]);
        let error = decode(&bytes).expect_err("Adobe YCCK transform");
        assert!(matches!(error, Error::Unsupported(_)), "{error}");

        let parts = vec![
            segment(0xEE, b"Adobe\x00\x64\x00\x00\x00\x00\x00"),
            sof(8, 8, &[(1, 0x11, 0), (2, 0x11, 1), (3, 0x11, 2)]),
            sos(&[(1, 0x00), (2, 0x11), (3, 0x11)], 0, 63, 0),
        ];
        let bytes = jpeg(&parts, &[0x00]);
        let error = decode(&bytes).expect_err("Adobe RGB with numeric ids");
        assert!(matches!(error, Error::Unsupported(_)), "{error}");
    }
}
