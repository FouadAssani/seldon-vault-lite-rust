//! Native RIFF/WAVE PCM and IEEE-float decoder.
//!
//! Only the formats announced by the frozen processor are accepted: mono or
//! multichannel PCM with 8, 16, 24, or 32 bits per sample, and IEEE float with
//! 32 or 64 bits per sample. `WAVE_FORMAT_EXTENSIBLE` is accepted when its
//! sub-format is one of those two families. Every other codec (ADPCM, A-law,
//! mu-law, MP3, and so on) is rejected as unsupported instead of being guessed.
//!
//! All offsets and sizes are checked before any read. The file, data chunk,
//! channel count, sample rate, and frame count are capped so a hostile header
//! cannot make the decoder allocate or loop without bound. Channels are
//! downmixed by averaging, and the resulting mono buffer is returned at its
//! original rate; `media::audio` performs the model-rate conversion.

use std::path::Path;

use crate::binary::read_file_bounded;
use crate::error::{Error, Result};
use crate::media::audio::{Audio, MAX_AUDIO_CHANNELS, MAX_SAMPLE_RATE, MIN_SAMPLE_RATE};

/// Maximum WAV file size accepted from disk (64 MiB).
pub const MAX_WAV_FILE_BYTES: u64 = 64 * 1024 * 1024;

/// Maximum audio payload size in bytes.
pub const MAX_WAV_DATA_BYTES: usize = 64 * 1024 * 1024;

/// Maximum number of source frames (30 seconds at the 192 kHz cap).
pub const MAX_WAV_FRAMES: usize = 30 * MAX_SAMPLE_RATE as usize;

const WAVE_FORMAT_PCM: u16 = 1;
const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;

/// Decoded WAV payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SampleFormat {
    Pcm,
    Float,
}

#[derive(Debug, Clone, Copy)]
struct FormatChunk {
    format: SampleFormat,
    channels: u16,
    sample_rate: u32,
    block_align: u16,
    bits_per_sample: u16,
}

impl FormatChunk {
    fn bytes_per_sample(self) -> usize {
        usize::from(self.bits_per_sample / 8)
    }
}

/// Decodes a WAV file with the 64 MiB file cap.
pub fn decode_file(path: &Path) -> Result<Audio> {
    let bytes = read_file_bounded(path, MAX_WAV_FILE_BYTES)?;
    decode_bytes(&bytes)
}

/// Decodes an in-memory RIFF/WAVE file.
///
/// # Errors
///
/// Returns a corruption error for malformed or truncated RIFF structures and
/// inconsistent format fields, an unsupported error for a codec or bit depth
/// outside the announced matrix, and a limit error for an oversized file,
/// data chunk, frame count, channel count, or sample rate.
pub fn decode_bytes(bytes: &[u8]) -> Result<Audio> {
    if bytes.len() < 12 {
        return Err(Error::corrupt(
            "WAV file is shorter than the 12-byte RIFF header",
        ));
    }
    if &bytes[0..4] != b"RIFF" {
        return Err(Error::corrupt("WAV file does not start with RIFF"));
    }
    if &bytes[8..12] != b"WAVE" {
        return Err(Error::corrupt("RIFF container is not WAVE"));
    }
    let riff_size = read_u32(bytes, 4)? as usize;
    let riff_end = riff_size
        .checked_add(8)
        .ok_or_else(|| Error::corrupt("RIFF size overflow"))?;
    if riff_end > bytes.len() {
        return Err(Error::corrupt(format!(
            "RIFF declares {riff_end} bytes but the file has {}",
            bytes.len()
        )));
    }
    if riff_end < 12 {
        return Err(Error::corrupt("RIFF size is smaller than the WAVE header"));
    }
    if riff_end < bytes.len() {
        return Err(Error::corrupt(format!(
            "WAV has {} trailing bytes after the RIFF container",
            bytes.len() - riff_end
        )));
    }

    let mut format = None;
    let mut data = None;
    let mut offset = 12usize;
    while offset < riff_end {
        if offset + 8 > riff_end {
            return Err(Error::corrupt("truncated WAV chunk header"));
        }
        let id = &bytes[offset..offset + 4];
        let size = read_u32(bytes, offset + 4)? as usize;
        let payload_start = offset + 8;
        let payload_end = payload_start
            .checked_add(size)
            .ok_or_else(|| Error::corrupt("WAV chunk size overflow"))?;
        if payload_end > riff_end {
            return Err(Error::corrupt(format!(
                "WAV chunk at {offset} extends past the RIFF end"
            )));
        }
        match id {
            b"fmt " => {
                if format.is_some() {
                    return Err(Error::corrupt("WAV has more than one fmt chunk"));
                }
                format = Some(parse_format(&bytes[payload_start..payload_end])?);
            }
            b"data" => {
                if data.is_some() {
                    return Err(Error::corrupt("WAV has more than one data chunk"));
                }
                if size > MAX_WAV_DATA_BYTES {
                    return Err(Error::limit_exceeded(format!(
                        "WAV data chunk has {size} bytes, above the {MAX_WAV_DATA_BYTES}-byte cap"
                    )));
                }
                data = Some((payload_start, size));
            }
            _ => {}
        }
        let padded_end = payload_end + (size & 1);
        if padded_end > riff_end && payload_end != riff_end {
            return Err(Error::corrupt(
                "WAV chunk padding extends past the RIFF end",
            ));
        }
        offset = padded_end.min(riff_end);
    }

    let format = format.ok_or_else(|| Error::corrupt("WAV has no fmt chunk"))?;
    let (data_start, data_size) = data.ok_or_else(|| Error::corrupt("WAV has no data chunk"))?;
    let block_align = usize::from(format.block_align);
    if block_align == 0 {
        return Err(Error::corrupt("WAV block alignment is zero"));
    }
    if data_size % block_align != 0 {
        return Err(Error::corrupt(format!(
            "WAV data size {data_size} is not a multiple of block alignment {block_align}"
        )));
    }
    let frames = data_size / block_align;
    if frames == 0 {
        return Err(Error::corrupt("WAV data chunk contains no frames"));
    }
    if frames > MAX_WAV_FRAMES {
        return Err(Error::limit_exceeded(format!(
            "WAV has {frames} frames, above the {MAX_WAV_FRAMES}-frame cap"
        )));
    }

    decode_samples(bytes, data_start, frames, format)
}

fn parse_format(payload: &[u8]) -> Result<FormatChunk> {
    if payload.len() < 16 {
        return Err(Error::corrupt("WAV fmt chunk is shorter than 16 bytes"));
    }
    let mut format_tag = read_u16(payload, 0)?;
    let channels = read_u16(payload, 2)?;
    let sample_rate = read_u32(payload, 4)?;
    let byte_rate = read_u32(payload, 8)?;
    let block_align = read_u16(payload, 12)?;
    let bits_per_sample = read_u16(payload, 14)?;

    if format_tag == WAVE_FORMAT_EXTENSIBLE {
        if payload.len() < 40 {
            return Err(Error::corrupt(
                "WAVE_FORMAT_EXTENSIBLE fmt chunk is shorter than 40 bytes",
            ));
        }
        let extension_size = read_u16(payload, 16)?;
        if extension_size < 22 {
            return Err(Error::corrupt(
                "WAVE_FORMAT_EXTENSIBLE extension is shorter than 22 bytes",
            ));
        }
        format_tag = read_u16(payload, 24)?;
    }

    let format = match format_tag {
        WAVE_FORMAT_PCM => SampleFormat::Pcm,
        WAVE_FORMAT_IEEE_FLOAT => SampleFormat::Float,
        other => {
            return Err(Error::unsupported(format!(
                "WAV format tag {other} is not PCM or IEEE float"
            )));
        }
    };

    if channels == 0 || channels > MAX_AUDIO_CHANNELS {
        return Err(Error::limit_exceeded(format!(
            "WAV has {channels} channels, outside 1..={MAX_AUDIO_CHANNELS}"
        )));
    }
    if !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&sample_rate) {
        return Err(Error::limit_exceeded(format!(
            "WAV sample rate {sample_rate} Hz is outside {MIN_SAMPLE_RATE}..={MAX_SAMPLE_RATE} Hz"
        )));
    }
    let bytes_per_sample = match (format, bits_per_sample) {
        (SampleFormat::Pcm, 8 | 16 | 24 | 32) => usize::from(bits_per_sample / 8),
        (SampleFormat::Float, 32 | 64) => usize::from(bits_per_sample / 8),
        (SampleFormat::Pcm, other) => {
            return Err(Error::unsupported(format!(
                "PCM WAV with {other} bits per sample is not supported"
            )));
        }
        (SampleFormat::Float, other) => {
            return Err(Error::unsupported(format!(
                "IEEE-float WAV with {other} bits per sample is not supported"
            )));
        }
    };
    let expected_align = usize::from(channels)
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| Error::corrupt("WAV block alignment overflow"))?;
    if usize::from(block_align) != expected_align {
        return Err(Error::corrupt(format!(
            "WAV block alignment {block_align} does not match {channels} channels of {bytes_per_sample} bytes"
        )));
    }
    let expected_rate = u64::from(sample_rate)
        .checked_mul(u64::from(block_align))
        .ok_or_else(|| Error::corrupt("WAV byte rate overflow"))?;
    if u64::from(byte_rate) != expected_rate {
        return Err(Error::corrupt(format!(
            "WAV byte rate {byte_rate} does not match {expected_rate}"
        )));
    }

    Ok(FormatChunk {
        format,
        channels,
        sample_rate,
        block_align,
        bits_per_sample,
    })
}

fn decode_samples(
    bytes: &[u8],
    data_start: usize,
    frames: usize,
    format: FormatChunk,
) -> Result<Audio> {
    let channels = usize::from(format.channels);
    let bytes_per_sample = format.bytes_per_sample();
    let block_align = usize::from(format.block_align);
    let total = frames
        .checked_mul(block_align)
        .ok_or_else(|| Error::limit_exceeded("WAV sample count overflow"))?;
    let data_end = data_start
        .checked_add(total)
        .ok_or_else(|| Error::corrupt("WAV data range overflow"))?;
    if data_end > bytes.len() {
        return Err(Error::corrupt("WAV data chunk is truncated"));
    }
    let data = &bytes[data_start..data_end];
    let mut samples = Vec::with_capacity(frames);
    for frame in 0..frames {
        let mut sum = 0.0f32;
        for channel in 0..channels {
            let offset = frame * block_align + channel * bytes_per_sample;
            let value = decode_sample(data, offset, format.format, format.bits_per_sample)?;
            sum += value;
        }
        samples.push(sum / channels as f32);
    }
    Audio::new(samples, format.sample_rate)
}

fn decode_sample(data: &[u8], offset: usize, format: SampleFormat, bits: u16) -> Result<f32> {
    let value = match (format, bits) {
        (SampleFormat::Pcm, 8) => (f32::from(data[offset]) - 128.0) / 128.0,
        (SampleFormat::Pcm, 16) => {
            let raw = i16::from_le_bytes([data[offset], data[offset + 1]]);
            f32::from(raw) / 32_768.0
        }
        (SampleFormat::Pcm, 24) => {
            let raw = i32::from(data[offset])
                | (i32::from(data[offset + 1]) << 8)
                | (i32::from(data[offset + 2]) << 16);
            let signed = (raw << 8) >> 8;
            signed as f32 / 8_388_608.0
        }
        (SampleFormat::Pcm, 32) => {
            let raw = i32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            raw as f32 / 2_147_483_648.0
        }
        (SampleFormat::Float, 32) => f32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]),
        (SampleFormat::Float, 64) => {
            let raw = f64::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]);
            raw as f32
        }
        _ => {
            return Err(Error::internal(
                "WAV sample decoder received an unsupported layout",
            ));
        }
    };
    if !value.is_finite() {
        return Err(Error::corrupt("WAV contains a non-finite sample"));
    }
    Ok(value)
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16> {
    let slice = bytes
        .get(offset..offset + 2)
        .ok_or_else(|| Error::corrupt("WAV header is truncated"))?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| Error::corrupt("WAV header is truncated"))?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_wav(format_tag: u16, channels: u16, rate: u32, bits: u16, data: &[u8]) -> Vec<u8> {
        let block_align = channels * (bits / 8);
        let byte_rate = rate * u32::from(block_align);
        let mut fmt = Vec::new();
        fmt.extend_from_slice(&format_tag.to_le_bytes());
        fmt.extend_from_slice(&channels.to_le_bytes());
        fmt.extend_from_slice(&rate.to_le_bytes());
        fmt.extend_from_slice(&byte_rate.to_le_bytes());
        fmt.extend_from_slice(&block_align.to_le_bytes());
        fmt.extend_from_slice(&bits.to_le_bytes());
        let mut chunks = Vec::new();
        chunks.extend_from_slice(b"fmt ");
        chunks.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
        chunks.extend_from_slice(&fmt);
        chunks.extend_from_slice(b"data");
        chunks.extend_from_slice(&(data.len() as u32).to_le_bytes());
        chunks.extend_from_slice(data);
        if data.len() & 1 == 1 {
            chunks.push(0);
        }
        let riff_size = 4 + chunks.len() as u32;
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&riff_size.to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(&chunks);
        wav
    }

    #[test]
    fn decodes_pcm16_and_downmixes_stereo() {
        let mut data = Vec::new();
        for (left, right) in [(32767i16, -32768i16), (16384, 16384)] {
            data.extend_from_slice(&left.to_le_bytes());
            data.extend_from_slice(&right.to_le_bytes());
        }
        let wav = build_wav(WAVE_FORMAT_PCM, 2, 16_000, 16, &data);
        let audio = decode_bytes(&wav).expect("decode");
        assert_eq!(audio.sample_rate(), 16_000);
        assert_eq!(audio.len(), 2);
        assert!((audio.samples()[0] - (-(1.0 / 65_536.0))).abs() <= 1e-6);
        assert!((audio.samples()[1] - 0.5).abs() <= 1e-6);
    }

    #[test]
    fn decodes_pcm8_pcm24_and_float32() {
        let wav = build_wav(WAVE_FORMAT_PCM, 1, 16_000, 8, &[0, 128, 255]);
        let audio = decode_bytes(&wav).expect("pcm8");
        assert!((audio.samples()[1]).abs() <= 1e-6);
        assert!((audio.samples()[2] - 0.9921875).abs() <= 1e-6);

        let mut data = Vec::new();
        for raw in [0i32, 8_388_607, -8_388_608] {
            data.push((raw & 0xff) as u8);
            data.push(((raw >> 8) & 0xff) as u8);
            data.push(((raw >> 16) & 0xff) as u8);
        }
        let wav = build_wav(WAVE_FORMAT_PCM, 1, 16_000, 24, &data);
        let audio = decode_bytes(&wav).expect("pcm24");
        assert!((audio.samples()[0]).abs() <= 1e-6);
        assert!((audio.samples()[1] - 0.9999999).abs() <= 1e-6);
        assert!((audio.samples()[2] + 1.0).abs() <= 1e-6);

        let mut data = Vec::new();
        for value in [0.0f32, 0.5, -0.5] {
            data.extend_from_slice(&value.to_le_bytes());
        }
        let wav = build_wav(WAVE_FORMAT_IEEE_FLOAT, 1, 16_000, 32, &data);
        let audio = decode_bytes(&wav).expect("float32");
        assert_eq!(audio.samples(), &[0.0, 0.5, -0.5]);
    }

    #[test]
    fn rejects_malformed_headers() {
        assert!(matches!(decode_bytes(&[]), Err(Error::Corrupt(_))));
        assert!(matches!(
            decode_bytes(b"RIFX0000WAVE"),
            Err(Error::Corrupt(_))
        ));
        let wav = build_wav(WAVE_FORMAT_PCM, 1, 16_000, 16, &[0, 0]);
        assert!(matches!(
            decode_bytes(&wav[..wav.len() - 1]),
            Err(Error::Corrupt(_))
        ));
        let mut bad = build_wav(WAVE_FORMAT_PCM, 1, 16_000, 16, &[0, 0]);
        bad[32..34].copy_from_slice(&4u16.to_le_bytes());
        assert!(matches!(decode_bytes(&bad), Err(Error::Corrupt(_))));
        let mut trailing = build_wav(WAVE_FORMAT_PCM, 1, 16_000, 16, &[0, 0]);
        trailing.push(0);
        assert!(matches!(decode_bytes(&trailing), Err(Error::Corrupt(_))));
    }

    #[test]
    fn rejects_unsupported_codecs_and_bit_depths() {
        let wav = build_wav(2, 1, 16_000, 4, &[0]);
        assert!(matches!(decode_bytes(&wav), Err(Error::Unsupported(_))));
        let wav = build_wav(WAVE_FORMAT_PCM, 1, 16_000, 12, &[0, 0]);
        assert!(matches!(decode_bytes(&wav), Err(Error::Unsupported(_))));
    }

    #[test]
    fn rejects_non_finite_float_samples() {
        let data = f32::NAN.to_le_bytes();
        let wav = build_wav(WAVE_FORMAT_IEEE_FLOAT, 1, 16_000, 32, &data);
        assert!(matches!(decode_bytes(&wav), Err(Error::Corrupt(_))));
    }
}
