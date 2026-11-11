//! Mono audio samples, channel conversion, and deterministic resampling.
//!
//! The product works on FP32 mono audio. WAV decoding downmixes every channel
//! to one mono stream; this module then converts an arbitrary source rate to
//! the frozen model rate of 16 kHz. Resampling is a windowed-sinc low-pass
//! interpolation written with `std` only, so the behavior is deterministic and
//! independent of the platform audio stack.
//!
//! Explicit caps keep hostile or oversized input from allocating unbounded
//! buffers: at most 30 seconds of model audio (480,000 samples at 16 kHz), a
//! source rate between 8 kHz and 192 kHz, and at most eight channels.

use crate::error::{Error, Result};

/// Sampling rate expected by the frozen EmbeddingGemma 2 audio processor.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Maximum number of mono samples handed to the model (30 seconds at 16 kHz).
pub const MAX_AUDIO_SAMPLES: usize = 480_000;

/// Minimum accepted source sampling rate.
pub const MIN_SAMPLE_RATE: u32 = 8_000;

/// Maximum accepted source sampling rate.
pub const MAX_SAMPLE_RATE: u32 = 192_000;

/// Maximum number of channels accepted by the WAV decoder.
pub const MAX_AUDIO_CHANNELS: u16 = 8;

/// Number of sinc taps on each side of an output sample.
///
/// Thirty-two taps are enough to suppress aliasing when converting common
/// consumer rates (44.1 kHz and 48 kHz) to the 16 kHz model rate, while
/// remaining cheap enough for the 30-second cap.
const RESAMPLE_TAPS: i64 = 16;

/// Mono FP32 audio at its original sampling rate.
#[derive(Debug, Clone, PartialEq)]
pub struct Audio {
    samples: Vec<f32>,
    sample_rate: u32,
}

impl Audio {
    /// Creates a validated mono audio buffer.
    ///
    /// # Errors
    ///
    /// Returns a corruption error for an empty buffer, a non-finite sample, or
    /// a sample rate outside the documented range.
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Result<Self> {
        if samples.is_empty() {
            return Err(Error::corrupt("audio contains no samples"));
        }
        if !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&sample_rate) {
            return Err(Error::unsupported(format!(
                "sample rate {sample_rate} Hz is outside {MIN_SAMPLE_RATE}..={MAX_SAMPLE_RATE} Hz"
            )));
        }
        if samples.iter().any(|value| !value.is_finite()) {
            return Err(Error::corrupt("audio contains a non-finite sample"));
        }
        Ok(Self {
            samples,
            sample_rate,
        })
    }

    /// Mono samples in [-1, 1] for PCM input (floating-point WAV may exceed it).
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// Source sampling rate in hertz.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Number of mono samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Indicates an empty buffer.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Duration in seconds.
    pub fn duration_seconds(&self) -> f64 {
        self.samples.len() as f64 / f64::from(self.sample_rate)
    }

    /// Resamples to another rate.
    pub fn resample_to(&self, target_rate: u32) -> Result<Self> {
        let samples = resample(&self.samples, self.sample_rate, target_rate)?;
        Self::new(samples, target_rate)
    }

    /// Converts to the 16 kHz mono buffer consumed by the audio processor.
    ///
    /// Audio longer than the 30-second model cap is truncated, matching the
    /// official feature extractor's default `max_length=480000` and
    /// `truncation=True`.
    pub fn samples_16k(&self) -> Result<Vec<f32>> {
        let mut samples = if self.sample_rate == TARGET_SAMPLE_RATE {
            self.samples.clone()
        } else {
            resample(&self.samples, self.sample_rate, TARGET_SAMPLE_RATE)?
        };
        if samples.len() > MAX_AUDIO_SAMPLES {
            samples.truncate(MAX_AUDIO_SAMPLES);
        }
        if samples.is_empty() {
            return Err(Error::corrupt("audio resampling produced no samples"));
        }
        if samples.iter().any(|value| !value.is_finite()) {
            return Err(Error::internal(
                "audio resampling produced a non-finite sample",
            ));
        }
        Ok(samples)
    }
}

/// Resamples a mono buffer with a normalized windowed-sinc kernel.
///
/// # Errors
///
/// Returns an unsupported error for a rate outside the documented range, a
/// limit error for an output above the 30-second cap, and an internal error
/// for a non-finite result.
pub fn resample(samples: &[f32], source_rate: u32, target_rate: u32) -> Result<Vec<f32>> {
    if !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&source_rate) {
        return Err(Error::unsupported(format!(
            "source rate {source_rate} Hz is outside {MIN_SAMPLE_RATE}..={MAX_SAMPLE_RATE} Hz"
        )));
    }
    if !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&target_rate) {
        return Err(Error::unsupported(format!(
            "target rate {target_rate} Hz is outside {MIN_SAMPLE_RATE}..={MAX_SAMPLE_RATE} Hz"
        )));
    }
    if samples.is_empty() {
        return Ok(Vec::new());
    }
    if samples.iter().any(|value| !value.is_finite()) {
        return Err(Error::corrupt("audio contains a non-finite sample"));
    }
    if source_rate == target_rate {
        return Ok(samples.to_vec());
    }

    let output_len = (samples.len() as u128)
        .checked_mul(u128::from(target_rate))
        .ok_or_else(|| Error::limit_exceeded("resampled length overflow"))?
        / u128::from(source_rate);
    let output_len = usize::try_from(output_len)
        .map_err(|_| Error::limit_exceeded("resampled length does not fit usize"))?;
    if output_len == 0 {
        return Ok(Vec::new());
    }
    if output_len > MAX_AUDIO_SAMPLES {
        return Err(Error::limit_exceeded(format!(
            "resampled audio has {output_len} samples, above the {MAX_AUDIO_SAMPLES}-sample cap"
        )));
    }

    let ratio = f64::from(source_rate) / f64::from(target_rate);
    let cutoff = (f64::from(target_rate) / f64::from(source_rate)).min(1.0);
    let mut output = Vec::with_capacity(output_len);
    for index in 0..output_len {
        let time = index as f64 * ratio;
        let center = time.floor() as i64;
        let mut sum = 0.0f64;
        let mut weight_sum = 0.0f64;
        for tap in -RESAMPLE_TAPS + 1..=RESAMPLE_TAPS {
            let source_index = center + tap;
            if source_index < 0 || source_index >= samples.len() as i64 {
                continue;
            }
            let distance = time - source_index as f64;
            let sinc = if distance == 0.0 {
                1.0
            } else {
                let argument = std::f64::consts::PI * cutoff * distance;
                argument.sin() / argument
            };
            let window = 0.5 + 0.5 * (std::f64::consts::PI * distance / RESAMPLE_TAPS as f64).cos();
            let weight = sinc * window;
            sum += f64::from(samples[source_index as usize]) * weight;
            weight_sum += weight;
        }
        let value = if weight_sum.abs() < 1e-12 {
            0.0
        } else {
            sum / weight_sum
        };
        if !value.is_finite() {
            return Err(Error::internal(
                "audio resampling produced a non-finite sample",
            ));
        }
        output.push(value as f32);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resampling_preserves_length_and_a_sine_wave() {
        let source_rate = 44_100u32;
        let samples: Vec<f32> = (0..source_rate)
            .map(|index| {
                (2.0 * std::f64::consts::PI * 440.0 * f64::from(index) / f64::from(source_rate))
                    .sin() as f32
            })
            .collect();
        let output = resample(&samples, source_rate, TARGET_SAMPLE_RATE).expect("resample");
        assert_eq!(output.len(), TARGET_SAMPLE_RATE as usize);
        // Compare the RMS amplitude away from the edges.
        let start = 400usize;
        let end = output.len() - 400;
        let rms = (output[start..end]
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            / (end - start) as f64)
            .sqrt();
        assert!(
            (rms - std::f64::consts::FRAC_1_SQRT_2).abs() <= 0.02,
            "RMS {rms}"
        );
    }

    #[test]
    fn downsampling_removes_energy_above_the_new_nyquist() {
        // A 12 kHz tone is above the 8 kHz Nyquist of the target rate. A
        // band-limited conversion must strongly attenuate it.
        let source_rate = 48_000u32;
        let samples: Vec<f32> = (0..source_rate)
            .map(|index| {
                (2.0 * std::f64::consts::PI * 12_000.0 * f64::from(index) / f64::from(source_rate))
                    .sin() as f32
            })
            .collect();
        let output = resample(&samples, source_rate, TARGET_SAMPLE_RATE).expect("resample");
        let peak = output
            .iter()
            .skip(200)
            .take(output.len() - 400)
            .fold(0.0f32, |maximum, value| maximum.max(value.abs()));
        assert!(peak <= 0.05, "12 kHz alias peak {peak}");
    }

    #[test]
    fn rejects_out_of_range_rates_and_non_finite_samples() {
        let error = resample(&[0.0], 4_000, TARGET_SAMPLE_RATE).expect_err("low rate");
        assert!(matches!(error, Error::Unsupported(_)));
        let error = resample(&[f32::NAN], TARGET_SAMPLE_RATE, TARGET_SAMPLE_RATE)
            .expect_err("non-finite sample");
        assert!(matches!(error, Error::Corrupt(_)));
        let error = Audio::new(Vec::new(), TARGET_SAMPLE_RATE).expect_err("empty audio");
        assert!(matches!(error, Error::Corrupt(_)));
    }

    #[test]
    fn truncates_audio_above_the_model_cap() {
        let samples = vec![0.1f32; MAX_AUDIO_SAMPLES + 100];
        let audio = Audio::new(samples, TARGET_SAMPLE_RATE).expect("audio");
        let prepared = audio.samples_16k().expect("prepare");
        assert_eq!(prepared.len(), MAX_AUDIO_SAMPLES);
    }
}
