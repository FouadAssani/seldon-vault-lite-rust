//! Official EmbeddingGemma 2 audio preprocessing.
//!
//! The frozen processor turns 16 kHz mono samples into a 128-bin log-mel
//! spectrogram. The implementation reproduces the official
//! `Gemma4AudioFeatureExtractor`:
//!
//! - 20 ms frames (320 samples) and a 10 ms hop (160 samples);
//! - a periodic Hann window;
//! - semicausal padding of 160 zeros, so the first frame is centered at t=0;
//! - a 512-point real FFT, with 257 magnitude bins;
//! - a 128-filter HTK mel bank from 0 Hz to 8000 Hz, without area
//!   normalization;
//! - `ln(mel + 0.001)`;
//! - right-padding to a multiple of 128 samples and a frame mask that marks
//!   only frames whose complete analysis window contains real audio.
//!
//! The FFT and the mel projection are computed in f64 and rounded to f32 once,
//! matching NumPy's default promotion. No third-party FFT or audio library is
//! used.

use crate::error::{Error, Result};
use crate::media::audio::TARGET_SAMPLE_RATE;

/// Frame length in samples (20 ms at 16 kHz).
pub const FRAME_LENGTH: usize = 320;

/// Hop length in samples (10 ms at 16 kHz).
pub const HOP_LENGTH: usize = 160;

/// FFT length.
pub const FFT_LENGTH: usize = 512;

/// Number of mel filters.
pub const FEATURE_SIZE: usize = 128;

/// Number of magnitude bins of the real FFT.
pub const FREQUENCY_BINS: usize = FFT_LENGTH / 2 + 1;

/// Floor added before the logarithm.
pub const MEL_FLOOR: f64 = 0.001;

/// Right-padding granularity applied by the official feature extractor.
pub const PAD_TO_MULTIPLE: usize = 128;

/// Maximum number of mel frames for the 30-second cap.
pub const MAX_MEL_FRAMES: usize = 3_000;

/// Validated log-mel spectrogram.
#[derive(Debug, Clone)]
pub struct MelSpectrogram {
    features: Vec<f32>,
    mask: Vec<bool>,
    frames: usize,
    valid_frames: usize,
    padded_samples: usize,
}

impl MelSpectrogram {
    /// Row-major [frames, 128] log-mel features.
    pub fn features(&self) -> &[f32] {
        &self.features
    }

    /// Frame mask: true for frames whose complete window contains real audio.
    pub fn mask(&self) -> &[bool] {
        &self.mask
    }

    /// Total number of mel frames, including right-padding frames.
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// Number of frames marked valid by the mask.
    pub fn valid_frames(&self) -> usize {
        self.valid_frames
    }

    /// Number of 16 kHz samples after right-padding to a multiple of 128.
    pub fn padded_samples(&self) -> usize {
        self.padded_samples
    }
}

/// Computes the official log-mel spectrogram from 16 kHz mono samples.
///
/// Audio longer than the 30-second cap is truncated. At least one valid frame
/// is required.
///
/// # Errors
///
/// Returns a corruption error for empty, non-finite, or too-short audio, and a
/// limit error for a frame count above the documented cap.
pub fn log_mel(samples: &[f32]) -> Result<MelSpectrogram> {
    if samples.is_empty() {
        return Err(Error::corrupt("audio contains no samples"));
    }
    if samples.iter().any(|value| !value.is_finite()) {
        return Err(Error::corrupt("audio contains a non-finite sample"));
    }
    let samples = if samples.len() > crate::media::audio::MAX_AUDIO_SAMPLES {
        &samples[..crate::media::audio::MAX_AUDIO_SAMPLES]
    } else {
        samples
    };

    let padded_samples = samples
        .len()
        .checked_add(PAD_TO_MULTIPLE - 1)
        .ok_or_else(|| Error::limit_exceeded("audio padding length overflow"))?
        / PAD_TO_MULTIPLE
        * PAD_TO_MULTIPLE;
    let padded_len = padded_samples
        .checked_add(FRAME_LENGTH / 2)
        .ok_or_else(|| Error::limit_exceeded("audio frame buffer overflow"))?;
    let frame_size = FRAME_LENGTH + 1;
    if padded_len < frame_size {
        return Err(Error::corrupt(format!(
            "audio has {} samples, too short for one {FRAME_LENGTH}-sample analysis frame",
            samples.len()
        )));
    }
    let frames = (padded_len - frame_size) / HOP_LENGTH + 1;
    if frames > MAX_MEL_FRAMES {
        return Err(Error::limit_exceeded(format!(
            "audio produces {frames} mel frames, above the {MAX_MEL_FRAMES}-frame cap"
        )));
    }

    let window = hann_window();
    let filters = mel_filter_bank()?;
    let fft = FftPlan::new(FFT_LENGTH)?;

    let mut features = Vec::with_capacity(
        frames
            .checked_mul(FEATURE_SIZE)
            .ok_or_else(|| Error::limit_exceeded("mel feature size overflow"))?,
    );
    let mut mask = Vec::with_capacity(frames);
    let mut valid_frames = 0usize;
    let mut frame_samples = vec![0.0f32; FFT_LENGTH];
    let mut spectrum = vec![Complex::default(); FFT_LENGTH];
    let mut magnitudes = vec![0.0f64; FREQUENCY_BINS];

    for frame in 0..frames {
        let start = frame * HOP_LENGTH;
        // The official feature extractor multiplies every frame by its mask
        // after extraction, so a frame whose analysis window is not complete
        // is exactly zero. Skipping it is equivalent and avoids useless FFT
        // work on trailing padding.
        let valid = start + FRAME_LENGTH / 2 < samples.len();
        mask.push(valid);
        if !valid {
            features.extend(std::iter::repeat_n(0.0f32, FEATURE_SIZE));
            continue;
        }
        valid_frames += 1;
        // The analysis window is frame_size samples; the last sample is
        // dropped by the official no-preemphasis branch, so FRAME_LENGTH
        // samples are windowed.
        for index in 0..FRAME_LENGTH {
            let sample = if start + index < FRAME_LENGTH / 2 {
                0.0
            } else {
                let source = start + index - FRAME_LENGTH / 2;
                if source < padded_samples {
                    samples.get(source).copied().unwrap_or(0.0)
                } else {
                    0.0
                }
            };
            frame_samples[index] = sample * window[index];
        }
        for value in &mut frame_samples[FRAME_LENGTH..] {
            *value = 0.0;
        }
        for (destination, value) in spectrum.iter_mut().zip(&frame_samples) {
            *destination = Complex {
                re: f64::from(*value),
                im: 0.0,
            };
        }
        fft.forward(&mut spectrum);
        for (index, magnitude) in magnitudes.iter_mut().enumerate() {
            let value = &spectrum[index];
            *magnitude = (value.re * value.re + value.im * value.im).sqrt();
        }

        for filter in 0..FEATURE_SIZE {
            let mut energy = 0.0f64;
            for (bin, magnitude) in magnitudes.iter().enumerate() {
                energy += *magnitude * filters[bin * FEATURE_SIZE + filter];
            }
            features.push((energy + MEL_FLOOR).ln() as f32);
        }
    }

    if valid_frames == 0 {
        return Err(Error::corrupt(
            "audio has no complete analysis window and cannot be embedded",
        ));
    }
    if features.iter().any(|value| !value.is_finite()) {
        return Err(Error::internal(
            "audio preprocessing produced a non-finite mel value",
        ));
    }
    Ok(MelSpectrogram {
        features,
        mask,
        frames,
        valid_frames,
        padded_samples,
    })
}

fn hann_window() -> Vec<f32> {
    let mut window = Vec::with_capacity(FRAME_LENGTH);
    for index in 0..FRAME_LENGTH {
        let angle = 2.0 * std::f64::consts::PI * index as f64 / FRAME_LENGTH as f64;
        window.push((0.5 - 0.5 * angle.cos()) as f32);
    }
    window
}

fn mel_filter_bank() -> Result<Vec<f64>> {
    let sample_rate = f64::from(TARGET_SAMPLE_RATE);
    let nyquist = sample_rate / 2.0;
    let mel_min = hertz_to_mel(0.0);
    let mel_max = hertz_to_mel(8000.0);
    let mut filter_freqs = Vec::with_capacity(FEATURE_SIZE + 2);
    for index in 0..=FEATURE_SIZE + 1 {
        let fraction = index as f64 / (FEATURE_SIZE + 1) as f64;
        let mel = mel_min + (mel_max - mel_min) * fraction;
        filter_freqs.push(mel_to_hertz(mel));
    }
    let mut filters = vec![0.0f64; FREQUENCY_BINS * FEATURE_SIZE];
    for bin in 0..FREQUENCY_BINS {
        let frequency = nyquist * bin as f64 / (FREQUENCY_BINS - 1) as f64;
        for filter in 0..FEATURE_SIZE {
            let lower = filter_freqs[filter];
            let center = filter_freqs[filter + 1];
            let upper = filter_freqs[filter + 2];
            let down = (frequency - lower) / (center - lower);
            let up = (upper - frequency) / (upper - center);
            let value = down.min(up).max(0.0);
            if !value.is_finite() {
                return Err(Error::internal(
                    "mel filter bank produced a non-finite value",
                ));
            }
            filters[bin * FEATURE_SIZE + filter] = value;
        }
    }
    Ok(filters)
}

fn hertz_to_mel(frequency: f64) -> f64 {
    2595.0 * (1.0 + frequency / 700.0).log10()
}

fn mel_to_hertz(mel: f64) -> f64 {
    700.0 * (10.0f64.powf(mel / 2595.0) - 1.0)
}

#[derive(Debug, Clone, Copy, Default)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn multiply(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    fn subtract(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }
}

/// Iterative radix-2 FFT plan for a power-of-two length.
struct FftPlan {
    length: usize,
    reversed: Vec<usize>,
    twiddles: Vec<Complex>,
}

impl FftPlan {
    fn new(length: usize) -> Result<Self> {
        if length == 0 || !length.is_power_of_two() {
            return Err(Error::internal(
                "FFT length must be a non-zero power of two",
            ));
        }
        let bits = length.trailing_zeros();
        let mut reversed = Vec::with_capacity(length);
        for index in 0..length {
            reversed.push(index.reverse_bits() >> (usize::BITS - bits));
        }
        let mut twiddles = Vec::with_capacity(length / 2);
        for index in 0..length / 2 {
            let angle = -2.0 * std::f64::consts::PI * index as f64 / length as f64;
            twiddles.push(Complex {
                re: angle.cos(),
                im: angle.sin(),
            });
        }
        Ok(Self {
            length,
            reversed,
            twiddles,
        })
    }

    fn forward(&self, values: &mut [Complex]) {
        debug_assert_eq!(values.len(), self.length);
        for (index, target) in self.reversed.iter().enumerate() {
            if index < *target {
                values.swap(index, *target);
            }
        }
        let mut span = 2usize;
        while span <= self.length {
            let half = span / 2;
            let step = self.length / span;
            for start in (0..self.length).step_by(span) {
                for offset in 0..half {
                    let twiddle = self.twiddles[offset * step];
                    let even = values[start + offset];
                    let odd = values[start + offset + half].multiply(twiddle);
                    values[start + offset] = even.add(odd);
                    values[start + offset + half] = even.subtract(odd);
                }
            }
            span *= 2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_the_periodic_hann_window() {
        let window = hann_window();
        assert_eq!(window.len(), FRAME_LENGTH);
        assert!(window[0].abs() <= 1e-7);
        assert!((window[FRAME_LENGTH / 2] - 1.0).abs() <= 1e-6);
        assert!((window[FRAME_LENGTH - 1] - 0.000096).abs() <= 1e-5);
    }

    #[test]
    fn fft_matches_a_direct_dft() {
        let plan = FftPlan::new(16).expect("plan");
        let input: Vec<Complex> = (0..16)
            .map(|index| Complex {
                re: (index as f64 * 0.7).sin(),
                im: 0.0,
            })
            .collect();
        let mut values = input.clone();
        plan.forward(&mut values);
        for (bin, value) in values.iter().enumerate() {
            let mut expected = Complex::default();
            for (index, sample) in input.iter().enumerate() {
                let angle = -2.0 * std::f64::consts::PI * bin as f64 * index as f64 / 16.0;
                expected = expected.add(Complex {
                    re: sample.re * angle.cos(),
                    im: sample.re * angle.sin(),
                });
            }
            assert!((value.re - expected.re).abs() <= 1e-9);
            assert!((value.im - expected.im).abs() <= 1e-9);
        }
    }

    #[test]
    fn mel_filters_are_triangular_and_cover_the_band() {
        let filters = mel_filter_bank().expect("filters");
        assert_eq!(filters.len(), FREQUENCY_BINS * FEATURE_SIZE);
        assert!(filters.iter().all(|value| *value >= 0.0 && *value <= 1.0));
        // The first filter has no energy at DC, and the bank has energy away
        // from DC.
        assert_eq!(filters[0], 0.0);
        assert!(filters.iter().any(|value| *value > 0.0));
    }

    #[test]
    fn rejects_empty_and_non_finite_audio() {
        assert!(matches!(log_mel(&[]), Err(Error::Corrupt(_))));
        assert!(matches!(log_mel(&[f32::NAN]), Err(Error::Corrupt(_))));
        assert!(matches!(log_mel(&[0.0; 10]), Err(Error::Corrupt(_))));
    }

    #[test]
    fn frames_and_mask_follow_the_semicausal_layout() {
        let samples = vec![0.0f32; 1_600];
        let mel = log_mel(&samples).expect("mel");
        assert_eq!(mel.frames(), 10);
        assert_eq!(mel.valid_frames(), 9);
        assert_eq!(mel.padded_samples(), 1_664);
        assert_eq!(mel.features().len(), 10 * FEATURE_SIZE);
        assert_eq!(
            mel.mask(),
            &[true; 9].iter().copied().chain([false]).collect::<Vec<_>>()
        );
    }
}
