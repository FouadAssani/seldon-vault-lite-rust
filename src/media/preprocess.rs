//! Official EmbeddingGemma 2 image preprocessing.
//!
//! The official processor resizes each image to the largest aspect-ratio
//! preserving size that fits the visual token budget, converts it to RGB,
//! rescales bytes to [0, 1], and emits 16x16 patches with their (x, y) grid
//! positions. Padding patches are zero and their positions are (-1, -1).
//!
//! This module reproduces that pipeline without an image library. The resize
//! follows the antialiased bicubic path used by torchvision for uint8 tensors:
//! separable horizontal then vertical passes, Keys cubic convolution with
//! a = -0.5, per-output normalized weights, and 16-bit fixed-point weight
//! accumulation with round-half-up. It matches the reference bit for bit on
//! the committed fixtures; the tests enforce that with SHA-256 fingerprints.
//!
//! Every dimension and allocation is bounded before use. The product never
//! accepts an image whose preprocessing would exceed the documented caps.

use crate::error::{Error, Result};
use crate::media::Image;

/// Edge length of one square image patch in pixels.
pub const PATCH_SIZE: usize = 16;

/// Spatial pooling kernel applied after patchification.
pub const POOLING_KERNEL_SIZE: usize = 3;

/// Default visual token budget, matching the frozen processor configuration.
pub const DEFAULT_MAX_SOFT_TOKENS: usize = 280;

/// Visual token budgets accepted by the official processor.
pub const SUPPORTED_MAX_SOFT_TOKENS: [usize; 5] = [70, 140, 280, 560, 1120];

/// Maximum width or height accepted before preprocessing.
pub const MAX_IMAGE_SIDE: usize = 16_384;

/// Maximum input pixel count accepted before preprocessing.
pub const MAX_INPUT_PIXELS: usize = 16_777_216;

/// Maximum number of patches emitted for the largest token budget.
pub const MAX_PATCHES: usize = 1120 * POOLING_KERNEL_SIZE * POOLING_KERNEL_SIZE;

/// Number of interleaved RGB values in one patch.
pub const PATCH_VALUES: usize = PATCH_SIZE * PATCH_SIZE * 3;

/// Maximum number of FP32 values in the padded patch tensor.
pub const MAX_PIXEL_VALUES: usize = MAX_PATCHES * PATCH_VALUES;

/// Maximum number of taps in one resize pass.
///
/// The largest supported downscale is 16,384 to 48 pixels, which needs 1,365
/// taps. The cap leaves headroom while refusing an unbounded weight table.
pub const MAX_RESIZE_TAPS: usize = 8_192;

/// Maximum number of fixed-point weights held for one resize pass.
pub const MAX_RESIZE_WEIGHTS: usize = 1 << 20;

/// Validated preprocessing options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreprocessOptions {
    max_soft_tokens: usize,
}

impl PreprocessOptions {
    /// Builds options for one of the official visual token budgets.
    ///
    /// # Errors
    ///
    /// Returns a usage error when the budget is not one of 70, 140, 280, 560,
    /// or 1,120.
    pub fn new(max_soft_tokens: usize) -> Result<Self> {
        if !SUPPORTED_MAX_SOFT_TOKENS.contains(&max_soft_tokens) {
            return Err(Error::usage(format!(
                "visual token budget {max_soft_tokens} is not one of {SUPPORTED_MAX_SOFT_TOKENS:?}"
            )));
        }
        Ok(Self { max_soft_tokens })
    }

    /// Visual token budget.
    pub fn max_soft_tokens(self) -> usize {
        self.max_soft_tokens
    }
}

impl Default for PreprocessOptions {
    fn default() -> Self {
        Self {
            max_soft_tokens: DEFAULT_MAX_SOFT_TOKENS,
        }
    }
}

/// Preprocessed image patches ready for the vision encoder.
#[derive(Debug, Clone)]
pub struct ImagePatches {
    pixel_values: Vec<f32>,
    positions: Vec<[i32; 2]>,
    valid_patches: usize,
    soft_tokens: usize,
    max_patches: usize,
    patch_grid_width: usize,
    patch_grid_height: usize,
    resized_width: usize,
    resized_height: usize,
}

impl ImagePatches {
    /// Padded patch tensor, row-major, max_patches * PATCH_VALUES values.
    ///
    /// Each row is one patch in pixel-major, channel-last order: for every
    /// pixel of the 16x16 patch, its red, green, and blue values in [0, 1].
    pub fn pixel_values(&self) -> &[f32] {
        &self.pixel_values
    }

    /// Padded patch positions, max_patches entries.
    ///
    /// Valid patches carry their (x, y) grid coordinates; padding entries are
    /// (-1, -1).
    pub fn positions(&self) -> &[[i32; 2]] {
        &self.positions
    }

    /// Number of real (non-padding) patches.
    pub fn valid_patches(&self) -> usize {
        self.valid_patches
    }

    /// Number of visual soft tokens after 3x3 pooling.
    pub fn soft_tokens(&self) -> usize {
        self.soft_tokens
    }

    /// Number of rows in the padded patch tensor.
    pub fn max_patches(&self) -> usize {
        self.max_patches
    }

    /// Patch grid size as (columns, rows).
    pub fn patch_grid(&self) -> (usize, usize) {
        (self.patch_grid_width, self.patch_grid_height)
    }

    /// Resized image size as (width, height).
    pub fn resized_size(&self) -> (usize, usize) {
        (self.resized_width, self.resized_height)
    }
}

/// Preprocesses one decoded image with the official pipeline.
///
/// # Errors
///
/// Returns a limit error when a dimension, pixel count, patch count, resize
/// tap count, or buffer would exceed its cap; a corrupt error when the image
/// geometry is inconsistent; and a usage error for an unsupported token
/// budget.
pub fn preprocess(image: &Image, options: PreprocessOptions) -> Result<ImagePatches> {
    let width = image.width();
    let height = image.height();
    validate_input_geometry(width, height)?;

    let rgb = to_rgb(image)?;
    let max_patches = options
        .max_soft_tokens
        .checked_mul(POOLING_KERNEL_SIZE * POOLING_KERNEL_SIZE)
        .ok_or_else(|| Error::limit_exceeded("visual patch budget overflow"))?;
    if max_patches > MAX_PATCHES {
        return Err(Error::limit_exceeded(format!(
            "visual patch budget {max_patches} exceeds the {MAX_PATCHES}-patch cap"
        )));
    }

    let (target_width, target_height) =
        aspect_ratio_preserving_size(width, height, PATCH_SIZE, max_patches, POOLING_KERNEL_SIZE)?;
    let resized = resize_rgb(&rgb, width, height, target_width, target_height)?;

    let patch_grid_width = target_width / PATCH_SIZE;
    let patch_grid_height = target_height / PATCH_SIZE;
    let valid_patches = patch_grid_width
        .checked_mul(patch_grid_height)
        .ok_or_else(|| Error::limit_exceeded("patch grid overflow"))?;
    if valid_patches > max_patches {
        return Err(Error::limit_exceeded(format!(
            "image produces {valid_patches} patches, above the {max_patches}-patch budget"
        )));
    }
    let pooled = POOLING_KERNEL_SIZE * POOLING_KERNEL_SIZE;
    if valid_patches % pooled != 0 {
        return Err(Error::internal(
            "resized dimensions are not divisible by the pooling grid",
        ));
    }
    let soft_tokens = valid_patches / pooled;
    if soft_tokens == 0 {
        return Err(Error::limit_exceeded(
            "image produces zero visual soft tokens",
        ));
    }
    if soft_tokens > options.max_soft_tokens {
        return Err(Error::internal(format!(
            "image produces {soft_tokens} soft tokens, above the declared budget {}",
            options.max_soft_tokens
        )));
    }

    let pixel_count = max_patches
        .checked_mul(PATCH_VALUES)
        .ok_or_else(|| Error::limit_exceeded("padded patch tensor overflow"))?;
    if pixel_count > MAX_PIXEL_VALUES {
        return Err(Error::limit_exceeded(format!(
            "padded patch tensor has {pixel_count} values, above the {MAX_PIXEL_VALUES}-value cap"
        )));
    }

    let mut pixel_values = vec![0.0f32; pixel_count];
    let mut positions = vec![[-1i32, -1i32]; max_patches];
    for patch_y in 0..patch_grid_height {
        for patch_x in 0..patch_grid_width {
            let patch_index = patch_y * patch_grid_width + patch_x;
            let destination =
                &mut pixel_values[patch_index * PATCH_VALUES..(patch_index + 1) * PATCH_VALUES];
            let mut cursor = 0;
            for pixel_y in 0..PATCH_SIZE {
                let source_row = (patch_y * PATCH_SIZE + pixel_y) * target_width;
                for pixel_x in 0..PATCH_SIZE {
                    let source = (source_row + patch_x * PATCH_SIZE + pixel_x) * 3;
                    for channel in 0..3 {
                        destination[cursor] = f32::from(resized[source + channel]) * RESCALE_FACTOR;
                        cursor += 1;
                    }
                }
            }
            positions[patch_index] = [
                i32::try_from(patch_x)
                    .map_err(|_| Error::limit_exceeded("patch column does not fit i32"))?,
                i32::try_from(patch_y)
                    .map_err(|_| Error::limit_exceeded("patch row does not fit i32"))?,
            ];
        }
    }

    Ok(ImagePatches {
        pixel_values,
        positions,
        valid_patches,
        soft_tokens,
        max_patches,
        patch_grid_width,
        patch_grid_height,
        resized_width: target_width,
        resized_height: target_height,
    })
}

/// Rescale factor declared by the frozen processor configuration.
///
/// The configuration stores 0.00392156862745098, the double-precision value of
/// 1/255. Multiplying a float32 tensor by that scalar casts it to the nearest
/// float32, which is exactly 1.0 / 255.0 evaluated in f32.
const RESCALE_FACTOR: f32 = 1.0 / 255.0;

fn validate_input_geometry(width: usize, height: usize) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(Error::corrupt("image dimensions must be positive"));
    }
    if width > MAX_IMAGE_SIDE || height > MAX_IMAGE_SIDE {
        return Err(Error::limit_exceeded(format!(
            "image is {width}x{height}, above the {MAX_IMAGE_SIDE}-pixel side cap"
        )));
    }
    let pixels = width
        .checked_mul(height)
        .ok_or_else(|| Error::limit_exceeded("image pixel count overflow"))?;
    if pixels > MAX_INPUT_PIXELS {
        return Err(Error::limit_exceeded(format!(
            "image has {pixels} pixels, above the {MAX_INPUT_PIXELS}-pixel cap"
        )));
    }
    Ok(())
}

/// Computes the official aspect-ratio preserving target size.
///
/// The target is the largest size that produces at most max_patches patches
/// and whose sides are multiples of pooling_kernel_size * patch_size.
///
/// # Errors
///
/// Returns a limit error when the image cannot be resized to a positive size
/// inside the patch budget, or when the result exceeds the image caps.
pub fn aspect_ratio_preserving_size(
    width: usize,
    height: usize,
    patch_size: usize,
    max_patches: usize,
    pooling_kernel_size: usize,
) -> Result<(usize, usize)> {
    validate_input_geometry(width, height)?;
    if patch_size == 0 || pooling_kernel_size == 0 {
        return Err(Error::internal(
            "patch size and pooling kernel size must be positive",
        ));
    }
    if max_patches == 0 {
        return Err(Error::usage("visual patch budget must be positive"));
    }
    let side_mult = patch_size
        .checked_mul(pooling_kernel_size)
        .ok_or_else(|| Error::limit_exceeded("resize side multiple overflow"))?;

    let total_px = (width as f64) * (height as f64);
    let target_px = (max_patches as f64) * (patch_size as f64) * (patch_size as f64);
    let factor = (target_px / total_px).sqrt();
    let ideal_width = factor * (width as f64);
    let ideal_height = factor * (height as f64);
    let mut target_width = ((ideal_width / side_mult as f64).floor() as usize) * side_mult;
    let mut target_height = ((ideal_height / side_mult as f64).floor() as usize) * side_mult;

    if target_width == 0 && target_height == 0 {
        return Err(Error::limit_exceeded(format!(
            "image {width}x{height} cannot be resized to a positive size divisible by {side_mult}"
        )));
    }

    let max_side_length = (max_patches / (pooling_kernel_size * pooling_kernel_size)) * side_mult;
    if target_height == 0 {
        target_height = side_mult;
        let ratio = (width as f64 / height as f64).floor() as usize;
        target_width = ratio
            .checked_mul(side_mult)
            .ok_or_else(|| Error::limit_exceeded("resize width overflow"))?
            .min(max_side_length);
    } else if target_width == 0 {
        target_width = side_mult;
        let ratio = (height as f64 / width as f64).floor() as usize;
        target_height = ratio
            .checked_mul(side_mult)
            .ok_or_else(|| Error::limit_exceeded("resize height overflow"))?
            .min(max_side_length);
    }

    let target_pixels = target_width
        .checked_mul(target_height)
        .ok_or_else(|| Error::limit_exceeded("resized pixel count overflow"))?;
    if target_pixels == 0 {
        return Err(Error::limit_exceeded(
            "image aspect ratio produces a zero-sized target",
        ));
    }
    if (target_pixels as f64) > target_px {
        return Err(Error::limit_exceeded(format!(
            "resize to {target_width}x{target_height} exceeds the {max_patches}-patch budget"
        )));
    }
    if target_width > MAX_IMAGE_SIDE || target_height > MAX_IMAGE_SIDE {
        return Err(Error::limit_exceeded(format!(
            "resize to {target_width}x{target_height} exceeds the {MAX_IMAGE_SIDE}-pixel side cap"
        )));
    }
    if target_pixels > MAX_INPUT_PIXELS {
        return Err(Error::limit_exceeded(format!(
            "resize to {target_width}x{target_height} exceeds the {MAX_INPUT_PIXELS}-pixel cap"
        )));
    }
    Ok((target_width, target_height))
}

/// Converts a decoded image to interleaved RGB bytes.
///
/// Grayscale is replicated across the three channels. RGBA is composited over
/// a white background with the float path used by the official tensor backend,
/// then rounded half to even.
fn to_rgb(image: &Image) -> Result<Vec<u8>> {
    let width = image.width();
    let height = image.height();
    let pixels = width
        .checked_mul(height)
        .ok_or_else(|| Error::limit_exceeded("image pixel count overflow"))?;
    let source = image.pixels();
    let channels = usize::from(image.channels());
    let expected = pixels
        .checked_mul(channels)
        .ok_or_else(|| Error::limit_exceeded("image byte count overflow"))?;
    if source.len() != expected {
        return Err(Error::corrupt(format!(
            "image has {} bytes, expected {expected}",
            source.len()
        )));
    }
    let output_length = pixels
        .checked_mul(3)
        .ok_or_else(|| Error::limit_exceeded("RGB byte count overflow"))?;
    let mut output = Vec::with_capacity(output_length);
    match channels {
        1 => {
            for &value in source {
                output.extend_from_slice(&[value, value, value]);
            }
        }
        3 => output.extend_from_slice(source),
        4 => {
            for pixel in source.as_chunks::<4>().0 {
                let alpha = f32::from(pixel[3]) / 255.0;
                for &channel in &pixel[..3] {
                    let value = (f32::from(channel) / 255.0) * alpha + (1.0 - alpha);
                    output.push(round_half_even(f64::from(value * 255.0)));
                }
            }
        }
        other => {
            return Err(Error::unsupported(format!(
                "image preprocessing does not accept {other} channels"
            )));
        }
    }
    if output.len() != output_length {
        return Err(Error::internal(
            "RGB conversion produced an unexpected length",
        ));
    }
    Ok(output)
}

/// Rounds a finite value to the nearest integer, ties to even, then clamps to
/// the uint8 range.
fn round_half_even(value: f64) -> u8 {
    let floor = value.floor();
    let fraction = value - floor;
    let rounded = if fraction > 0.5 || (fraction == 0.5 && (floor as i64) % 2 != 0) {
        floor + 1.0
    } else {
        floor
    };
    rounded.clamp(0.0, 255.0) as u8
}

/// Resizes interleaved RGB bytes with antialiased bicubic interpolation.
fn resize_rgb(
    source: &[u8],
    width: usize,
    height: usize,
    target_width: usize,
    target_height: usize,
) -> Result<Vec<u8>> {
    if width == target_width && height == target_height {
        return Ok(source.to_vec());
    }
    let expected = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| Error::limit_exceeded("resize source length overflow"))?;
    if source.len() != expected {
        return Err(Error::internal(format!(
            "resize source has {} bytes, expected {expected}",
            source.len()
        )));
    }

    let horizontal = if target_width != width {
        Some(resize_horizontal(source, width, height, target_width)?)
    } else {
        None
    };
    let intermediate = horizontal.as_deref().unwrap_or(source);
    if target_height == height {
        return Ok(intermediate.to_vec());
    }
    resize_vertical(intermediate, target_width, height, target_height)
}

fn resize_horizontal(
    source: &[u8],
    width: usize,
    height: usize,
    target_width: usize,
) -> Result<Vec<u8>> {
    let weights = ResizeWeights::new(width, target_width)?;
    let length = target_width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| Error::limit_exceeded("horizontal resize output overflow"))?;
    let mut output = vec![0u8; length];
    for row in 0..height {
        let source_row = &source[row * width * 3..(row + 1) * width * 3];
        let destination_row = &mut output[row * target_width * 3..(row + 1) * target_width * 3];
        for x in 0..target_width {
            let (start, count) = weights.range(x);
            for channel in 0..3 {
                let mut total = weights.rounding();
                for tap in 0..count {
                    total += i64::from(source_row[(start + tap) * 3 + channel])
                        * i64::from(weights.weight(x, tap));
                }
                destination_row[x * 3 + channel] = clamp_shift(total, weights.precision());
            }
        }
    }
    Ok(output)
}

fn resize_vertical(
    source: &[u8],
    width: usize,
    input_height: usize,
    target_height: usize,
) -> Result<Vec<u8>> {
    let weights = ResizeWeights::new(input_height, target_height)?;
    let length = width
        .checked_mul(target_height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| Error::limit_exceeded("vertical resize output overflow"))?;
    let mut output = vec![0u8; length];
    let row_bytes = width
        .checked_mul(3)
        .ok_or_else(|| Error::limit_exceeded("resize row length overflow"))?;
    for y in 0..target_height {
        let (start, count) = weights.range(y);
        for x in 0..width {
            for channel in 0..3 {
                let mut total = weights.rounding();
                for tap in 0..count {
                    let row = start + tap;
                    total += i64::from(source[row * row_bytes + x * 3 + channel])
                        * i64::from(weights.weight(y, tap));
                }
                output[y * width * 3 + x * 3 + channel] = clamp_shift(total, weights.precision());
            }
        }
    }
    Ok(output)
}

fn clamp_shift(total: i64, precision: u32) -> u8 {
    (total >> precision).clamp(0, 255) as u8
}

/// Fixed-point resize weights for one axis.
struct ResizeWeights {
    starts: Vec<usize>,
    counts: Vec<usize>,
    weights: Vec<i16>,
    stride: usize,
    precision: u32,
}

impl ResizeWeights {
    fn new(input: usize, output: usize) -> Result<Self> {
        if input == 0 || output == 0 {
            return Err(Error::internal("resize axis must be positive"));
        }
        if input > MAX_IMAGE_SIDE || output > MAX_IMAGE_SIDE {
            return Err(Error::limit_exceeded(format!(
                "resize axis {input}->{output} exceeds the {MAX_IMAGE_SIDE}-pixel cap"
            )));
        }
        let scale = input as f64 / output as f64;
        let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
        let max_taps = ((support.ceil() as usize).saturating_mul(2)).saturating_add(1);
        if max_taps > MAX_RESIZE_TAPS {
            return Err(Error::limit_exceeded(format!(
                "resize needs {max_taps} taps, above the {MAX_RESIZE_TAPS}-tap cap"
            )));
        }
        let total_weights = output
            .checked_mul(max_taps)
            .ok_or_else(|| Error::limit_exceeded("resize weight count overflow"))?;
        if total_weights > MAX_RESIZE_WEIGHTS {
            return Err(Error::limit_exceeded(format!(
                "resize needs {total_weights} weights, above the {MAX_RESIZE_WEIGHTS}-weight cap"
            )));
        }

        let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
        let mut starts = Vec::with_capacity(output);
        let mut counts = Vec::with_capacity(output);
        let mut normalized = vec![0.0f64; total_weights];
        let mut max_weight = 0.0f64;

        for index in 0..output {
            let center = scale * (index as f64 + 0.5);
            let start = ((center - support + 0.5) as i64).max(0) as usize;
            let end = ((center + support + 0.5) as i64).clamp(0, input as i64) as usize;
            let count = end.saturating_sub(start).min(max_taps);
            let slice = &mut normalized[index * max_taps..index * max_taps + count];
            let mut total = 0.0f64;
            for (tap, value) in slice.iter_mut().enumerate() {
                let x = (tap as f64 + start as f64 - center + 0.5) * invscale;
                *value = cubic_filter(x);
                total += *value;
            }
            if total != 0.0 {
                for value in slice.iter_mut() {
                    *value /= total;
                    max_weight = max_weight.max(*value);
                }
            }
            starts.push(start);
            counts.push(count);
        }

        if max_weight <= 0.0 {
            return Err(Error::internal("resize filter produced no positive weight"));
        }
        let mut precision = 0u32;
        while precision < 22 {
            let next = (0.5 + max_weight * f64::from(1u32 << (precision + 1))) as i64;
            if next >= (1 << 15) {
                break;
            }
            precision += 1;
        }
        if precision == 0 {
            return Err(Error::internal("resize weight precision underflow"));
        }

        let mut weights = vec![0i16; total_weights];
        for index in 0..output {
            let count = counts[index];
            for tap in 0..count {
                let value = normalized[index * max_taps + tap];
                weights[index * max_taps + tap] = quantize(value, precision)?;
            }
        }

        Ok(Self {
            starts,
            counts,
            weights,
            stride: max_taps,
            precision,
        })
    }

    fn range(&self, index: usize) -> (usize, usize) {
        (self.starts[index], self.counts[index])
    }

    fn weight(&self, index: usize, tap: usize) -> i16 {
        self.weights[index * self.stride + tap]
    }

    fn precision(&self) -> u32 {
        self.precision
    }

    fn rounding(&self) -> i64 {
        1i64 << (self.precision - 1)
    }
}

fn quantize(value: f64, precision: u32) -> Result<i16> {
    let scaled = value * f64::from(1u32 << precision);
    let rounded = if scaled < 0.0 {
        (scaled - 0.5) as i64
    } else {
        (scaled + 0.5) as i64
    };
    i16::try_from(rounded)
        .map_err(|_| Error::limit_exceeded("resize weight does not fit a 16-bit integer"))
}

/// Keys cubic convolution with a = -0.5, as used by the antialiased path.
fn cubic_filter(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A
    } else {
        0.0
    }
}
