//! NEON kernels for the macOS Apple Silicon target.
//!
//! The scalar kernels in [`crate::tensor::kernels`] remain the correctness
//! oracle. These functions are an execution path only: every safe wrapper
//! checks [`is_available`] at runtime and falls back to the scalar
//! implementation when NEON is not usable, so a caller cannot execute an
//! illegal instruction on a processor without the feature.
//!
//! Numerical note: the vector reductions accumulate in four independent lanes
//! and the fused multiply-add rounds once instead of twice. Sums therefore
//! differ from the sequential scalar path by a few units in the last place.
//! The differential tests in `kernels` bound that difference, and the frozen
//! model parity tests bound the end-to-end embedding drift.

use crate::error::{Error, Result};
use crate::tensor::kernels::{
    add_in_place_scalar, add_scalar, dot_scalar, l2_normalize_scalar, linear_into_scalar,
    matmul_into_scalar, mean_pool_scalar, mul_scalar, rms_norm_rows_scalar,
    rms_norm_rows_unscaled_scalar, rms_norm_scalar, scale_scalar, sub_scalar,
    validate_l2_normalize, validate_linear_shapes, validate_matmul_shapes, validate_mean_pool,
    validate_rms_norm, validate_rms_norm_rows, validate_rms_norm_rows_unscaled,
};
use std::sync::OnceLock;

/// Returns true when the NEON feature is available at runtime.
///
/// The result is cached because the check sits on every dispatch path. AArch64
/// requires NEON, but the runtime check keeps the kernel honest if a future
/// target makes the feature optional.
pub(crate) fn is_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| std::arch::is_aarch64_feature_detected!("neon"))
}

/// Element-wise addition.
pub(crate) fn add(left: &[f32], right: &[f32]) -> Vec<f32> {
    if !is_available() {
        return add_scalar(left, right);
    }
    debug_assert_eq!(left.len(), right.len());
    // SAFETY: the caller validated equal lengths, and is_available() returned
    // true, so the target feature is present for this thread.
    unsafe { imp::add(left, right) }
}

/// Element-wise subtraction.
pub(crate) fn sub(left: &[f32], right: &[f32]) -> Vec<f32> {
    if !is_available() {
        return sub_scalar(left, right);
    }
    debug_assert_eq!(left.len(), right.len());
    // SAFETY: the caller validated equal lengths, and is_available() returned
    // true, so the target feature is present for this thread.
    unsafe { imp::sub(left, right) }
}

/// Element-wise multiplication.
pub(crate) fn mul(left: &[f32], right: &[f32]) -> Vec<f32> {
    if !is_available() {
        return mul_scalar(left, right);
    }
    debug_assert_eq!(left.len(), right.len());
    // SAFETY: the caller validated equal lengths, and is_available() returned
    // true, so the target feature is present for this thread.
    unsafe { imp::mul(left, right) }
}

/// Multiplies every element by a finite scalar.
pub(crate) fn scale(values: &[f32], factor: f32) -> Vec<f32> {
    if !is_available() {
        return scale_scalar(values, factor);
    }
    // SAFETY: is_available() returned true, so the target feature is present.
    unsafe { imp::scale(values, factor) }
}

/// Dot product of two vectors of equal length.
pub(crate) fn dot(left: &[f32], right: &[f32]) -> f32 {
    if !is_available() {
        return dot_scalar(left, right);
    }
    debug_assert_eq!(left.len(), right.len());
    // SAFETY: the caller validated equal lengths, and is_available() returned
    // true, so the target feature is present for this thread.
    unsafe { imp::dot(left, right) }
}

/// Adds one slice into another in place.
pub(crate) fn add_in_place(target: &mut [f32], other: &[f32]) -> Result<()> {
    if !is_available() {
        add_in_place_scalar(target, other);
        return Ok(());
    }
    debug_assert_eq!(target.len(), other.len());
    // SAFETY: the caller validated equal lengths, and is_available() returned
    // true, so the target feature is present for this thread. The two slices
    // are distinct by the Rust borrow rules.
    unsafe { imp::add_in_place(target, other) };
    Ok(())
}

/// Matrix multiplication C = A * B with row-major operands.
pub(crate) fn matmul_into(
    left: &[f32],
    right: &[f32],
    m: usize,
    k: usize,
    n: usize,
    output: &mut [f32],
) -> Result<()> {
    validate_matmul_shapes(left, right, m, k, n, output)?;
    if !is_available() {
        return matmul_into_scalar(left, right, m, k, n, output);
    }
    debug_assert_eq!(left.len(), m * k);
    debug_assert_eq!(right.len(), k * n);
    debug_assert_eq!(output.len(), m * n);
    // SAFETY: the caller validated all three lengths against m, k, and n, and
    // is_available() returned true, so the target feature is present.
    unsafe { imp::matmul_into(left, right, m, k, n, output) };
    if output.iter().any(|value| !value.is_finite()) {
        return Err(Error::internal("matmul produced a non-finite value"));
    }
    Ok(())
}

/// Linear projection against a weight matrix stored as [out_features, in_features].
pub(crate) fn linear_into(
    input: &[f32],
    weight: &[f32],
    rows: usize,
    in_features: usize,
    out_features: usize,
    output: &mut [f32],
) -> Result<()> {
    validate_linear_shapes(input, weight, rows, in_features, out_features, output)?;
    if !is_available() {
        return linear_into_scalar(input, weight, rows, in_features, out_features, output);
    }
    debug_assert_eq!(input.len(), rows * in_features);
    debug_assert_eq!(weight.len(), out_features * in_features);
    debug_assert_eq!(output.len(), rows * out_features);
    // SAFETY: the caller validated all three lengths against the shape, and
    // is_available() returned true, so the target feature is present.
    unsafe { imp::linear_into(input, weight, rows, in_features, out_features, output) };
    if output.iter().any(|value| !value.is_finite()) {
        return Err(Error::internal(
            "linear projection produced a non-finite value",
        ));
    }
    Ok(())
}

/// Root mean square normalization of one vector.
pub(crate) fn rms_norm(input: &[f32], weight: &[f32], epsilon: f32) -> Result<Vec<f32>> {
    validate_rms_norm(input, weight, epsilon)?;
    if !is_available() {
        return rms_norm_scalar(input, weight, epsilon);
    }
    debug_assert_eq!(input.len(), weight.len());
    debug_assert!(!input.is_empty());
    // SAFETY: the caller validated equal non-zero lengths, and is_available()
    // returned true, so the target feature is present.
    let sum = unsafe { imp::dot(input, input) };
    if !sum.is_finite() {
        return Err(Error::internal("RMSNorm sum of squares overflowed"));
    }
    let inverse = 1.0 / (sum / input.len() as f32 + epsilon).sqrt();
    let mut output = vec![0.0f32; input.len()];
    // SAFETY: input, weight, and output have the same validated length, and
    // is_available() returned true.
    unsafe { imp::scale_weighted(input, weight, inverse, &mut output) };
    if output.iter().any(|value| !value.is_finite()) {
        return Err(Error::internal("RMSNorm produced a non-finite value"));
    }
    Ok(output)
}

/// RMSNorm applied independently to each contiguous row of a matrix.
pub(crate) fn rms_norm_rows(
    input: &[f32],
    weight: &[f32],
    rows: usize,
    width: usize,
    epsilon: f32,
) -> Result<Vec<f32>> {
    validate_rms_norm_rows(input, weight, rows, width, epsilon)?;
    if !is_available() {
        return rms_norm_rows_scalar(input, weight, rows, width, epsilon);
    }
    debug_assert_eq!(weight.len(), width);
    debug_assert_eq!(input.len(), rows * width);
    debug_assert!(width > 0);
    let mut output = vec![0.0f32; input.len()];
    for row in 0..rows {
        let input_row = &input[row * width..(row + 1) * width];
        let output_row = &mut output[row * width..(row + 1) * width];
        // SAFETY: input_row has width elements, and is_available() returned true.
        let sum = unsafe { imp::dot(input_row, input_row) };
        if !sum.is_finite() {
            return Err(Error::internal("RMSNorm sum of squares overflowed"));
        }
        let inverse = 1.0 / (sum / width as f32 + epsilon).sqrt();
        // SAFETY: input_row, weight, and output_row have the validated width,
        // and is_available() returned true.
        unsafe { imp::scale_weighted(input_row, weight, inverse, output_row) };
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(Error::internal("RMSNorm produced a non-finite value"));
    }
    Ok(output)
}

/// RMSNorm without a learned scale, applied to each contiguous row.
pub(crate) fn rms_norm_rows_unscaled(
    input: &[f32],
    rows: usize,
    width: usize,
    epsilon: f32,
) -> Result<Vec<f32>> {
    validate_rms_norm_rows_unscaled(input, rows, width, epsilon)?;
    if !is_available() {
        return rms_norm_rows_unscaled_scalar(input, rows, width, epsilon);
    }
    debug_assert_eq!(input.len(), rows * width);
    debug_assert!(width > 0);
    let mut output = vec![0.0f32; input.len()];
    for row in 0..rows {
        let input_row = &input[row * width..(row + 1) * width];
        let output_row = &mut output[row * width..(row + 1) * width];
        // SAFETY: input_row has width elements, and is_available() returned true.
        let sum = unsafe { imp::dot(input_row, input_row) };
        if !sum.is_finite() {
            return Err(Error::internal("RMSNorm sum of squares overflowed"));
        }
        let inverse = 1.0 / (sum / width as f32 + epsilon).sqrt();
        // SAFETY: input_row and output_row have the validated width, and
        // is_available() returned true.
        unsafe { imp::scale_unweighted(input_row, inverse, output_row) };
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(Error::internal("RMSNorm produced a non-finite value"));
    }
    Ok(output)
}

/// L2 normalization of one vector.
pub(crate) fn l2_normalize(input: &[f32]) -> Result<Vec<f32>> {
    validate_l2_normalize(input)?;
    if !is_available() {
        return l2_normalize_scalar(input);
    }
    debug_assert!(!input.is_empty());
    // SAFETY: input is non-empty, and is_available() returned true.
    let sum = unsafe { imp::dot(input, input) };
    if !sum.is_finite() || sum <= 0.0 {
        return Err(Error::internal(
            "cannot normalize a zero or non-finite vector",
        ));
    }
    let inverse = 1.0 / sum.sqrt();
    let mut output = vec![0.0f32; input.len()];
    // SAFETY: input and output have the same length, and is_available()
    // returned true.
    unsafe { imp::scale_unweighted(input, inverse, &mut output) };
    if output.iter().any(|value| !value.is_finite()) {
        return Err(Error::internal(
            "L2 normalization produced a non-finite value",
        ));
    }
    Ok(output)
}

/// Mean-pools a row-major matrix over its rows.
pub(crate) fn mean_pool(values: &[f32], rows: usize, width: usize) -> Result<Vec<f32>> {
    validate_mean_pool(values, rows, width)?;
    if !is_available() {
        return mean_pool_scalar(values, rows, width);
    }
    debug_assert_eq!(values.len(), rows * width);
    debug_assert!(rows > 0);
    let mut output = vec![0.0f32; width];
    // SAFETY: values has rows * width elements, output has width elements, and
    // is_available() returned true.
    unsafe { imp::mean_pool(values, rows, width, &mut output) };
    if output.iter().any(|value| !value.is_finite()) {
        return Err(Error::internal("mean pooling produced a non-finite value"));
    }
    Ok(output)
}

#[cfg(target_arch = "aarch64")]
mod imp {
    use core::arch::aarch64::*;

    /// Loads four contiguous f32 values.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `ptr` is valid for four f32 reads.
    #[inline]
    unsafe fn load(ptr: *const f32) -> float32x4_t {
        // SAFETY: the caller guarantees the four-element bounds.
        unsafe { vld1q_f32(ptr) }
    }

    /// Stores four contiguous f32 values.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `ptr` is valid for four f32 writes.
    #[inline]
    unsafe fn store(ptr: *mut f32, value: float32x4_t) {
        // SAFETY: the caller guarantees the four-element bounds.
        unsafe { vst1q_f32(ptr, value) }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn add(left: &[f32], right: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0f32; left.len()];
        let mut index = 0;
        while index + 4 <= left.len() {
            // SAFETY: index + 4 <= length for both inputs and the output.
            let a = unsafe { load(left.as_ptr().add(index)) };
            let b = unsafe { load(right.as_ptr().add(index)) };
            let sum = vaddq_f32(a, b);
            // SAFETY: the output has the same length as the inputs.
            unsafe { store(output.as_mut_ptr().add(index), sum) };
            index += 4;
        }
        while index < left.len() {
            output[index] = left[index] + right[index];
            index += 1;
        }
        output
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn sub(left: &[f32], right: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0f32; left.len()];
        let mut index = 0;
        while index + 4 <= left.len() {
            // SAFETY: index + 4 <= length for both inputs and the output.
            let a = unsafe { load(left.as_ptr().add(index)) };
            let b = unsafe { load(right.as_ptr().add(index)) };
            let difference = vsubq_f32(a, b);
            // SAFETY: the output has the same length as the inputs.
            unsafe { store(output.as_mut_ptr().add(index), difference) };
            index += 4;
        }
        while index < left.len() {
            output[index] = left[index] - right[index];
            index += 1;
        }
        output
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn mul(left: &[f32], right: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0f32; left.len()];
        let mut index = 0;
        while index + 4 <= left.len() {
            // SAFETY: index + 4 <= length for both inputs and the output.
            let a = unsafe { load(left.as_ptr().add(index)) };
            let b = unsafe { load(right.as_ptr().add(index)) };
            let product = vmulq_f32(a, b);
            // SAFETY: the output has the same length as the inputs.
            unsafe { store(output.as_mut_ptr().add(index), product) };
            index += 4;
        }
        while index < left.len() {
            output[index] = left[index] * right[index];
            index += 1;
        }
        output
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn scale(values: &[f32], factor: f32) -> Vec<f32> {
        let mut output = vec![0.0f32; values.len()];
        let broadcast = vdupq_n_f32(factor);
        let mut index = 0;
        while index + 4 <= values.len() {
            // SAFETY: index + 4 <= length for the input and the output.
            let value = unsafe { load(values.as_ptr().add(index)) };
            let scaled = vmulq_f32(value, broadcast);
            // SAFETY: the output has the same length as the input.
            unsafe { store(output.as_mut_ptr().add(index), scaled) };
            index += 4;
        }
        while index < values.len() {
            output[index] = values[index] * factor;
            index += 1;
        }
        output
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn dot(left: &[f32], right: &[f32]) -> f32 {
        let length = left.len();
        let mut index = 0;
        let mut accumulator0 = vdupq_n_f32(0.0);
        let mut accumulator1 = vdupq_n_f32(0.0);
        let mut accumulator2 = vdupq_n_f32(0.0);
        let mut accumulator3 = vdupq_n_f32(0.0);

        while index + 16 <= length {
            // SAFETY: index + 16 <= length for both slices.
            let a0 = unsafe { load(left.as_ptr().add(index)) };
            let a1 = unsafe { load(left.as_ptr().add(index + 4)) };
            let a2 = unsafe { load(left.as_ptr().add(index + 8)) };
            let a3 = unsafe { load(left.as_ptr().add(index + 12)) };
            let b0 = unsafe { load(right.as_ptr().add(index)) };
            let b1 = unsafe { load(right.as_ptr().add(index + 4)) };
            let b2 = unsafe { load(right.as_ptr().add(index + 8)) };
            let b3 = unsafe { load(right.as_ptr().add(index + 12)) };
            accumulator0 = vmlaq_f32(accumulator0, a0, b0);
            accumulator1 = vmlaq_f32(accumulator1, a1, b1);
            accumulator2 = vmlaq_f32(accumulator2, a2, b2);
            accumulator3 = vmlaq_f32(accumulator3, a3, b3);
            index += 16;
        }

        while index + 4 <= length {
            // SAFETY: index + 4 <= length for both slices.
            let a = unsafe { load(left.as_ptr().add(index)) };
            let b = unsafe { load(right.as_ptr().add(index)) };
            accumulator0 = vmlaq_f32(accumulator0, a, b);
            index += 4;
        }

        let mut sum = vaddvq_f32(vaddq_f32(
            vaddq_f32(accumulator0, accumulator1),
            vaddq_f32(accumulator2, accumulator3),
        ));
        while index < length {
            sum += left[index] * right[index];
            index += 1;
        }
        sum
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn add_in_place(target: &mut [f32], other: &[f32]) {
        let mut index = 0;
        while index + 4 <= target.len() {
            // SAFETY: index + 4 <= length for both slices.
            let current = unsafe { load(target.as_ptr().add(index)) };
            let addition = unsafe { load(other.as_ptr().add(index)) };
            let sum = vaddq_f32(current, addition);
            // SAFETY: the target has the validated length.
            unsafe { store(target.as_mut_ptr().add(index), sum) };
            index += 4;
        }
        while index < target.len() {
            target[index] += other[index];
            index += 1;
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn matmul_into(
        left: &[f32],
        right: &[f32],
        m: usize,
        k: usize,
        n: usize,
        output: &mut [f32],
    ) {
        output.fill(0.0);
        for row in 0..m {
            let left_row = &left[row * k..(row + 1) * k];
            let output_row = &mut output[row * n..(row + 1) * n];
            for (inner, left_value) in left_row.iter().copied().enumerate() {
                let broadcast = vdupq_n_f32(left_value);
                let right_row = &right[inner * n..(inner + 1) * n];
                let mut column = 0;
                while column + 4 <= n {
                    // SAFETY: column + 4 <= n for both rows.
                    let values = unsafe { load(right_row.as_ptr().add(column)) };
                    let accumulated = unsafe { load(output_row.as_ptr().add(column)) };
                    let sum = vmlaq_f32(accumulated, broadcast, values);
                    // SAFETY: column + 4 <= n for the output row.
                    unsafe { store(output_row.as_mut_ptr().add(column), sum) };
                    column += 4;
                }
                while column < n {
                    output_row[column] += left_value * right_row[column];
                    column += 1;
                }
            }
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn linear_into(
        input: &[f32],
        weight: &[f32],
        rows: usize,
        in_features: usize,
        out_features: usize,
        output: &mut [f32],
    ) {
        for row in 0..rows {
            let input_row = &input[row * in_features..(row + 1) * in_features];
            let output_row = &mut output[row * out_features..(row + 1) * out_features];
            for (column, destination) in output_row.iter_mut().enumerate() {
                let weight_row = &weight[column * in_features..(column + 1) * in_features];
                // SAFETY: both rows have in_features elements, and this
                // function has the NEON target feature.
                *destination = unsafe { dot(input_row, weight_row) };
            }
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn scale_weighted(
        input: &[f32],
        weight: &[f32],
        factor: f32,
        output: &mut [f32],
    ) {
        let broadcast = vdupq_n_f32(factor);
        let mut index = 0;
        while index + 4 <= input.len() {
            // SAFETY: index + 4 <= length for all three slices.
            let value = unsafe { load(input.as_ptr().add(index)) };
            let scale = unsafe { load(weight.as_ptr().add(index)) };
            let result = vmulq_f32(vmulq_f32(value, broadcast), scale);
            // SAFETY: the output has the validated length.
            unsafe { store(output.as_mut_ptr().add(index), result) };
            index += 4;
        }
        while index < input.len() {
            output[index] = input[index] * factor * weight[index];
            index += 1;
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn scale_unweighted(input: &[f32], factor: f32, output: &mut [f32]) {
        let broadcast = vdupq_n_f32(factor);
        let mut index = 0;
        while index + 4 <= input.len() {
            // SAFETY: index + 4 <= length for both slices.
            let value = unsafe { load(input.as_ptr().add(index)) };
            let result = vmulq_f32(value, broadcast);
            // SAFETY: the output has the validated length.
            unsafe { store(output.as_mut_ptr().add(index), result) };
            index += 4;
        }
        while index < input.len() {
            output[index] = input[index] * factor;
            index += 1;
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn mean_pool(values: &[f32], rows: usize, width: usize, output: &mut [f32]) {
        for row in 0..rows {
            let base = row * width;
            let mut column = 0;
            while column + 4 <= width {
                // SAFETY: base + column + 4 <= rows * width.
                let accumulated = unsafe { load(output.as_ptr().add(column)) };
                let value = unsafe { load(values.as_ptr().add(base + column)) };
                let sum = vaddq_f32(accumulated, value);
                // SAFETY: column + 4 <= width for the output.
                unsafe { store(output.as_mut_ptr().add(column), sum) };
                column += 4;
            }
            while column < width {
                output[column] += values[base + column];
                column += 1;
            }
        }

        let inverse = vdupq_n_f32(1.0 / rows as f32);
        let mut column = 0;
        while column + 4 <= width {
            // SAFETY: column + 4 <= width for the output.
            let value = unsafe { load(output.as_ptr().add(column)) };
            let scaled = vmulq_f32(value, inverse);
            // SAFETY: column + 4 <= width for the output.
            unsafe { store(output.as_mut_ptr().add(column), scaled) };
            column += 4;
        }
        while column < width {
            output[column] *= 1.0 / rows as f32;
            column += 1;
        }
    }
}
