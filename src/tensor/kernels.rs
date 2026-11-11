//! Reference FP32 scalar kernels.
//!
//! These functions are the correctness reference for the engine. They are
//! deterministic, single-threaded, and free of unsafe code. SIMD kernels will
//! be compared against them before any optimization is accepted.
//!
//! Every operation validates lengths and allocation sizes before touching
//! memory, and rejects non-finite intermediate results where a silent NaN
//! would corrupt the embedding space.

use crate::error::{Error, Result};
use crate::tensor::owned::MAX_TENSOR_ELEMENTS;

/// Validates that two slices have the same length.
fn check_same_length(left: &[f32], right: &[f32], operation: &str) -> Result<()> {
    if left.len() != right.len() {
        return Err(Error::internal(format!(
            "{operation} length mismatch: {} and {}",
            left.len(),
            right.len()
        )));
    }
    Ok(())
}

/// Validates that an output allocation stays under the tensor cap.
fn check_allocation(count: usize, operation: &str) -> Result<()> {
    if count > MAX_TENSOR_ELEMENTS {
        return Err(Error::limit_exceeded(format!(
            "{operation} would allocate {count} elements, above the {MAX_TENSOR_ELEMENTS}-element cap"
        )));
    }
    Ok(())
}

/// Element-wise addition.
pub fn add(left: &[f32], right: &[f32]) -> Result<Vec<f32>> {
    check_same_length(left, right, "addition")?;
    check_allocation(left.len(), "addition")?;
    Ok(left.iter().zip(right).map(|(a, b)| a + b).collect())
}

/// Element-wise subtraction.
pub fn sub(left: &[f32], right: &[f32]) -> Result<Vec<f32>> {
    check_same_length(left, right, "subtraction")?;
    check_allocation(left.len(), "subtraction")?;
    Ok(left.iter().zip(right).map(|(a, b)| a - b).collect())
}

/// Element-wise multiplication.
pub fn mul(left: &[f32], right: &[f32]) -> Result<Vec<f32>> {
    check_same_length(left, right, "multiplication")?;
    check_allocation(left.len(), "multiplication")?;
    Ok(left.iter().zip(right).map(|(a, b)| a * b).collect())
}

/// Multiplies every element by a finite scalar.
pub fn scale(values: &[f32], factor: f32) -> Result<Vec<f32>> {
    if !factor.is_finite() {
        return Err(Error::internal("scale factor is not finite"));
    }
    check_allocation(values.len(), "scaling")?;
    Ok(values.iter().map(|value| value * factor).collect())
}

/// Dot product of two vectors.
///
/// # Errors
///
/// Returns Error::Internal on a length mismatch or a non-finite result.
pub fn dot(left: &[f32], right: &[f32]) -> Result<f32> {
    check_same_length(left, right, "dot product")?;
    let mut sum = 0.0f32;
    for (a, b) in left.iter().zip(right) {
        sum += a * b;
    }
    if !sum.is_finite() {
        return Err(Error::internal("dot product produced a non-finite value"));
    }
    Ok(sum)
}

/// Adds one slice into another in place.
pub fn add_in_place(target: &mut [f32], other: &[f32]) -> Result<()> {
    if target.len() != other.len() {
        return Err(Error::internal(format!(
            "in-place addition length mismatch: {} and {}",
            target.len(),
            other.len()
        )));
    }
    for (destination, value) in target.iter_mut().zip(other) {
        *destination += value;
    }
    Ok(())
}

/// Matrix multiplication C = A * B.
///
/// A is m by k, B is k by n, and C is m by n, all row-major.
pub fn matmul(left: &[f32], right: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
    let output_len = m
        .checked_mul(n)
        .ok_or_else(|| Error::limit_exceeded("matmul output size overflow"))?;
    check_allocation(output_len, "matmul")?;
    let mut output = vec![0.0f32; output_len];
    matmul_into(left, right, m, k, n, &mut output)?;
    Ok(output)
}

/// Matrix multiplication into a caller-provided output buffer.
///
/// The output buffer is cleared before accumulation, so it can be reused
/// across layers without reallocating.
pub fn matmul_into(
    left: &[f32],
    right: &[f32],
    m: usize,
    k: usize,
    n: usize,
    output: &mut [f32],
) -> Result<()> {
    let left_len = m
        .checked_mul(k)
        .ok_or_else(|| Error::limit_exceeded("matmul left operand size overflow"))?;
    let right_len = k
        .checked_mul(n)
        .ok_or_else(|| Error::limit_exceeded("matmul right operand size overflow"))?;
    let output_len = m
        .checked_mul(n)
        .ok_or_else(|| Error::limit_exceeded("matmul output size overflow"))?;

    if left.len() != left_len {
        return Err(Error::internal(format!(
            "matmul left operand has {} elements, expected {left_len} for {m}x{k}",
            left.len()
        )));
    }
    if right.len() != right_len {
        return Err(Error::internal(format!(
            "matmul right operand has {} elements, expected {right_len} for {k}x{n}",
            right.len()
        )));
    }
    if output.len() != output_len {
        return Err(Error::internal(format!(
            "matmul output has {} elements, expected {output_len} for {m}x{n}",
            output.len()
        )));
    }
    check_allocation(output_len, "matmul")?;

    output.fill(0.0);
    for row in 0..m {
        let left_row = &left[row * k..(row + 1) * k];
        let output_row = &mut output[row * n..(row + 1) * n];
        for (inner, left_value) in left_row.iter().enumerate() {
            let right_row = &right[inner * n..(inner + 1) * n];
            for (destination, right_value) in output_row.iter_mut().zip(right_row) {
                *destination += left_value * right_value;
            }
        }
    }

    if output.iter().any(|value| !value.is_finite()) {
        return Err(Error::internal("matmul produced a non-finite value"));
    }
    Ok(())
}

/// Numerically stable softmax.
///
/// The maximum is subtracted before exponentiation. A vector containing only
/// negative infinity has no valid distribution and is rejected. A vector with
/// positive infinity entries distributes the mass uniformly over them.
pub fn softmax(values: &[f32]) -> Result<Vec<f32>> {
    if values.is_empty() {
        return Err(Error::internal("softmax of an empty slice"));
    }
    check_allocation(values.len(), "softmax")?;

    let mut maximum = f32::NEG_INFINITY;
    for &value in values {
        if value.is_nan() {
            return Err(Error::internal("softmax input contains NaN"));
        }
        if value > maximum {
            maximum = value;
        }
    }

    if maximum == f32::NEG_INFINITY {
        return Err(Error::internal(
            "softmax input is entirely negative infinity",
        ));
    }

    if maximum == f32::INFINITY {
        let infinite_count = values
            .iter()
            .filter(|value| **value == f32::INFINITY)
            .count();
        let probability = 1.0 / infinite_count as f32;
        return Ok(values
            .iter()
            .map(|value| {
                if *value == f32::INFINITY {
                    probability
                } else {
                    0.0
                }
            })
            .collect());
    }

    let mut output = Vec::with_capacity(values.len());
    let mut sum = 0.0f32;
    for &value in values {
        let exponential = (value - maximum).exp();
        sum += exponential;
        output.push(exponential);
    }
    if sum == 0.0 || !sum.is_finite() {
        return Err(Error::internal("softmax normalization failed"));
    }
    for value in &mut output {
        *value /= sum;
    }
    Ok(output)
}

/// Root mean square normalization.
///
/// Computes input / sqrt(mean(input^2) + epsilon) * weight. The weight is
/// applied after normalization, as in the reference model.
pub fn rms_norm(input: &[f32], weight: &[f32], epsilon: f32) -> Result<Vec<f32>> {
    check_same_length(input, weight, "RMSNorm")?;
    if input.is_empty() {
        return Err(Error::internal("RMSNorm of an empty slice"));
    }
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(Error::internal(
            "RMSNorm epsilon must be finite and non-negative",
        ));
    }
    check_allocation(input.len(), "RMSNorm")?;

    let mut sum = 0.0f32;
    for &value in input {
        if !value.is_finite() {
            return Err(Error::internal("RMSNorm input contains a non-finite value"));
        }
        sum += value * value;
    }
    if !sum.is_finite() {
        return Err(Error::internal("RMSNorm sum of squares overflowed"));
    }

    let mean = sum / input.len() as f32;
    let inverse = 1.0 / (mean + epsilon).sqrt();
    let output: Vec<f32> = input
        .iter()
        .zip(weight)
        .map(|(value, scale)| value * inverse * scale)
        .collect();
    if output.iter().any(|value| !value.is_finite()) {
        return Err(Error::internal("RMSNorm produced a non-finite value"));
    }
    Ok(output)
}

/// GELU with the tanh approximation used by Gemma-style models.
pub fn gelu_tanh_value(value: f32) -> f32 {
    let cubic = value * value * value;
    0.5 * value * (1.0 + (0.797_884_6 * (value + 0.044_715 * cubic)).tanh())
}

/// Element-wise GELU with the tanh approximation.
pub fn gelu_tanh(values: &[f32]) -> Vec<f32> {
    values.iter().copied().map(gelu_tanh_value).collect()
}

/// Gated GELU: gelu(gate) * up.
pub fn geglu(gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
    check_same_length(gate, up, "GeGLU")?;
    check_allocation(gate.len(), "GeGLU")?;
    let activated = gelu_tanh(gate);
    Ok(activated
        .iter()
        .zip(up)
        .map(|(gate_value, up_value)| gate_value * up_value)
        .collect())
}

/// L2 normalization.
///
/// A zero or non-finite vector has no defined direction and is rejected.
pub fn l2_normalize(input: &[f32]) -> Result<Vec<f32>> {
    if input.is_empty() {
        return Err(Error::internal("cannot normalize an empty vector"));
    }
    check_allocation(input.len(), "L2 normalization")?;

    let mut sum = 0.0f32;
    for &value in input {
        if !value.is_finite() {
            return Err(Error::internal(
                "L2 normalization input contains a non-finite value",
            ));
        }
        sum += value * value;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(Error::internal(
            "cannot normalize a zero or non-finite vector",
        ));
    }

    let inverse = 1.0 / sum.sqrt();
    Ok(input.iter().map(|value| value * inverse).collect())
}

/// Applies rotary position embeddings in place.
///
/// The layout is [sequence, heads, head_dim], row-major. Within each head,
/// the first half is paired with the second half, matching the rotate-half
/// convention used by the reference implementation. head_dim must be even.
pub fn rope(
    values: &mut [f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    base: f32,
    position_offset: usize,
) -> Result<()> {
    if head_dim == 0 || head_dim % 2 != 0 {
        return Err(Error::internal(
            "RoPE head dimension must be a positive even number",
        ));
    }
    if !base.is_finite() || base <= 1.0 {
        return Err(Error::internal(
            "RoPE base must be finite and greater than one",
        ));
    }
    let expected = seq_len
        .checked_mul(num_heads)
        .and_then(|count| count.checked_mul(head_dim))
        .ok_or_else(|| Error::limit_exceeded("RoPE shape overflow"))?;
    if expected != values.len() {
        return Err(Error::internal(format!(
            "RoPE expects {expected} elements, got {}",
            values.len()
        )));
    }

    let half = head_dim / 2;
    for position in 0..seq_len {
        let absolute = position_offset
            .checked_add(position)
            .ok_or_else(|| Error::limit_exceeded("RoPE position overflow"))?;
        for head in 0..num_heads {
            let offset = (position * num_heads + head) * head_dim;
            for pair in 0..half {
                let exponent = (2 * pair) as f32 / head_dim as f32;
                let angle = absolute as f32 / base.powf(exponent);
                let (sine, cosine) = angle.sin_cos();
                let first = values[offset + pair];
                let second = values[offset + pair + half];
                values[offset + pair] = first * cosine - second * sine;
                values[offset + pair + half] = first * sine + second * cosine;
            }
        }
    }
    Ok(())
}

/// Scaled dot-product attention for one head.
///
/// Query, key, and value are [sequence, head_dim] row-major. When causal is
/// true, position i attends only to positions j <= i.
pub fn attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    seq_len: usize,
    head_dim: usize,
    causal: bool,
) -> Result<Vec<f32>> {
    if head_dim == 0 {
        return Err(Error::internal("attention head dimension must be positive"));
    }
    let expected = seq_len
        .checked_mul(head_dim)
        .ok_or_else(|| Error::limit_exceeded("attention shape overflow"))?;
    if query.len() != expected || key.len() != expected || value.len() != expected {
        return Err(Error::internal(format!(
            "attention expects three buffers of {expected} elements, got {}, {}, {}",
            query.len(),
            key.len(),
            value.len()
        )));
    }
    check_allocation(expected, "attention")?;

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; expected];
    let mut scores = vec![0.0f32; seq_len];

    for row in 0..seq_len {
        let query_row = &query[row * head_dim..(row + 1) * head_dim];
        for column in 0..seq_len {
            if causal && column > row {
                scores[column] = f32::NEG_INFINITY;
                continue;
            }
            let key_row = &key[column * head_dim..(column + 1) * head_dim];
            let mut dot_product = 0.0f32;
            for (query_value, key_value) in query_row.iter().zip(key_row) {
                dot_product += query_value * key_value;
            }
            let score = dot_product * scale;
            if !score.is_finite() {
                return Err(Error::internal(
                    "attention score overflowed to a non-finite value",
                ));
            }
            scores[column] = score;
        }

        let probabilities = softmax(&scores)?;
        let output_row = &mut output[row * head_dim..(row + 1) * head_dim];
        for (column, probability) in probabilities.iter().enumerate() {
            if *probability == 0.0 {
                continue;
            }
            let value_row = &value[column * head_dim..(column + 1) * head_dim];
            for (destination, value_element) in output_row.iter_mut().zip(value_row) {
                *destination += probability * value_element;
            }
        }
    }

    if output.iter().any(|value| !value.is_finite()) {
        return Err(Error::internal(
            "attention produced a non-finite output value",
        ));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() <= tolerance,
                "index {index}: {a} differs from {e} by more than {tolerance}"
            );
        }
    }

    #[test]
    fn elementwise_operations_work() {
        assert_eq!(add(&[1.0, 2.0], &[3.0, 4.0]).expect("add"), vec![4.0, 6.0]);
        assert_eq!(
            sub(&[1.0, 2.0], &[3.0, 4.0]).expect("sub"),
            vec![-2.0, -2.0]
        );
        assert_eq!(mul(&[1.0, 2.0], &[3.0, 4.0]).expect("mul"), vec![3.0, 8.0]);
        assert_eq!(scale(&[1.0, -2.0], 2.0).expect("scale"), vec![2.0, -4.0]);
    }

    #[test]
    fn elementwise_operations_reject_length_mismatches() {
        for result in [
            add(&[1.0], &[1.0, 2.0]),
            sub(&[1.0], &[1.0, 2.0]),
            mul(&[1.0], &[1.0, 2.0]),
        ] {
            let error = result.expect_err("length mismatch must fail");
            assert!(matches!(error, Error::Internal(_)));
        }
    }

    #[test]
    fn scale_rejects_non_finite_factor() {
        let error = scale(&[1.0], f32::NAN).expect_err("NaN factor must fail");
        assert!(matches!(error, Error::Internal(_)));
    }

    #[test]
    fn dot_product_is_correct() {
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).expect("dot"), 32.0);
        assert_eq!(dot(&[], &[]).expect("empty dot"), 0.0);
        let error = dot(&[1.0], &[2.0, 3.0]).expect_err("mismatch must fail");
        assert!(matches!(error, Error::Internal(_)));
    }

    #[test]
    fn in_place_addition_works() {
        let mut target = [1.0, 2.0];
        add_in_place(&mut target, &[3.0, 4.0]).expect("add in place");
        assert_eq!(target, [4.0, 6.0]);
        let error = add_in_place(&mut target, &[1.0]).expect_err("mismatch must fail");
        assert!(matches!(error, Error::Internal(_)));
    }

    #[test]
    fn matmul_is_correct() {
        let left = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let right = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let output = matmul(&left, &right, 2, 3, 2).expect("matmul");
        assert_eq!(output, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn matmul_identity_is_identity() {
        let identity = [1.0, 0.0, 0.0, 1.0];
        let values = [2.0, 3.0, 4.0, 5.0];
        let output = matmul(&values, &identity, 2, 2, 2).expect("matmul");
        assert_eq!(output, values);
    }

    #[test]
    fn matmul_into_reuses_the_buffer() {
        let left = [1.0, 2.0, 3.0, 4.0];
        let right = [5.0, 6.0, 7.0, 8.0];
        let mut output = [99.0; 4];
        matmul_into(&left, &right, 2, 2, 2, &mut output).expect("matmul into");
        assert_eq!(output, [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matmul_rejects_shape_mismatches() {
        let error = matmul(&[1.0], &[1.0], 1, 1, 2).expect_err("mismatch must fail");
        assert!(matches!(error, Error::Internal(_)));
    }

    #[test]
    fn softmax_sums_to_one_and_preserves_order() {
        let output = softmax(&[1.0, 2.0, 3.0]).expect("softmax");
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(output[0] < output[1]);
        assert!(output[1] < output[2]);
    }

    #[test]
    fn softmax_is_stable_for_large_values() {
        let output = softmax(&[1000.0, 1001.0]).expect("softmax");
        assert!(output.iter().all(|value| value.is_finite()));
        assert!((output[0] + output[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_handles_infinities() {
        let output = softmax(&[f32::INFINITY, 1.0]).expect("softmax");
        assert_eq!(output, vec![1.0, 0.0]);

        let error = softmax(&[f32::NEG_INFINITY, f32::NEG_INFINITY])
            .expect_err("all negative infinity must fail");
        assert!(matches!(error, Error::Internal(_)));
    }

    #[test]
    fn softmax_rejects_nan_and_empty_input() {
        assert!(matches!(
            softmax(&[1.0, f32::NAN]).expect_err("NaN must fail"),
            Error::Internal(_)
        ));
        assert!(matches!(
            softmax(&[]).expect_err("empty input must fail"),
            Error::Internal(_)
        ));
    }

    #[test]
    fn rms_norm_is_correct() {
        let output = rms_norm(&[3.0, 4.0], &[1.0, 1.0], 0.0).expect("RMSNorm");
        assert_close(&output, &[0.848_528, 1.131_371], 1e-5);
        let weighted = rms_norm(&[3.0, 4.0], &[2.0, 2.0], 0.0).expect("weighted RMSNorm");
        assert_close(&weighted, &[1.697_056, 2.262_741], 1e-5);
    }

    #[test]
    fn rms_norm_handles_zero_input_with_epsilon() {
        let output = rms_norm(&[0.0, 0.0], &[1.0, 1.0], 1e-6).expect("zero input");
        assert_eq!(output, vec![0.0, 0.0]);
    }

    #[test]
    fn rms_norm_rejects_invalid_inputs() {
        assert!(matches!(
            rms_norm(&[], &[], 0.0).expect_err("empty must fail"),
            Error::Internal(_)
        ));
        assert!(matches!(
            rms_norm(&[1.0], &[1.0], -1.0).expect_err("negative epsilon must fail"),
            Error::Internal(_)
        ));
        assert!(matches!(
            rms_norm(&[f32::NAN], &[1.0], 0.0).expect_err("NaN must fail"),
            Error::Internal(_)
        ));
        assert!(matches!(
            rms_norm(&[1.0], &[1.0, 2.0], 0.0).expect_err("mismatch must fail"),
            Error::Internal(_)
        ));
    }

    #[test]
    fn gelu_is_correct_at_reference_points() {
        assert_close(&[gelu_tanh_value(0.0)], &[0.0], 1e-7);
        assert_close(&[gelu_tanh_value(1.0)], &[0.841_192], 1e-5);
        assert_close(&[gelu_tanh_value(-1.0)], &[-0.158_808], 1e-5);
        assert_close(&[gelu_tanh_value(10.0)], &[10.0], 1e-4);
    }

    #[test]
    fn geglu_multiplies_the_activated_gate() {
        let output = geglu(&[1.0, 0.0], &[2.0, 3.0]).expect("GeGLU");
        assert_close(&output, &[1.682_384, 0.0], 1e-5);
        assert!(matches!(
            geglu(&[1.0], &[1.0, 2.0]).expect_err("mismatch must fail"),
            Error::Internal(_)
        ));
    }

    #[test]
    fn l2_normalization_produces_unit_vectors() {
        let output = l2_normalize(&[3.0, 4.0]).expect("normalize");
        assert_close(&output, &[0.6, 0.8], 1e-7);
        let norm: f32 = output.iter().map(|value| value * value).sum();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn l2_normalization_rejects_zero_and_non_finite_vectors() {
        assert!(matches!(
            l2_normalize(&[0.0, 0.0]).expect_err("zero vector must fail"),
            Error::Internal(_)
        ));
        assert!(matches!(
            l2_normalize(&[f32::NAN, 1.0]).expect_err("NaN must fail"),
            Error::Internal(_)
        ));
        assert!(matches!(
            l2_normalize(&[]).expect_err("empty vector must fail"),
            Error::Internal(_)
        ));
    }

    #[test]
    fn rope_at_position_zero_is_identity() {
        let mut values = [1.0, 2.0, 3.0, 4.0];
        let original = values;
        rope(&mut values, 1, 1, 4, 10_000.0, 0).expect("RoPE");
        assert_close(&values, &original, 1e-6);
    }

    #[test]
    fn rope_rotates_and_preserves_the_norm() {
        let mut values = [1.0, 0.0];
        rope(&mut values, 1, 1, 2, 10_000.0, 1).expect("RoPE");
        assert_close(&values, &[0.540_302, 0.841_471], 1e-5);
        let norm: f32 = values.iter().map(|value| value * value).sum();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rope_rejects_invalid_parameters() {
        assert!(matches!(
            rope(&mut [1.0], 1, 1, 1, 10_000.0, 0).expect_err("odd head_dim must fail"),
            Error::Internal(_)
        ));
        assert!(matches!(
            rope(&mut [1.0, 2.0], 1, 1, 2, 1.0, 0).expect_err("base one must fail"),
            Error::Internal(_)
        ));
        assert!(matches!(
            rope(&mut [1.0, 2.0], 2, 1, 2, 10_000.0, 0).expect_err("length mismatch must fail"),
            Error::Internal(_)
        ));
    }

    #[test]
    fn causal_attention_masks_future_positions() {
        let query = [1.0, 1.0];
        let key = [1.0, 0.0];
        let value = [10.0, 20.0];
        let causal = attention(&query, &key, &value, 2, 1, true).expect("causal attention");
        assert_close(&causal, &[10.0, 12.689_4], 1e-4);

        let full = attention(&query, &key, &value, 2, 1, false).expect("full attention");
        assert_close(&full, &[12.689_4, 12.689_4], 1e-4);
    }

    #[test]
    fn attention_rejects_invalid_shapes() {
        assert!(matches!(
            attention(&[1.0], &[1.0], &[1.0], 1, 0, true).expect_err("zero head_dim must fail"),
            Error::Internal(_)
        ));
        assert!(matches!(
            attention(&[1.0, 2.0], &[1.0], &[1.0, 2.0], 1, 1, true)
                .expect_err("length mismatch must fail"),
            Error::Internal(_)
        ));
    }
}
