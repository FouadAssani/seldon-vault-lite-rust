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

/// Matrix-vector product against a weight matrix stored as [out_features, in_features].
///
/// This is the layout used by every linear layer of the reference model:
/// output[row, column] = sum_k input[row, k] * weight[column, k].
/// The weight row is contiguous, which keeps the inner loop cache-friendly.
pub fn linear(
    input: &[f32],
    weight: &[f32],
    rows: usize,
    in_features: usize,
    out_features: usize,
) -> Result<Vec<f32>> {
    let output_len = rows
        .checked_mul(out_features)
        .ok_or_else(|| Error::limit_exceeded("linear output size overflow"))?;
    check_allocation(output_len, "linear")?;
    let mut output = vec![0.0f32; output_len];
    linear_into(input, weight, rows, in_features, out_features, &mut output)?;
    Ok(output)
}

/// Linear projection into a caller-provided output buffer.
///
/// The output buffer is cleared before accumulation so it can be reused
/// between layers without reallocating.
pub fn linear_into(
    input: &[f32],
    weight: &[f32],
    rows: usize,
    in_features: usize,
    out_features: usize,
    output: &mut [f32],
) -> Result<()> {
    let input_len = rows
        .checked_mul(in_features)
        .ok_or_else(|| Error::limit_exceeded("linear input size overflow"))?;
    let weight_len = out_features
        .checked_mul(in_features)
        .ok_or_else(|| Error::limit_exceeded("linear weight size overflow"))?;
    let output_len = rows
        .checked_mul(out_features)
        .ok_or_else(|| Error::limit_exceeded("linear output size overflow"))?;

    if input.len() != input_len {
        return Err(Error::internal(format!(
            "linear input has {} elements, expected {input_len} for {rows}x{in_features}",
            input.len()
        )));
    }
    if weight.len() != weight_len {
        return Err(Error::internal(format!(
            "linear weight has {} elements, expected {weight_len} for {out_features}x{in_features}",
            weight.len()
        )));
    }
    if output.len() != output_len {
        return Err(Error::internal(format!(
            "linear output has {} elements, expected {output_len} for {rows}x{out_features}",
            output.len()
        )));
    }
    check_allocation(output_len, "linear")?;

    output.fill(0.0);
    for row in 0..rows {
        let input_row = &input[row * in_features..(row + 1) * in_features];
        let output_row = &mut output[row * out_features..(row + 1) * out_features];
        for (column, destination) in output_row.iter_mut().enumerate() {
            let weight_row = &weight[column * in_features..(column + 1) * in_features];
            let mut sum = 0.0f32;
            for (value, weight_value) in input_row.iter().zip(weight_row) {
                sum += value * weight_value;
            }
            *destination = sum;
        }
    }

    if output.iter().any(|value| !value.is_finite()) {
        return Err(Error::internal(
            "linear projection produced a non-finite value",
        ));
    }
    Ok(())
}

/// RMSNorm applied independently to each contiguous row of a matrix.
///
/// The layout is [rows, width] row-major. The same weight vector of length
/// width is applied to every row, which is the layout of the per-head query
/// and key normalizations.
pub fn rms_norm_rows(
    input: &[f32],
    weight: &[f32],
    rows: usize,
    width: usize,
    epsilon: f32,
) -> Result<Vec<f32>> {
    if weight.len() != width {
        return Err(Error::internal(format!(
            "RMSNorm row weight has {} elements, expected {width}",
            weight.len()
        )));
    }
    let expected = rows
        .checked_mul(width)
        .ok_or_else(|| Error::limit_exceeded("RMSNorm row count overflow"))?;
    if input.len() != expected {
        return Err(Error::internal(format!(
            "RMSNorm input has {} elements, expected {expected} for {rows}x{width}",
            input.len()
        )));
    }
    if width == 0 {
        return Err(Error::internal("RMSNorm row width must be positive"));
    }
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(Error::internal(
            "RMSNorm epsilon must be finite and non-negative",
        ));
    }
    check_allocation(expected, "RMSNorm")?;

    let mut output = vec![0.0f32; expected];
    for row in 0..rows {
        let input_row = &input[row * width..(row + 1) * width];
        let output_row = &mut output[row * width..(row + 1) * width];
        let mut sum = 0.0f32;
        for value in input_row {
            if !value.is_finite() {
                return Err(Error::internal("RMSNorm input contains a non-finite value"));
            }
            sum += value * value;
        }
        if !sum.is_finite() {
            return Err(Error::internal("RMSNorm sum of squares overflowed"));
        }
        let inverse = 1.0 / (sum / width as f32 + epsilon).sqrt();
        for ((destination, value), scale) in output_row.iter_mut().zip(input_row).zip(weight) {
            *destination = value * inverse * scale;
        }
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(Error::internal("RMSNorm produced a non-finite value"));
    }
    Ok(output)
}

/// RMSNorm without a learned scale, applied to each contiguous row.
///
/// The reference model uses this form for the value projection, whose norm
/// deliberately has no weight parameter.
pub fn rms_norm_rows_unscaled(
    input: &[f32],
    rows: usize,
    width: usize,
    epsilon: f32,
) -> Result<Vec<f32>> {
    let expected = rows
        .checked_mul(width)
        .ok_or_else(|| Error::limit_exceeded("RMSNorm row count overflow"))?;
    if input.len() != expected {
        return Err(Error::internal(format!(
            "RMSNorm input has {} elements, expected {expected} for {rows}x{width}",
            input.len()
        )));
    }
    if width == 0 {
        return Err(Error::internal("RMSNorm row width must be positive"));
    }
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(Error::internal(
            "RMSNorm epsilon must be finite and non-negative",
        ));
    }
    check_allocation(expected, "RMSNorm")?;

    let mut output = vec![0.0f32; expected];
    for row in 0..rows {
        let input_row = &input[row * width..(row + 1) * width];
        let output_row = &mut output[row * width..(row + 1) * width];
        let mut sum = 0.0f32;
        for value in input_row {
            if !value.is_finite() {
                return Err(Error::internal("RMSNorm input contains a non-finite value"));
            }
            sum += value * value;
        }
        if !sum.is_finite() {
            return Err(Error::internal("RMSNorm sum of squares overflowed"));
        }
        let inverse = 1.0 / (sum / width as f32 + epsilon).sqrt();
        for (destination, value) in output_row.iter_mut().zip(input_row) {
            *destination = value * inverse;
        }
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(Error::internal("RMSNorm produced a non-finite value"));
    }
    Ok(output)
}

/// Mean-pools a row-major matrix over its rows.
///
/// The reference SentenceTransformers pooling is the mean over all
/// non-padding tokens. For a single unpadded sequence, this is the mean over
/// every row of the projected hidden state.
pub fn mean_pool(values: &[f32], rows: usize, width: usize) -> Result<Vec<f32>> {
    let expected = rows
        .checked_mul(width)
        .ok_or_else(|| Error::limit_exceeded("mean pooling size overflow"))?;
    if values.len() != expected {
        return Err(Error::internal(format!(
            "mean pooling input has {} elements, expected {expected} for {rows}x{width}",
            values.len()
        )));
    }
    if rows == 0 {
        return Err(Error::internal("mean pooling of an empty sequence"));
    }
    check_allocation(width, "mean pooling")?;

    let mut output = vec![0.0f32; width];
    for row in 0..rows {
        for (column, destination) in output.iter_mut().enumerate() {
            *destination += values[row * width + column];
        }
    }
    let inverse = 1.0 / rows as f32;
    for value in &mut output {
        *value *= inverse;
        if !value.is_finite() {
            return Err(Error::internal("mean pooling produced a non-finite value"));
        }
    }
    Ok(output)
}

/// Truncates a vector to its leading dimensions and renormalizes it.
///
/// Matryoshka truncation does not preserve the unit norm: a slice of a unit
/// vector is shorter than one. The renormalization is mandatory and is applied
/// here so callers cannot silently skip it.
pub fn truncate_and_normalize(values: &[f32], dimension: usize) -> Result<Vec<f32>> {
    if dimension == 0 {
        return Err(Error::internal("embedding dimension must be positive"));
    }
    if dimension > values.len() {
        return Err(Error::internal(format!(
            "cannot truncate {} elements to {dimension}",
            values.len()
        )));
    }
    l2_normalize(&values[..dimension])
}

/// Applies rotary position embeddings using precomputed inverse frequencies.
///
/// The layout is [sequence, heads, head_dim], row-major. Within each head, the
/// first half is paired with the second half, matching the rotate-half
/// convention used by the reference implementation. The inv_freq slice contains
/// head_dim / 2 frequencies; precomputing them once per layer type avoids
/// recomputing powers inside the sequence loop.
pub fn rope_with_inv_freq(
    values: &mut [f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    inv_freq: &[f32],
    position_offset: usize,
) -> Result<()> {
    if head_dim == 0 || head_dim % 2 != 0 {
        return Err(Error::internal(
            "RoPE head dimension must be a positive even number",
        ));
    }
    if inv_freq.len() != head_dim / 2 {
        return Err(Error::internal(format!(
            "RoPE expects {} inverse frequencies for head_dim {head_dim}, got {}",
            head_dim / 2,
            inv_freq.len()
        )));
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
                let angle = absolute as f32 * inv_freq[pair];
                let (sine, cosine) = angle.sin_cos();
                let first = values[offset + pair];
                let second = values[offset + pair + half];
                values[offset + pair] = first * cosine - second * sine;
                values[offset + pair + half] = first * sine + second * cosine;
            }
        }
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Error::internal("RoPE produced a non-finite value"));
    }
    Ok(())
}

/// Builds the inverse frequencies of the default RoPE scheme.
pub fn rope_inv_freq(head_dim: usize, base: f32) -> Result<Vec<f32>> {
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
    let half = head_dim / 2;
    check_allocation(half, "RoPE inverse frequencies")?;
    let mut frequencies = Vec::with_capacity(half);
    for pair in 0..half {
        let exponent = (2 * pair) as f32 / head_dim as f32;
        frequencies.push(1.0 / base.powf(exponent));
    }
    Ok(frequencies)
}

/// Shape and masking parameters of one bidirectional attention call.
#[derive(Debug, Clone, Copy)]
pub struct AttentionShape {
    /// Number of query positions.
    pub seq_len: usize,
    /// Number of query heads.
    pub num_heads: usize,
    /// Number of key/value heads before grouped-query expansion.
    pub num_key_value_heads: usize,
    /// Dimension of one head.
    pub head_dim: usize,
    /// Inclusive sliding-window radius, or None for full attention.
    pub sliding_window: Option<usize>,
    /// Multiplicative score scale; EmbeddingGemma 2 uses 1.0.
    pub scale: f32,
}

/// Bidirectional grouped-query attention for one layer.
///
/// Query is [sequence, num_heads, head_dim], key and value are
/// [sequence, num_key_value_heads, head_dim]. Query head h reads key/value head
/// h / (num_heads / num_key_value_heads), matching the repeat_kv order of the
/// reference implementation. When sliding_window is Some(r), position i
/// attends to every position j with |i - j| <= r; otherwise attention is full.
/// The caller supplies the scale, which is 1.0 for EmbeddingGemma 2.
pub fn bidirectional_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    shape: AttentionShape,
) -> Result<Vec<f32>> {
    let AttentionShape {
        seq_len,
        num_heads,
        num_key_value_heads,
        head_dim,
        sliding_window,
        scale,
    } = shape;
    if head_dim == 0 {
        return Err(Error::internal("attention head dimension must be positive"));
    }
    if num_heads == 0 || num_key_value_heads == 0 {
        return Err(Error::internal("attention head counts must be positive"));
    }
    if num_heads % num_key_value_heads != 0 {
        return Err(Error::internal(
            "attention head count is not a multiple of the key/value head count",
        ));
    }
    if !scale.is_finite() {
        return Err(Error::internal("attention scale must be finite"));
    }
    let query_len = seq_len
        .checked_mul(num_heads)
        .and_then(|count| count.checked_mul(head_dim))
        .ok_or_else(|| Error::limit_exceeded("attention query size overflow"))?;
    let key_len = seq_len
        .checked_mul(num_key_value_heads)
        .and_then(|count| count.checked_mul(head_dim))
        .ok_or_else(|| Error::limit_exceeded("attention key size overflow"))?;
    if query.len() != query_len || key.len() != key_len || value.len() != key_len {
        return Err(Error::internal(format!(
            "attention expects {query_len} query and {key_len} key/value elements, got {}, {}, {}",
            query.len(),
            key.len(),
            value.len()
        )));
    }
    check_allocation(query_len, "attention")?;

    let groups = num_heads / num_key_value_heads;
    let mut output = vec![0.0f32; query_len];
    let mut scores = vec![0.0f32; seq_len];

    for position in 0..seq_len {
        let (start, end) = match sliding_window {
            Some(window) => (
                position.saturating_sub(window),
                (position + window).min(seq_len - 1),
            ),
            None => (0, seq_len - 1),
        };
        let width = end - start + 1;
        for head in 0..num_heads {
            let key_head = head / groups;
            let query_offset = (position * num_heads + head) * head_dim;
            let query_row = &query[query_offset..query_offset + head_dim];
            for (column, offset) in (start..=end).enumerate() {
                let key_offset = (offset * num_key_value_heads + key_head) * head_dim;
                let key_row = &key[key_offset..key_offset + head_dim];
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

            let probabilities = softmax(&scores[..width])?;
            let output_row = &mut output[query_offset..query_offset + head_dim];
            for (column, probability) in probabilities.iter().enumerate() {
                if *probability == 0.0 {
                    continue;
                }
                let offset = start + column;
                let value_offset = (offset * num_key_value_heads + key_head) * head_dim;
                let value_row = &value[value_offset..value_offset + head_dim];
                for (destination, value_element) in output_row.iter_mut().zip(value_row) {
                    *destination += probability * value_element;
                }
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

    #[test]
    fn linear_projects_rows_against_weight_rows() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weight = [1.0, 0.0, -1.0, 0.0, 1.0, 1.0];
        let output = linear(&input, &weight, 2, 3, 2).expect("linear");
        assert_eq!(output, vec![-2.0, 5.0, -2.0, 11.0]);

        let mut reused = [99.0; 4];
        linear_into(&input, &weight, 2, 3, 2, &mut reused).expect("linear into");
        assert_eq!(reused, [-2.0, 5.0, -2.0, 11.0]);
    }

    #[test]
    fn linear_rejects_shape_mismatches() {
        assert!(matches!(
            linear(&[1.0], &[1.0], 1, 1, 2).expect_err("output mismatch must fail"),
            Error::Internal(_)
        ));
        assert!(matches!(
            linear(&[1.0], &[1.0, 2.0], 1, 2, 1).expect_err("input mismatch must fail"),
            Error::Internal(_)
        ));
        assert!(matches!(
            linear(&[1.0], &[1.0], 1, 1, 1)
                .and_then(|_| linear(&[], &[], 1, 1, 1))
                .expect_err("empty input mismatch must fail"),
            Error::Internal(_)
        ));
    }

    #[test]
    fn rms_norm_rows_normalizes_each_row() {
        let output =
            rms_norm_rows(&[3.0, 4.0, 0.0, 2.0], &[1.0, 1.0], 2, 2, 0.0).expect("RMSNorm rows");
        assert_close(
            &output,
            &[0.848_528, 1.131_371, 0.0, std::f32::consts::SQRT_2],
            1e-5,
        );

        let unscaled =
            rms_norm_rows_unscaled(&[3.0, 4.0], 1, 2, 0.0).expect("unscaled RMSNorm rows");
        assert_close(&unscaled, &[0.848_528, 1.131_371], 1e-5);
    }

    #[test]
    fn rms_norm_rows_rejects_invalid_inputs() {
        assert!(matches!(
            rms_norm_rows(&[1.0], &[1.0, 1.0], 1, 2, 0.0).expect_err("input mismatch"),
            Error::Internal(_)
        ));
        assert!(matches!(
            rms_norm_rows(&[1.0], &[1.0], 1, 0, 0.0).expect_err("zero width"),
            Error::Internal(_)
        ));
        assert!(matches!(
            rms_norm_rows(&[1.0], &[1.0], 1, 1, -1.0).expect_err("negative epsilon"),
            Error::Internal(_)
        ));
        assert!(matches!(
            rms_norm_rows(&[f32::NAN], &[1.0], 1, 1, 0.0).expect_err("NaN input"),
            Error::Internal(_)
        ));
    }

    #[test]
    fn mean_pool_averages_rows() {
        let pooled = mean_pool(&[1.0, 2.0, 3.0, 4.0], 2, 2).expect("mean pool");
        assert_eq!(pooled, vec![2.0, 3.0]);
        assert!(matches!(
            mean_pool(&[], 0, 2).expect_err("empty sequence must fail"),
            Error::Internal(_)
        ));
        assert!(matches!(
            mean_pool(&[1.0], 1, 2).expect_err("length mismatch must fail"),
            Error::Internal(_)
        ));
    }

    #[test]
    fn truncation_renormalizes_the_slice() {
        let truncated = truncate_and_normalize(&[3.0, 4.0, 12.0], 2).expect("truncate");
        assert_close(&truncated, &[0.6, 0.8], 1e-7);
        assert!(matches!(
            truncate_and_normalize(&[1.0], 2).expect_err("dimension too large"),
            Error::Internal(_)
        ));
        assert!(matches!(
            truncate_and_normalize(&[1.0], 0).expect_err("zero dimension"),
            Error::Internal(_)
        ));
    }

    #[test]
    fn rope_with_inv_freq_matches_the_reference_rotation() {
        let inv_freq = rope_inv_freq(2, 10_000.0).expect("inverse frequencies");
        assert_eq!(inv_freq, vec![1.0]);
        let mut values = [1.0, 0.0];
        rope_with_inv_freq(&mut values, 1, 1, 2, &inv_freq, 1).expect("RoPE");
        assert_close(&values, &[0.540_302, 0.841_471], 1e-5);
        assert!(matches!(
            rope_with_inv_freq(&mut [1.0, 2.0], 1, 1, 2, &[1.0, 2.0], 0)
                .expect_err("frequency mismatch"),
            Error::Internal(_)
        ));
        assert!(matches!(
            rope_inv_freq(1, 10_000.0).expect_err("odd head dimension"),
            Error::Internal(_)
        ));
    }

    fn attention_shape(
        seq_len: usize,
        num_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        sliding_window: Option<usize>,
    ) -> AttentionShape {
        AttentionShape {
            seq_len,
            num_heads,
            num_key_value_heads,
            head_dim,
            sliding_window,
            scale: 1.0,
        }
    }

    #[test]
    fn bidirectional_attention_matches_full_attention() {
        let query = [1.0, 1.0];
        let key = [1.0, 0.0];
        let value = [10.0, 20.0];
        let output =
            bidirectional_attention(&query, &key, &value, attention_shape(2, 1, 1, 1, None))
                .expect("bidirectional attention");
        assert_close(&output, &[12.689_4, 12.689_4], 1e-4);
    }

    #[test]
    fn sliding_window_restricts_each_position_to_its_neighbourhood() {
        let query = [1.0, 1.0, 1.0];
        let key = [1.0, 0.0, 0.0];
        let value = [10.0, 20.0, 30.0];
        let output =
            bidirectional_attention(&query, &key, &value, attention_shape(3, 1, 1, 1, Some(0)))
                .expect("windowed attention");
        assert_eq!(output, vec![10.0, 20.0, 30.0]);

        let output =
            bidirectional_attention(&query, &key, &value, attention_shape(3, 1, 1, 1, Some(1)))
                .expect("windowed attention");
        assert_close(&output, &[12.689_4, 16.358_2, 25.0], 1e-4);
    }

    #[test]
    fn grouped_query_attention_repeats_key_value_heads_in_order() {
        // Four query heads, two key/value heads, one position. Heads 0 and 1
        // must read key/value head 0; heads 2 and 3 must read head 1.
        let query = [1.0, 1.0, 0.0, 0.0];
        let key = [1.0, 0.0];
        let value = [5.0, 7.0];
        let output =
            bidirectional_attention(&query, &key, &value, attention_shape(1, 4, 2, 1, None))
                .expect("grouped attention");
        assert_close(&output, &[5.0, 5.0, 7.0, 7.0], 1e-6);
    }

    #[test]
    fn bidirectional_attention_rejects_invalid_shapes() {
        assert!(matches!(
            bidirectional_attention(&[1.0], &[1.0], &[1.0], attention_shape(1, 3, 2, 1, None))
                .expect_err("head count mismatch must fail"),
            Error::Internal(_)
        ));
        assert!(matches!(
            bidirectional_attention(&[1.0], &[1.0], &[1.0], attention_shape(1, 1, 1, 0, None))
                .expect_err("zero head dimension must fail"),
            Error::Internal(_)
        ));
        assert!(matches!(
            bidirectional_attention(
                &[1.0],
                &[1.0],
                &[1.0],
                AttentionShape {
                    scale: f32::NAN,
                    ..attention_shape(1, 1, 1, 1, None)
                }
            )
            .expect_err("NaN scale must fail"),
            Error::Internal(_)
        ));
    }
}
