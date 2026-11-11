//! FP32 tensors and borrowed views.
//!
//! Internal tensors are always FP32. BF16 or F32 file data is converted at
//! load time, never kept in a second precision on the compute path. The
//! element cap is explicit: a tensor may not exceed 2^30 elements, which is
//! 4 GiB of FP32 data.

use crate::error::{Error, Result};
use crate::tensor::shape::Shape;

/// Maximum number of elements in one tensor.
///
/// 2^30 FP32 elements occupy 4 GiB. This cap is a hard guard against a
/// hostile header requesting an unbounded allocation; model shards are loaded
/// tensor by tensor.
pub const MAX_TENSOR_ELEMENTS: usize = 1 << 30;

/// Maximum number of bytes in one FP32 tensor.
pub const MAX_TENSOR_BYTES: usize = MAX_TENSOR_ELEMENTS * 4;

/// Owned FP32 tensor with a validated shape.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    shape: Shape,
    data: Vec<f32>,
}

impl Tensor {
    /// Builds a tensor from a shape and owned data.
    ///
    /// # Errors
    ///
    /// Returns Error::Internal when the data length does not match the shape,
    /// and Error::LimitExceeded when the element cap is exceeded.
    pub fn new(shape: Shape, data: Vec<f32>) -> Result<Self> {
        if shape.numel() != data.len() {
            return Err(Error::internal(format!(
                "tensor shape {shape} expects {} elements, got {}",
                shape.numel(),
                data.len()
            )));
        }
        if data.len() > MAX_TENSOR_ELEMENTS {
            return Err(Error::limit_exceeded(format!(
                "tensor has {} elements, above the {MAX_TENSOR_ELEMENTS}-element cap",
                data.len()
            )));
        }
        Ok(Self { shape, data })
    }

    /// Builds a zero-filled tensor.
    pub fn zeros(shape: Shape) -> Result<Self> {
        if shape.numel() > MAX_TENSOR_ELEMENTS {
            return Err(Error::limit_exceeded(format!(
                "tensor has {} elements, above the {MAX_TENSOR_ELEMENTS}-element cap",
                shape.numel()
            )));
        }
        let data = vec![0.0f32; shape.numel()];
        Self::new(shape, data)
    }

    /// Builds a tensor filled with a constant value.
    pub fn filled(shape: Shape, value: f32) -> Result<Self> {
        if shape.numel() > MAX_TENSOR_ELEMENTS {
            return Err(Error::limit_exceeded(format!(
                "tensor has {} elements, above the {MAX_TENSOR_ELEMENTS}-element cap",
                shape.numel()
            )));
        }
        let data = vec![value; shape.numel()];
        Self::new(shape, data)
    }

    /// Copies a slice into a new tensor.
    pub fn from_slice(shape: Shape, data: &[f32]) -> Result<Self> {
        Self::new(shape, data.to_vec())
    }

    /// Shape of the tensor.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Borrowed element data.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Mutable element data.
    ///
    /// The shape is not changed, so the length invariant is preserved.
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Decomposes the tensor into its shape and data.
    pub fn into_parts(self) -> (Shape, Vec<f32>) {
        (self.shape, self.data)
    }

    /// Borrows the tensor as a view.
    pub fn view(&self) -> TensorView<'_> {
        TensorView {
            shape: &self.shape,
            data: &self.data,
        }
    }

    /// Returns true when every element is finite.
    pub fn is_finite(&self) -> bool {
        self.data.iter().all(|value| value.is_finite())
    }

    /// Validates that every element is finite.
    ///
    /// # Errors
    ///
    /// Returns Error::Internal with the first offending index.
    pub fn validate_finite(&self) -> Result<()> {
        for (index, value) in self.data.iter().enumerate() {
            if !value.is_finite() {
                return Err(Error::internal(format!(
                    "non-finite tensor value {value} at index {index}"
                )));
            }
        }
        Ok(())
    }
}

/// Borrowed FP32 tensor.
#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    shape: &'a Shape,
    data: &'a [f32],
}

impl<'a> TensorView<'a> {
    /// Builds a view, checking that the data length matches the shape.
    ///
    /// # Errors
    ///
    /// Returns Error::Internal when the length does not match the shape.
    pub fn new(shape: &'a Shape, data: &'a [f32]) -> Result<Self> {
        if shape.numel() != data.len() {
            return Err(Error::internal(format!(
                "view shape {shape} expects {} elements, got {}",
                shape.numel(),
                data.len()
            )));
        }
        Ok(Self { shape, data })
    }

    /// Shape of the view.
    pub fn shape(&self) -> &Shape {
        self.shape
    }

    /// Borrowed element data.
    pub fn data(&self) -> &'a [f32] {
        self.data
    }

    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Returns true when every element is finite.
    pub fn is_finite(&self) -> bool {
        self.data.iter().all(|value| value.is_finite())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_a_tensor_with_matching_shape() {
        let shape = Shape::new(vec![2, 2]).expect("valid shape");
        let tensor = Tensor::new(shape.clone(), vec![1.0, 2.0, 3.0, 4.0]).expect("valid tensor");
        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.numel(), 4);
        assert_eq!(tensor.data(), &[1.0, 2.0, 3.0, 4.0]);
        assert!(tensor.is_finite());
    }

    #[test]
    fn rejects_length_mismatch() {
        let shape = Shape::new(vec![2, 2]).expect("valid shape");
        let error = Tensor::new(shape, vec![1.0]).expect_err("mismatch must fail");
        assert!(matches!(error, Error::Internal(_)));
    }

    #[test]
    fn zeros_and_filled_respect_the_shape() {
        let shape = Shape::new(vec![2, 3]).expect("valid shape");
        let zeros = Tensor::zeros(shape.clone()).expect("zeros");
        assert_eq!(zeros.data(), &[0.0; 6]);
        let filled = Tensor::filled(shape, 2.5).expect("filled");
        assert_eq!(filled.data(), &[2.5; 6]);
    }

    #[test]
    fn view_checks_length_and_borrows() {
        let shape = Shape::new(vec![3]).expect("valid shape");
        let data = [1.0, 2.0, 3.0];
        let view = TensorView::new(&shape, &data).expect("valid view");
        assert_eq!(view.shape(), &shape);
        assert_eq!(view.numel(), 3);
        assert_eq!(view.data(), &data);
        assert!(view.is_finite());

        let error = TensorView::new(&shape, &data[..2]).expect_err("short view must fail");
        assert!(matches!(error, Error::Internal(_)));
    }

    #[test]
    fn detects_non_finite_values() {
        let shape = Shape::new(vec![3]).expect("valid shape");
        let tensor = Tensor::new(shape, vec![1.0, f32::NAN, 3.0]).expect("valid tensor");
        assert!(!tensor.is_finite());
        let error = tensor.validate_finite().expect_err("NaN must be rejected");
        assert!(matches!(error, Error::Internal(_)));
        assert!(error.to_string().contains("index 1"));
    }

    #[test]
    fn scalar_tensor_has_one_element() {
        let tensor = Tensor::new(Shape::scalar(), vec![42.0]).expect("scalar");
        assert_eq!(tensor.numel(), 1);
        assert!(tensor.shape().is_scalar());
    }
}
