//! Tensor shapes with checked element counts and row-major strides.
//!
//! A Shape validates its rank, dimensions, and total element count at
//! construction. Once constructed, numel is exact and cannot overflow, so the
//! compute kernels can trust it without repeating the checks.

use std::fmt;

use crate::error::{Error, Result};

/// Maximum number of dimensions accepted by the engine.
///
/// The reference model uses at most four dimensions. Eight leaves room for
/// packed or reshaped tensors without allowing a hostile header to force a
/// pathological rank.
pub const MAX_RANK: usize = 8;

/// Maximum value of a single dimension.
pub const MAX_DIMENSION: usize = u32::MAX as usize;

/// Validated tensor shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<usize>,
    elements: usize,
}

impl Shape {
    /// Builds a validated shape.
    ///
    /// # Errors
    ///
    /// Returns Error::LimitExceeded when the rank exceeds MAX_RANK, a
    /// dimension exceeds MAX_DIMENSION, or the element count overflows.
    pub fn new(dims: impl Into<Vec<usize>>) -> Result<Self> {
        let dims = dims.into();
        if dims.len() > MAX_RANK {
            return Err(Error::limit_exceeded(format!(
                "tensor rank {} exceeds the cap of {MAX_RANK}",
                dims.len()
            )));
        }

        let mut elements: usize = 1;
        for &dim in &dims {
            if dim > MAX_DIMENSION {
                return Err(Error::limit_exceeded(format!(
                    "tensor dimension {dim} exceeds the cap of {MAX_DIMENSION}"
                )));
            }
            elements = elements.checked_mul(dim).ok_or_else(|| {
                Error::limit_exceeded("tensor element count overflows the address space")
            })?;
        }

        Ok(Self { dims, elements })
    }

    /// Builds a scalar shape (rank zero, one element).
    pub fn scalar() -> Self {
        Self {
            dims: Vec::new(),
            elements: 1,
        }
    }

    /// Dimensions in row-major order.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Number of dimensions.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.elements
    }

    /// Returns true for a rank-zero shape.
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }

    /// Returns one dimension, or None when the index is out of range.
    pub fn dim(&self, index: usize) -> Option<usize> {
        self.dims.get(index).copied()
    }

    /// Row-major strides, in elements.
    pub fn strides(&self) -> Vec<usize> {
        let mut strides = vec![0usize; self.dims.len()];
        let mut stride: usize = 1;
        for index in (0..self.dims.len()).rev() {
            strides[index] = stride;
            stride = stride.saturating_mul(self.dims[index]);
        }
        strides
    }

    /// Builds a new shape with the same element count.
    ///
    /// # Errors
    ///
    /// Returns Error::Internal when the new shape has a different element
    /// count, and the usual shape errors for an invalid new shape.
    pub fn reshape(&self, dims: impl Into<Vec<usize>>) -> Result<Self> {
        let reshaped = Self::new(dims)?;
        if reshaped.numel() != self.numel() {
            return Err(Error::internal(format!(
                "cannot reshape {} elements into {} elements",
                self.numel(),
                reshaped.numel()
            )));
        }
        Ok(reshaped)
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("[")?;
        for (index, dim) in self.dims.iter().enumerate() {
            if index > 0 {
                formatter.write_str(", ")?;
            }
            write!(formatter, "{dim}")?;
        }
        formatter.write_str("]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_shape_has_one_element_and_no_dimension() {
        let shape = Shape::scalar();
        assert!(shape.is_scalar());
        assert_eq!(shape.rank(), 0);
        assert_eq!(shape.numel(), 1);
        assert!(shape.dims().is_empty());
        assert!(shape.strides().is_empty());
        assert_eq!(shape.to_string(), "[]");
    }

    #[test]
    fn computes_element_count_and_strides() {
        let shape = Shape::new(vec![2, 3, 4]).expect("valid shape");
        assert_eq!(shape.rank(), 3);
        assert_eq!(shape.numel(), 24);
        assert_eq!(shape.dims(), &[2, 3, 4]);
        assert_eq!(shape.strides(), vec![12, 4, 1]);
        assert_eq!(shape.dim(1), Some(3));
        assert_eq!(shape.dim(3), None);
        assert_eq!(shape.to_string(), "[2, 3, 4]");
    }

    #[test]
    fn zero_dimension_yields_zero_elements() {
        let shape = Shape::new(vec![0, 3]).expect("zero-sized shape");
        assert_eq!(shape.numel(), 0);
        assert_eq!(shape.strides(), vec![3, 1]);
    }

    #[test]
    fn rejects_rank_above_the_cap() {
        let dims = vec![1usize; MAX_RANK + 1];
        let error = Shape::new(dims).expect_err("rank cap must be enforced");
        assert!(matches!(error, Error::LimitExceeded(_)));
    }

    #[test]
    fn rejects_dimensions_above_the_cap() {
        let error =
            Shape::new(vec![MAX_DIMENSION + 1]).expect_err("dimension cap must be enforced");
        assert!(matches!(error, Error::LimitExceeded(_)));
    }

    #[test]
    fn rejects_element_count_overflow() {
        let error = Shape::new(vec![usize::MAX, 2]).expect_err("overflow must be rejected");
        assert!(matches!(error, Error::LimitExceeded(_)));
    }

    #[test]
    fn reshape_requires_the_same_element_count() {
        let shape = Shape::new(vec![2, 3]).expect("valid shape");
        let reshaped = shape.reshape(vec![3, 2]).expect("valid reshape");
        assert_eq!(reshaped.dims(), &[3, 2]);
        let error = shape.reshape(vec![2, 2]).expect_err("mismatch must fail");
        assert!(matches!(error, Error::Internal(_)));
    }
}
