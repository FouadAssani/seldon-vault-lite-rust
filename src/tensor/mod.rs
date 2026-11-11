//! Tensor primitives and reference FP32 kernels.
//!
//! The tensor module owns shapes, dtypes, owned tensors, borrowed views, and
//! the deterministic scalar kernels used as the numerical reference.

pub mod dtype;
pub mod kernels;
pub mod owned;
pub mod shape;

pub use dtype::DType;
pub use owned::{MAX_TENSOR_BYTES, MAX_TENSOR_ELEMENTS, Tensor, TensorView};
pub use shape::{MAX_DIMENSION, MAX_RANK, Shape};
