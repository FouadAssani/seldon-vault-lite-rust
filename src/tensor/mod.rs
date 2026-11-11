//! Tensor primitives and FP32 kernels.
//!
//! The tensor module owns shapes, dtypes, owned tensors, borrowed views, the
//! deterministic scalar kernels used as the numerical reference, and the NEON
//! execution path for the macOS Apple Silicon target.

pub mod dtype;
pub mod kernels;
#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;
pub mod owned;
pub mod shape;

pub use dtype::DType;
pub use owned::{MAX_TENSOR_BYTES, MAX_TENSOR_ELEMENTS, Tensor, TensorView};
pub use shape::{MAX_DIMENSION, MAX_RANK, Shape};
