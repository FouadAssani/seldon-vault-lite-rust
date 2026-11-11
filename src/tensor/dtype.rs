//! Tensor data types accepted by the engine.
//!
//! BF16 weights are accepted because the reference model publishes them.
//! FP16 is deliberately rejected: the model contract forbids FP16 for
//! activations, and accepting it silently would invite a non-compliant path.

use crate::error::{Error, Result};

/// Data type of a tensor as stored in a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// 32-bit IEEE-754 float.
    F32,
    /// Brain floating point, 16 bits.
    Bf16,
}

impl DType {
    /// Canonical name used by Safetensors.
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::Bf16 => "BF16",
        }
    }

    /// Size of one element in bytes.
    pub const fn size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::Bf16 => 2,
        }
    }

    /// Parses a Safetensors dtype name.
    ///
    /// # Errors
    ///
    /// Returns Error::Unsupported for FP16 and for any dtype the engine does
    /// not implement. This is not a corruption: the file is valid, the engine
    /// simply refuses a precision or format it cannot honour.
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "F32" => Ok(Self::F32),
            "BF16" => Ok(Self::Bf16),
            "F16" => Err(Error::unsupported(
                "F16 tensors are not supported: FP16 is forbidden for activations",
            )),
            other => Err(Error::unsupported(format!("tensor dtype {other}"))),
        }
    }

    /// Returns true when the dtype is a floating-point format.
    pub const fn is_float(self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_sizes_are_stable() {
        assert_eq!(DType::F32.name(), "F32");
        assert_eq!(DType::F32.size(), 4);
        assert_eq!(DType::Bf16.name(), "BF16");
        assert_eq!(DType::Bf16.size(), 2);
    }

    #[test]
    fn parses_supported_dtypes() {
        assert_eq!(DType::parse("F32").expect("F32"), DType::F32);
        assert_eq!(DType::parse("BF16").expect("BF16"), DType::Bf16);
    }

    #[test]
    fn rejects_fp16_and_unknown_dtypes() {
        for name in ["F16", "F64", "I8", "U32"] {
            let error = DType::parse(name).expect_err("dtype must be rejected");
            assert!(matches!(error, Error::Unsupported(_)), "dtype {name}");
        }
    }
}
