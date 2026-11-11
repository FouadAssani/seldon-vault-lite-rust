//! Central error type for Seldon Vault Lite.
//!
//! All product errors converge here so that the CLI, then the HTTP API, can
//! choose an exit code or status without inspecting ad hoc errors. The
//! variants distinguish a user error, corrupted input, an exceeded limit, an
//! unsupported feature, and an internal invariant violation.

use std::error::Error as StdError;
use std::fmt;
use std::io;

/// Internal crate result.
pub type Result<T> = std::result::Result<T, Error>;

/// Central product error.
///
/// The taxonomy is deliberately small and stable. Parsers return
/// Error::Corrupt for malformed input, Error::LimitExceeded when a
/// documented cap is reached, and Error::Unsupported for a valid but
/// unimplemented format or feature. Error::Internal signals a violated
/// invariant in product code, never hostile input.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Invalid command or argument provided by the user.
    Usage(String),
    /// Input/output failure.
    Io(io::Error),
    /// Malformed or truncated external input.
    Corrupt(String),
    /// A documented resource or format limit was exceeded.
    LimitExceeded(String),
    /// A valid but unsupported format, dtype, or feature.
    Unsupported(String),
    /// An internal invariant was violated; this is a product bug.
    Internal(String),
}

impl Error {
    /// Builds a usage error.
    pub fn usage(message: impl Into<String>) -> Self {
        Self::Usage(message.into())
    }

    /// Builds a corrupted-input error.
    pub fn corrupt(message: impl Into<String>) -> Self {
        Self::Corrupt(message.into())
    }

    /// Builds a limit-exceeded error.
    pub fn limit_exceeded(message: impl Into<String>) -> Self {
        Self::LimitExceeded(message.into())
    }

    /// Builds an unsupported-feature error.
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::Unsupported(message.into())
    }

    /// Builds an internal-invariant error.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
    }

    /// Process exit code associated with this error.
    ///
    /// 0 is reserved for success, 1 for execution errors, and 2 for
    /// usage errors.
    pub fn exit_code(&self) -> u8 {
        match self {
            Self::Usage(_) => 2,
            Self::Io(_)
            | Self::Corrupt(_)
            | Self::LimitExceeded(_)
            | Self::Unsupported(_)
            | Self::Internal(_) => 1,
        }
    }

    /// Indicates whether the error comes from a pipe closed by the calling process.
    ///
    /// A closed pipe is not a program failure: the CLI can then finish with
    /// the success code, as Unix tools do.
    pub fn is_broken_pipe(&self) -> bool {
        matches!(self, Self::Io(error) if error.kind() == io::ErrorKind::BrokenPipe)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage(message) => formatter.write_str(message),
            Self::Io(error) => write!(formatter, "{error}"),
            Self::Corrupt(message) => write!(formatter, "corrupt input: {message}"),
            Self::LimitExceeded(message) => write!(formatter, "limit exceeded: {message}"),
            Self::Unsupported(message) => write!(formatter, "unsupported: {message}"),
            Self::Internal(message) => write!(formatter, "internal error: {message}"),
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Usage(_)
            | Self::Corrupt(_)
            | Self::LimitExceeded(_)
            | Self::Unsupported(_)
            | Self::Internal(_) => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_errors_exit_with_code_two() {
        let error = Error::usage("invalid command");
        assert!(matches!(error, Error::Usage(_)));
        assert_eq!(error.exit_code(), 2);
        assert!(!error.is_broken_pipe());
        assert_eq!(error.to_string(), "invalid command");
        assert!(StdError::source(&error).is_none());
    }

    #[test]
    fn io_errors_exit_with_code_one() {
        let error = Error::from(io::Error::other("failure"));
        assert_eq!(error.exit_code(), 1);
        assert!(!error.is_broken_pipe());
        assert_eq!(
            StdError::source(&error)
                .expect("an I/O error must expose its source")
                .to_string(),
            "failure"
        );
    }

    #[test]
    fn execution_errors_exit_with_code_one() {
        for error in [
            Error::corrupt("truncated"),
            Error::limit_exceeded("too large"),
            Error::unsupported("F16"),
            Error::internal("broken invariant"),
        ] {
            assert_eq!(error.exit_code(), 1);
            assert!(!error.is_broken_pipe());
            assert!(StdError::source(&error).is_none());
        }
    }

    #[test]
    fn corrupt_and_limit_messages_are_prefixed() {
        assert_eq!(
            Error::corrupt("bad offset").to_string(),
            "corrupt input: bad offset"
        );
        assert_eq!(
            Error::limit_exceeded("too many tensors").to_string(),
            "limit exceeded: too many tensors"
        );
        assert_eq!(
            Error::unsupported("F16 activations").to_string(),
            "unsupported: F16 activations"
        );
        assert_eq!(
            Error::internal("shape mismatch").to_string(),
            "internal error: shape mismatch"
        );
    }

    #[test]
    fn broken_pipe_is_detected() {
        let error = Error::from(io::Error::from(io::ErrorKind::BrokenPipe));
        assert!(error.is_broken_pipe());
    }
}
