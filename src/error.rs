//! Central error type for Seldon Vault Lite.
//!
//! All product errors converge here so that the CLI, then the HTTP API, can
//! choose an exit code or status without inspecting ad hoc errors. The
//! corrupted-input and limit-exceeded variants will be added with the
//! milestone M1 parsers.

use std::error::Error as StdError;
use std::fmt;
use std::io;

/// Internal crate result.
pub type Result<T> = std::result::Result<T, Error>;

/// Central product error.
///
/// The variants distinguish a usage error (exit code 2) from an execution
/// error (exit code 1). The taxonomy will be extended with corrupted-input
/// and limit-exceeded errors when the corresponding parsers exist.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Invalid command or argument provided by the user.
    Usage(String),
    /// Input/output failure.
    Io(io::Error),
}

impl Error {
    /// Builds a usage error.
    pub fn usage(message: impl Into<String>) -> Self {
        Self::Usage(message.into())
    }

    /// Process exit code associated with this error.
    ///
    /// `0` is reserved for success, `1` for execution errors, and `2` for
    /// usage errors.
    pub fn exit_code(&self) -> u8 {
        match self {
            Self::Usage(_) => 2,
            Self::Io(_) => 1,
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
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Usage(_) => None,
            Self::Io(error) => Some(error),
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
    fn broken_pipe_is_detected() {
        let error = Error::from(io::Error::from(io::ErrorKind::BrokenPipe));
        assert!(error.is_broken_pipe());
    }
}
