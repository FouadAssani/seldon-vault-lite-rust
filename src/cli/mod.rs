//! Argument parsing and command execution for the CLI.
//!
//! The CLI is deliberately minimal at milestone M0: `version` and `help`.
//! The `serve`, `embed`, `index`, `search`, and `verify` commands will be
//! added with their functionality in later milestones. No stub simulates
//! nonexistent behavior.

use std::ffi::OsString;
use std::io::Write;

use crate::error::{Error, Result};
use crate::{PROGRAM_NAME, TARGET_TRIPLE, VERSION};

/// Help text displayed by the `help` command.
const HELP: &str = concat!(
    env!("CARGO_PKG_NAME"),
    r#" — local multimodal embedding and search engine

USAGE:
    "#,
    env!("CARGO_PKG_NAME"),
    r#" <COMMAND>

COMMANDS:
    version    Prints the version and build target
    help       Prints this help

OPTIONS:
    -V, --version    Equivalent to `version`
    -h, --help       Equivalent to `help`

EXIT CODES:
    0    success
    1    execution error
    2    usage error

The `serve`, `embed`, `index`, `search`, and `verify` commands will be added
with milestones M1 to M6. No command is simulated in the meantime.
"#
);

/// Command recognized by the CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Command {
    /// Prints the program version and build target.
    Version,
    /// Prints help and exit codes.
    Help,
}

/// Parses the arguments following the program name.
///
/// Non-UTF-8 arguments are rejected with a usage error rather than causing a
/// panic: the CLI must never abort on hostile input.
pub fn parse<I, S>(args: I) -> Result<Command>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    let mut args = args.into_iter();
    let Some(first) = args.next() else {
        return Err(Error::usage(
            "no command provided; use `--help` to display help",
        ));
    };
    let first: OsString = first.into();
    let first = first
        .to_str()
        .ok_or_else(|| Error::usage("non-UTF-8 argument"))?;

    let command = match first {
        "version" | "--version" | "-V" => Command::Version,
        "help" | "--help" | "-h" => Command::Help,
        other => {
            return Err(Error::usage(format!(
                "unknown command `{other}`; use `--help`"
            )));
        }
    };

    if args.next().is_some() {
        return Err(Error::usage(format!("extra arguments after `{first}`")));
    }

    Ok(command)
}

/// Executes a command and writes its output to `stdout`.
///
/// # Errors
///
/// Returns an I/O error if writing fails, particularly when standard output
/// is closed by the calling process.
pub fn run<W: Write>(command: Command, stdout: &mut W) -> Result<()> {
    match command {
        Command::Version => writeln!(stdout, "{PROGRAM_NAME} {VERSION} ({TARGET_TRIPLE})")?,
        Command::Help => stdout.write_all(HELP.as_bytes())?,
    }
    stdout.flush()?;
    Ok(())
}

/// CLI entry point from the process's raw arguments.
///
/// The first argument is the program name, as with
/// [`std::env::args_os`]; it is ignored. The rest is parsed by
/// [`parse`] then executed by [`run`].
pub fn main_entry<I, S>(args: I, stdout: &mut impl Write) -> Result<()>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    let mut args = args.into_iter();
    let _program_name = args.next();
    let command = parse(args)?;
    run(command, stdout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;

    fn parse_args(args: &[&str]) -> Result<Command> {
        parse(args.iter().copied())
    }

    #[test]
    fn accepts_version_aliases() {
        for args in [&["version"][..], &["--version"], &["-V"]] {
            assert_eq!(
                parse_args(args).expect("version must be accepted"),
                Command::Version
            );
        }
    }

    #[test]
    fn accepts_help_aliases() {
        for args in [&["help"][..], &["--help"], &["-h"]] {
            assert_eq!(
                parse_args(args).expect("help must be accepted"),
                Command::Help
            );
        }
    }

    #[test]
    fn rejects_missing_command() {
        let error = parse_args(&[]).expect_err("a missing command must fail");
        assert!(matches!(error, Error::Usage(_)));
        assert_eq!(error.exit_code(), 2);
        assert!(error.to_string().contains("no command"));
    }

    #[test]
    fn rejects_unknown_command() {
        let error = parse_args(&["serve"]).expect_err("serve does not exist yet");
        assert!(matches!(error, Error::Usage(_)));
        assert!(error.to_string().contains("unknown command"));
    }

    #[test]
    fn rejects_extra_arguments() {
        let error = parse_args(&["version", "extra"]).expect_err("extra arguments must fail");
        assert!(matches!(error, Error::Usage(_)));
        assert!(error.to_string().contains("extra arguments"));
    }

    #[test]
    fn rejects_non_utf8_argument() {
        let args = [OsStr::from_bytes(b"\xff")];
        let error = parse(args.iter().copied()).expect_err("a non-UTF-8 argument must fail");
        assert!(matches!(error, Error::Usage(_)));
        assert!(error.to_string().contains("non-UTF-8"));
    }

    #[test]
    fn version_output_is_stable() {
        let mut output = Vec::new();
        run(Command::Version, &mut output).expect("writing must succeed");
        assert_eq!(
            String::from_utf8(output).expect("output must be UTF-8"),
            format!("{PROGRAM_NAME} {VERSION} ({TARGET_TRIPLE})\n")
        );
    }

    #[test]
    fn help_contains_usage_and_exit_codes() {
        let mut output = Vec::new();
        run(Command::Help, &mut output).expect("writing must succeed");
        let text = String::from_utf8(output).expect("output must be UTF-8");
        assert!(text.contains("USAGE:"));
        assert!(text.contains("EXIT CODES:"));
        assert!(text.contains("version"));
    }

    #[test]
    fn main_entry_ignores_program_name() {
        let mut output = Vec::new();
        main_entry(["seldon-vault-lite", "version"], &mut output)
            .expect("the command must succeed");
        assert_eq!(
            String::from_utf8(output).expect("output must be UTF-8"),
            format!("{PROGRAM_NAME} {VERSION} ({TARGET_TRIPLE})\n")
        );
    }
}
