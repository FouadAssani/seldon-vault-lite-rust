//! `seldon-vault-lite` binary.
//!
//! `main` only translates the CLI result into an exit code. All parsing and
//! execution logic lives in the library so it can be tested without launching
//! a process.

use std::io::Write;
use std::process::ExitCode;

use seldon_vault_lite::cli;

fn main() -> ExitCode {
    let mut stdout = std::io::stdout().lock();
    let mut stderr = std::io::stderr().lock();

    match cli::main_entry(std::env::args_os(), &mut stdout) {
        Ok(()) => ExitCode::SUCCESS,
        // A closed pipe is normal Unix tool behavior, not a failure.
        Err(error) if error.is_broken_pipe() => ExitCode::SUCCESS,
        Err(error) => {
            // Writing the diagnostic is best-effort: if stderr is also closed,
            // the exit code remains the only available information.
            let _ = writeln!(stderr, "error: {error}");
            let _ = stderr.flush();
            ExitCode::from(error.exit_code())
        }
    }
}
