//! CLI integration tests for milestone M0.
//!
//! They run the real binary to verify the observable contract: standard
//! output, standard error, and exit codes.

use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_seldon-vault-lite");

fn run(args: &[&str]) -> Output {
    Command::new(BIN)
        .args(args)
        .output()
        .expect("the test binary must be launchable")
}

fn stdout(output: &Output) -> String {
    String::from_utf8(output.stdout.clone()).expect("stdout must be UTF-8")
}

fn stderr(output: &Output) -> String {
    String::from_utf8(output.stderr.clone()).expect("stderr must be UTF-8")
}

fn expected_version() -> String {
    format!(
        "seldon-vault-lite {} ({})\n",
        env!("CARGO_PKG_VERSION"),
        seldon_vault_lite::TARGET_TRIPLE
    )
}

#[test]
fn version_prints_name_version_and_target() {
    let output = run(&["version"]);
    assert!(output.status.success());
    assert_eq!(stdout(&output), expected_version());
    assert!(output.stderr.is_empty());
}

#[test]
fn version_aliases_are_equivalent() {
    for args in [&["version"][..], &["--version"], &["-V"]] {
        let output = run(args);
        assert!(output.status.success(), "failed for {args:?}");
        assert_eq!(
            stdout(&output),
            expected_version(),
            "output differs for {args:?}"
        );
        assert!(output.stderr.is_empty());
    }
}

#[test]
fn help_explains_usage_and_exit_codes() {
    for args in [&["help"][..], &["--help"], &["-h"]] {
        let output = run(args);
        assert!(output.status.success(), "failed for {args:?}");
        let text = stdout(&output);
        assert!(text.contains("USAGE:"), "help without USAGE for {args:?}");
        assert!(
            text.contains("EXIT CODES:"),
            "help without codes for {args:?}"
        );
        assert!(text.contains("version"));
        assert!(output.stderr.is_empty());
    }
}

#[test]
fn missing_command_is_a_usage_error() {
    let output = run(&[]);
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    let text = stderr(&output);
    assert!(text.contains("error:"));
    assert!(text.contains("no command"));
}

#[test]
fn unknown_command_is_a_usage_error() {
    // `serve` is planned but not yet implemented: the CLI must
    // explicitly refuse instead of simulating a service.
    let output = run(&["serve"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(stderr(&output).contains("unknown command"));
}

#[test]
fn extra_arguments_are_rejected() {
    let output = run(&["version", "extra"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(stderr(&output).contains("extra arguments"));
}
