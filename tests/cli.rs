//! CLI integration tests for milestones M0 to M2.
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
#[test]
fn embed_ids_prints_a_json_vector() {
    let output = run(&[
        "embed",
        "--model-dir",
        "fixtures/model/tiny_text",
        "--ids",
        "1,5,7,9,2",
        "--dim",
        "8",
    ]);
    assert!(output.status.success(), "stderr: {}", stderr(&output));
    assert!(output.stderr.is_empty());
    let text = stdout(&output);
    let value = seldon_vault_lite::json::parse(text.trim()).expect("stdout must be JSON");
    let array = value.as_array().expect("output must be an array");
    assert_eq!(array.len(), 8);
    let norm: f64 = array
        .iter()
        .map(|value| {
            let value = value.as_number().expect("a number").as_f64();
            value * value
        })
        .sum();
    assert!((norm - 1.0).abs() <= 1e-4, "vector norm {norm}");
}

#[test]
fn embed_rejects_a_dimension_above_the_model_output() {
    let output = run(&[
        "embed",
        "--model-dir",
        "fixtures/model/tiny_text",
        "--ids",
        "1,5,7,9,2",
        "--dim",
        "9",
    ]);
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(stderr(&output).contains("dimension"));
}

#[test]
fn embed_rejects_an_out_of_vocabulary_id() {
    let output = run(&[
        "embed",
        "--model-dir",
        "fixtures/model/tiny_text",
        "--ids",
        "1,64",
        "--dim",
        "8",
    ]);
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(stderr(&output).contains("vocabulary"));
}

#[test]
fn embed_help_is_available() {
    let output = run(&["embed", "--help"]);
    assert!(output.status.success());
    assert!(stdout(&output).contains("USAGE:"));
    assert!(output.stderr.is_empty());
}
