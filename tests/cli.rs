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
    let text = stdout(&output);
    assert!(text.contains("USAGE:"));
    assert!(text.contains("--image"));
    assert!(output.stderr.is_empty());
}

#[test]
#[ignore = "loads the full multimodal model and runs the scalar vision tower"]
fn indexes_an_image_and_finds_it_with_a_text_query() {
    use std::fs;

    let unique = format!(
        "seldon-image-cli-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    );
    let dir = std::env::temp_dir().join(unique);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create temp dir");
    let jsonl = dir.join("images.jsonl");
    fs::write(
        &jsonl,
        "{\"id\":\"img-1\",\"image\":\"fixtures/media/quadrants_16x16.png\",\"title\":\"Quadrants\"}\n",
    )
    .expect("write JSONL");
    let data_dir = dir.join("data");

    let output = run(&[
        "index",
        "--data-dir",
        data_dir.to_str().expect("UTF-8 path"),
        "--vision-tokens",
        "70",
        jsonl.to_str().expect("UTF-8 path"),
    ]);
    assert!(output.status.success(), "index stderr: {}", stderr(&output));
    let summary =
        seldon_vault_lite::json::parse(stdout(&output).trim()).expect("index summary must be JSON");
    assert_eq!(
        summary
            .get("documents")
            .and_then(|value| value.as_number())
            .and_then(|number| number.as_u64()),
        Some(1)
    );

    let output = run(&[
        "search",
        "--data-dir",
        data_dir.to_str().expect("UTF-8 path"),
        "--mode",
        "vector",
        "--dim",
        "768",
        "a colorful four-quadrant test pattern",
    ]);
    assert!(
        output.status.success(),
        "search stderr: {}",
        stderr(&output)
    );
    let result =
        seldon_vault_lite::json::parse(stdout(&output).trim()).expect("search result must be JSON");
    let hits = result.as_array().expect("search output is a hits array");
    assert_eq!(hits.len(), 1, "the image corpus has one document");
    assert_eq!(
        hits[0].get("modality").and_then(|value| value.as_str()),
        Some("image")
    );
    assert_eq!(
        hits[0].get("source").and_then(|value| value.as_str()),
        Some("fixtures/media/quadrants_16x16.png")
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "loads the full multimodal model and embeds two images with the scalar vision tower"]
fn text_query_ranks_image_documents() {
    use std::fs;

    let unique = format!(
        "seldon-image-rank-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    );
    let dir = std::env::temp_dir().join(unique);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create temp dir");
    let jsonl = dir.join("images.jsonl");
    fs::write(
        &jsonl,
        "{\"id\":\"img-quadrants\",\"image\":\"fixtures/media/quadrants_16x16.png\"}\n         {\"id\":\"img-gray\",\"image\":\"fixtures/media/gray_2x2.png\"}\n",
    )
    .expect("write JSONL");
    let data_dir = dir.join("data");

    let output = run(&[
        "index",
        "--data-dir",
        data_dir.to_str().expect("UTF-8 path"),
        "--vision-tokens",
        "70",
        jsonl.to_str().expect("UTF-8 path"),
    ]);
    assert!(output.status.success(), "index stderr: {}", stderr(&output));

    let output = run(&[
        "search",
        "--data-dir",
        data_dir.to_str().expect("UTF-8 path"),
        "--mode",
        "vector",
        "--dim",
        "768",
        "a colorful four-quadrant test pattern",
    ]);
    assert!(
        output.status.success(),
        "search stderr: {}",
        stderr(&output)
    );
    let result =
        seldon_vault_lite::json::parse(stdout(&output).trim()).expect("search result must be JSON");
    let hits = result.as_array().expect("search output is a hits array");
    assert_eq!(hits.len(), 2);
    assert_eq!(
        hits[0].get("document_id").and_then(|value| value.as_str()),
        Some("img-quadrants"),
        "the colorful quadrant image must rank first"
    );
    assert_eq!(
        hits[1].get("document_id").and_then(|value| value.as_str()),
        Some("img-gray")
    );
    assert_eq!(
        hits[0].get("modality").and_then(|value| value.as_str()),
        Some("image")
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn embed_rejects_an_unknown_image_extension() {
    let output = run(&["embed", "--image", "fixtures/media/README.md"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    let text = stderr(&output);
    assert!(text.contains("extension"), "stderr: {text}");
}
