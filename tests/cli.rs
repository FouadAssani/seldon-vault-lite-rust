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
    let output = run(&["nope"]);
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
fn text_and_image_queries_rank_image_documents() {
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

    let output = run(&[
        "search",
        "--data-dir",
        data_dir.to_str().expect("UTF-8 path"),
        "--mode",
        "vector",
        "--dim",
        "768",
        "--vision-tokens",
        "70",
        "--image",
        "fixtures/media/quadrants_16x16.png",
    ]);
    assert!(
        output.status.success(),
        "image search stderr: {}",
        stderr(&output)
    );
    let result =
        seldon_vault_lite::json::parse(stdout(&output).trim()).expect("image search must be JSON");
    let hits = result
        .as_array()
        .expect("image search output is a hits array");
    assert_eq!(hits.len(), 2);
    assert_eq!(
        hits[0].get("document_id").and_then(|value| value.as_str()),
        Some("img-quadrants"),
        "the identical image must rank first"
    );
    assert_eq!(
        hits[1].get("document_id").and_then(|value| value.as_str()),
        Some("img-gray")
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

#[test]
#[ignore = "loads the full model and embeds audio documents; execute manually"]
fn audio_documents_can_be_indexed_and_found_by_text_or_audio() {
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir =
        std::env::temp_dir().join(format!("seldon-cli-audio-{}-{unique}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let jsonl = dir.join("audio.jsonl");
    std::fs::write(
        &jsonl,
        concat!(
            "{\"id\":\"rain\",\"audio\":\"fixtures/media/audio/rain_noise_2s.wav\",\"title\":\"Rain\"}\n",
            "{\"id\":\"music\",\"audio\":\"fixtures/media/audio/music_chords_2s.wav\",\"title\":\"Music\"}\n",
            "{\"id\":\"speech\",\"audio\":\"fixtures/media/audio/french_speech.wav\",\"title\":\"Speech\"}\n",
        ),
    )
    .expect("write JSONL");

    let output = run(&[
        "index",
        "--data-dir",
        dir.to_str().expect("UTF-8 temp dir"),
        "--model-dir",
        ".model-cache/f9972ac22e696a105c482202646979e33e4b5a1e",
        "--dim",
        "768",
        jsonl.to_str().expect("UTF-8 JSONL path"),
    ]);
    assert!(output.status.success(), "index stderr: {}", stderr(&output));

    let output = run(&[
        "search",
        "--data-dir",
        dir.to_str().expect("UTF-8 temp dir"),
        "--model-dir",
        ".model-cache/f9972ac22e696a105c482202646979e33e4b5a1e",
        "--dim",
        "768",
        "--mode",
        "vector",
        "--limit",
        "3",
        "rain falling",
    ]);
    assert!(
        output.status.success(),
        "search stderr: {}",
        stderr(&output)
    );
    let value = seldon_vault_lite::json::parse(stdout(&output).trim()).expect("search JSON");
    let hits = value.as_array().expect("hits array");
    assert_eq!(hits.len(), 3);
    let top = hits[0].as_object().expect("hit object");
    assert_eq!(
        top.get("source").and_then(|value| value.as_str()),
        Some("fixtures/media/audio/rain_noise_2s.wav")
    );
    assert_eq!(
        top.get("modality").and_then(|value| value.as_str()),
        Some("audio")
    );

    let output = run(&[
        "search",
        "--data-dir",
        dir.to_str().expect("UTF-8 temp dir"),
        "--model-dir",
        ".model-cache/f9972ac22e696a105c482202646979e33e4b5a1e",
        "--dim",
        "768",
        "--mode",
        "vector",
        "--limit",
        "3",
        "--audio",
        "fixtures/media/audio/music_chords_2s.wav",
    ]);
    assert!(
        output.status.success(),
        "audio search stderr: {}",
        stderr(&output)
    );
    let value = seldon_vault_lite::json::parse(stdout(&output).trim()).expect("search JSON");
    let hits = value.as_array().expect("hits array");
    assert_eq!(hits.len(), 3);
    let top = hits[0].as_object().expect("hit object");
    assert_eq!(
        top.get("source").and_then(|value| value.as_str()),
        Some("fixtures/media/audio/music_chords_2s.wav")
    );
}

#[test]
fn serve_command_starts_serves_and_stops() {
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpStream;
    use std::process::Stdio;

    let unique = format!(
        "seldon-cli-serve-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    );
    let dir = std::env::temp_dir().join(unique);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");

    let mut child = Command::new(BIN)
        .args([
            "serve",
            "--data-dir",
            dir.to_str().expect("UTF-8 temp dir"),
            "--model-dir",
            "fixtures/model/tiny_text",
            "--bind",
            "127.0.0.1:0",
            "--dim",
            "8",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("serve must start");
    let stdout = child.stdout.take().expect("stdout pipe");
    let mut reader = BufReader::new(stdout);
    let mut listening = String::new();
    reader
        .read_line(&mut listening)
        .expect("read the listening record");
    let listening =
        seldon_vault_lite::json::parse(listening.trim()).expect("listening record must be JSON");
    let address = listening
        .get("address")
        .and_then(|value| value.as_str())
        .expect("listening record has an address");
    assert_eq!(
        listening.get("status").and_then(|value| value.as_str()),
        Some("listening")
    );

    let mut stream = TcpStream::connect(address).expect("connect to serve");
    stream
        .write_all(
            b"POST /v1/shutdown HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        )
        .expect("write shutdown request");
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .expect("read shutdown response");
    let response = String::from_utf8(response).expect("UTF-8 response");
    assert!(
        response.starts_with("HTTP/1.1 202"),
        "unexpected shutdown response: {response}"
    );

    let status = child.wait().expect("wait for serve to stop");
    assert!(status.success());
    let mut stopped = String::new();
    reader
        .read_line(&mut stopped)
        .expect("read the stopped record");
    let stopped =
        seldon_vault_lite::json::parse(stopped.trim()).expect("stopped record must be JSON");
    assert_eq!(
        stopped.get("status").and_then(|value| value.as_str()),
        Some("stopped")
    );

    let _ = std::fs::remove_dir_all(&dir);
}
