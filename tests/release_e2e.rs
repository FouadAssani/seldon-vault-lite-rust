//! End-to-end release validation with the committed tiny text and audio model.
//!
//! This test runs the real binary in a sanitized environment: a temporary
//! working directory, a temporary HOME and TMPDIR, and a system-only PATH.
//! It does not use the development model cache. The flow covers token-id text
//! embedding, text and audio ingestion, lexical and vector search, verification,
//! compaction, and a search after the compaction restart. The shell harness
//! scripts/release_e2e.sh covers the full frozen model and the HTTP API on a
//! clean machine; this test keeps the same lifecycle in the default test suite.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

use seldon_vault_lite::json::{self, Value};

const BIN: &str = env!("CARGO_BIN_EXE_seldon-vault-lite");

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_root() -> PathBuf {
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let root = std::env::temp_dir().join(format!(
        "seldon-release-e2e-{}-{unique}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(root.join("home")).expect("create home");
    fs::create_dir_all(root.join("tmp")).expect("create tmp");
    root
}

fn run(root: &Path, args: &[&str]) -> Output {
    Command::new(BIN)
        .args(args)
        .current_dir(root)
        .env_clear()
        .env("HOME", root.join("home"))
        .env("TMPDIR", root.join("tmp"))
        .env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
        .env("LANG", "en_US.UTF-8")
        .env("LC_ALL", "en_US.UTF-8")
        .output()
        .expect("the release binary must be launchable")
}

fn stdout(output: &Output) -> String {
    String::from_utf8(output.stdout.clone()).expect("stdout must be UTF-8")
}

fn stderr(output: &Output) -> String {
    String::from_utf8(output.stderr.clone()).expect("stderr must be UTF-8")
}

fn parse_json(output: &Output, label: &str) -> Value {
    let text = stdout(output);
    json::parse(text.trim()).unwrap_or_else(|error| panic!("{label} is not JSON: {error}\n{text}"))
}

fn array<'a>(value: &'a Value, label: &str) -> &'a [Value] {
    value
        .as_array()
        .unwrap_or_else(|| panic!("{label} must be an array"))
}

fn number(value: &Value, key: &str) -> u64 {
    value
        .get(key)
        .and_then(Value::as_number)
        .and_then(|number| number.as_u64())
        .unwrap_or_else(|| panic!("missing unsigned number {key}"))
}

fn string<'a>(value: &'a Value, key: &str) -> &'a str {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("missing string {key}"))
}

fn vector(value: &Value, label: &str) -> Vec<f32> {
    array(value, label)
        .iter()
        .map(|item| {
            item.as_number()
                .unwrap_or_else(|| panic!("{label} contains a non-number"))
                .as_f64() as f32
        })
        .collect()
}

fn assert_unit_norm(values: &[f32], dimension: usize) {
    assert_eq!(values.len(), dimension, "vector length");
    let norm: f64 = values
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt();
    assert!(
        (norm - 1.0).abs() <= 1e-4,
        "vector norm {norm} is not 1 within tolerance"
    );
}

#[test]
fn release_lifecycle_runs_in_a_clean_environment() {
    let root = temp_root();
    let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/model/tiny_audio");
    let audio_source =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/media/audio/short_tone_100ms.wav");
    let fixtures = root.join("fixtures");
    fs::create_dir_all(&fixtures).expect("create fixtures");
    fs::copy(&audio_source, fixtures.join("short_tone_100ms.wav")).expect("copy audio fixture");

    let version = run(&root, &["version"]);
    assert!(
        version.status.success(),
        "version stderr: {}",
        stderr(&version)
    );
    assert_eq!(
        stdout(&version),
        format!(
            "seldon-vault-lite {} ({})\n",
            env!("CARGO_PKG_VERSION"),
            seldon_vault_lite::TARGET_TRIPLE
        )
    );

    let help = run(&root, &["help"]);
    assert!(help.status.success());
    let help_text = stdout(&help);
    for command in ["embed", "index", "search", "verify", "compact", "serve"] {
        assert!(help_text.contains(command), "help must mention {command}");
    }

    let model_arg = model.to_str().expect("UTF-8 model path");
    let embedded = run(
        &root,
        &[
            "embed",
            "--model-dir",
            model_arg,
            "--ids",
            "1,5,7,9,2",
            "--dim",
            "8",
        ],
    );
    assert!(
        embedded.status.success(),
        "embed stderr: {}",
        stderr(&embedded)
    );
    let embedded_vector = parse_json(&embedded, "embed output");
    assert_unit_norm(&vector(&embedded_vector, "embedding"), 8);

    fs::write(
        root.join("text.jsonl"),
        concat!(
            "{\"id\":\"text-1\",\"ids\":[1,5,7,9,2],\"text\":\"ownership and borrowing in rust\",\"title\":\"Rust\"}\n",
            "{\"id\":\"text-2\",\"ids\":[2,6,8,3,1],\"text\":\"the northern lights are caused by solar wind\",\"title\":\"Aurora\"}\n",
        ),
    )
    .expect("write text JSONL");
    fs::write(
        root.join("audio.jsonl"),
        "{\"id\":\"audio-1\",\"audio\":\"fixtures/short_tone_100ms.wav\",\"title\":\"Tone\"}\n",
    )
    .expect("write audio JSONL");

    let text_index = run(
        &root,
        &[
            "index",
            "--data-dir",
            "data",
            "--model-dir",
            model_arg,
            "--dim",
            "8",
            "--ids",
            "text.jsonl",
        ],
    );
    assert!(
        text_index.status.success(),
        "text index stderr: {}",
        stderr(&text_index)
    );
    let text_summary = parse_json(&text_index, "text index summary");
    assert_eq!(number(&text_summary, "documents"), 2);
    assert_eq!(number(&text_summary, "chunks"), 2);
    assert_eq!(number(&text_summary, "dimension"), 8);

    let audio_index = run(
        &root,
        &[
            "index",
            "--data-dir",
            "data",
            "--model-dir",
            model_arg,
            "--dim",
            "8",
            "audio.jsonl",
        ],
    );
    assert!(
        audio_index.status.success(),
        "audio index stderr: {}",
        stderr(&audio_index)
    );
    let audio_summary = parse_json(&audio_index, "audio index summary");
    assert_eq!(number(&audio_summary, "documents"), 1);
    assert_eq!(number(&audio_summary, "chunks"), 1);
    assert_eq!(number(&audio_summary, "dimension"), 8);

    let lexical = run(
        &root,
        &[
            "search",
            "--data-dir",
            "data",
            "--mode",
            "lexical",
            "northern lights",
        ],
    );
    assert!(
        lexical.status.success(),
        "lexical stderr: {}",
        stderr(&lexical)
    );
    let lexical_hits = parse_json(&lexical, "lexical hits");
    let lexical_hits = array(&lexical_hits, "lexical hits");
    assert!(!lexical_hits.is_empty(), "lexical search must find text-2");
    assert_eq!(string(&lexical_hits[0], "document_id"), "text-2");
    assert_eq!(string(&lexical_hits[0], "modality"), "text");

    let vector_text = run(
        &root,
        &[
            "search",
            "--data-dir",
            "data",
            "--model-dir",
            model_arg,
            "--dim",
            "8",
            "--mode",
            "vector",
            "--ids",
            "1,5,7,9,2",
        ],
    );
    assert!(
        vector_text.status.success(),
        "vector text stderr: {}",
        stderr(&vector_text)
    );
    let vector_text_hits = parse_json(&vector_text, "vector text hits");
    let vector_text_hits = array(&vector_text_hits, "vector text hits");
    assert_eq!(vector_text_hits.len(), 3);
    assert_eq!(string(&vector_text_hits[0], "document_id"), "text-1");

    let vector_audio = run(
        &root,
        &[
            "search",
            "--data-dir",
            "data",
            "--model-dir",
            model_arg,
            "--dim",
            "8",
            "--mode",
            "vector",
            "--audio",
            "fixtures/short_tone_100ms.wav",
        ],
    );
    assert!(
        vector_audio.status.success(),
        "vector audio stderr: {}",
        stderr(&vector_audio)
    );
    let vector_audio_hits = parse_json(&vector_audio, "vector audio hits");
    let vector_audio_hits = array(&vector_audio_hits, "vector audio hits");
    assert_eq!(vector_audio_hits.len(), 3);
    assert_eq!(string(&vector_audio_hits[0], "document_id"), "audio-1");
    assert_eq!(string(&vector_audio_hits[0], "modality"), "audio");

    let verify_before = run(&root, &["verify", "--data-dir", "data"]);
    assert!(
        verify_before.status.success(),
        "verify stderr: {}",
        stderr(&verify_before)
    );
    let verify_before = parse_json(&verify_before, "verify output");
    assert_eq!(number(&verify_before, "documents"), 3);
    assert_eq!(number(&verify_before, "chunks"), 3);
    assert_eq!(number(&verify_before, "truncated_bytes"), 0);

    let compact = run(&root, &["compact", "--data-dir", "data"]);
    assert!(
        compact.status.success(),
        "compact stderr: {}",
        stderr(&compact)
    );
    let compact = parse_json(&compact, "compact output");
    assert_eq!(number(&compact, "documents"), 3);
    assert_eq!(number(&compact, "chunks"), 3);
    assert!(number(&compact, "snapshot_bytes") > 0);

    let verify_after = run(&root, &["verify", "--data-dir", "data"]);
    assert!(
        verify_after.status.success(),
        "verify after compaction stderr: {}",
        stderr(&verify_after)
    );
    let verify_after = parse_json(&verify_after, "verify after compaction");
    assert_eq!(number(&verify_after, "documents"), 3);
    assert_eq!(number(&verify_after, "chunks"), 3);
    assert_eq!(number(&verify_after, "records"), 0);
    assert_eq!(number(&verify_after, "snapshot_records"), 3);

    let vector_after = run(
        &root,
        &[
            "search",
            "--data-dir",
            "data",
            "--model-dir",
            model_arg,
            "--dim",
            "8",
            "--mode",
            "vector",
            "--ids",
            "1,5,7,9,2",
        ],
    );
    assert!(
        vector_after.status.success(),
        "search after compaction stderr: {}",
        stderr(&vector_after)
    );
    let vector_after = parse_json(&vector_after, "search after compaction");
    let vector_after = array(&vector_after, "search after compaction");
    assert_eq!(vector_after.len(), 3);
    assert_eq!(string(&vector_after[0], "document_id"), "text-1");

    assert!(
        root.join("data/journal.log").is_file(),
        "journal must exist"
    );
    assert!(
        root.join("data/snapshot.sv").is_file(),
        "snapshot must exist after compaction"
    );
    assert!(
        !root.join(".model-cache").exists(),
        "the clean run must not create a model cache"
    );

    let _ = fs::remove_dir_all(&root);
}
