//! Text embedding parity tests for milestone M2.
//!
//! The tiny fixture is a complete EmbeddingGemma 2 text model generated with
//! the official Transformers implementation. It exercises sliding and full
//! attention, grouped-query attention, per-layer embeddings, RoPE with two
//! theta values, pooling, normalization, and Matryoshka truncation. It runs
//! without the 1.5 GB weights.
//!
//! When the frozen model cache is present, the same test suite replays the
//! official golden corpus on the real weights. The full-file comparison is
//! skipped on a clean checkout; the tiny fixture still runs.

use std::path::Path;

use seldon_vault_lite::binary::sha256::sha256_hex;
use seldon_vault_lite::json::{self, Number, Value};
use seldon_vault_lite::model::config::TextConfig;
use seldon_vault_lite::model::prompt::{self, Prompt};
use seldon_vault_lite::model::text::{TextEmbedder, TextModel};
use seldon_vault_lite::model::weights::TextWeights;

const TINY_CONFIG: &str = include_str!("../fixtures/model/tiny_text/config.json");
const TINY_WEIGHTS: &[u8] = include_bytes!("../fixtures/model/tiny_text/model.safetensors");
const TINY_GOLDEN: &str = include_str!("../fixtures/model/tiny_text/golden.json");
const TEXT_GOLDEN: &str = include_str!("../fixtures/model/text_golden.json");
const MANIFEST: &str = include_str!("../fixtures/model/manifest.json");
const REVISION: &str = "f9972ac22e696a105c482202646979e33e4b5a1e";

fn string<'a>(value: &'a Value, key: &str) -> &'a str {
    value
        .get(key)
        .and_then(Value::as_str)
        .expect("a string field")
}

fn ids(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .expect("an id list")
        .iter()
        .map(|id| {
            let id = id
                .as_number()
                .and_then(Number::as_u64)
                .expect("an unsigned token id");
            u32::try_from(id).expect("a token id that fits u32")
        })
        .collect()
}

fn floats(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .expect("a float array")
        .iter()
        .map(|value| value.as_number().expect("a number").as_f64() as f32)
        .collect()
}

fn float_matrix(value: &Value) -> Vec<Vec<f32>> {
    value
        .as_array()
        .expect("a matrix")
        .iter()
        .map(floats)
        .collect()
}

fn prompt_from_name(name: &str) -> Prompt {
    match name {
        "SearchQuery" => Prompt::SearchQuery,
        "Document" => Prompt::Document,
        "Classification" => Prompt::Classification,
        "Clustering" => Prompt::Clustering,
        "CodeRetrieval" => Prompt::CodeRetrieval,
        "FactChecking" => Prompt::FactChecking,
        "SentenceSimilarity" => Prompt::SentenceSimilarity,
        "QuestionAnswering" => Prompt::QuestionAnswering,
        other => panic!("unknown prompt name {other}"),
    }
}

fn assert_close(actual: &[f32], expected: &[f32], atol: f32, rtol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "length mismatch for {label}");
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        let difference = (a - e).abs();
        let tolerance = atol + rtol * e.abs();
        assert!(
            difference <= tolerance,
            "{label}[{index}]: {a} differs from {e} by {difference} (tolerance {tolerance})"
        );
    }
}

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len());
    let mut dot = 0.0f32;
    let mut left_norm = 0.0f32;
    let mut right_norm = 0.0f32;
    for (a, b) in left.iter().zip(right) {
        dot += a * b;
        left_norm += a * a;
        right_norm += b * b;
    }
    dot / (left_norm.sqrt() * right_norm.sqrt())
}

fn assert_embedding(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "dimension mismatch for {label}"
    );
    assert_close(actual, expected, 1e-4, 1e-4, label);
    let norm: f32 = actual.iter().map(|value| value * value).sum();
    assert!(
        (norm - 1.0).abs() <= 1e-4,
        "{label} is not unit length: squared norm {norm}"
    );
    let similarity = cosine(actual, expected);
    assert!(
        similarity >= 0.9999,
        "{label} cosine {similarity} is below 0.9999"
    );
}

#[test]
fn tiny_model_matches_the_official_layer_and_embedding_outputs() {
    let config = TextConfig::from_json_str(TINY_CONFIG).expect("tiny configuration must parse");
    let weights = TextWeights::from_bytes(TINY_WEIGHTS, &config).expect("tiny weights must load");
    let model = TextModel::new(config, weights).expect("tiny model");
    let golden = json::parse(TINY_GOLDEN).expect("tiny golden fixture must parse");
    let cases = golden
        .get("cases")
        .and_then(Value::as_array)
        .expect("tiny golden cases");

    for (case_index, case) in cases.iter().enumerate() {
        let input_ids = ids(case.get("input_ids").expect("input_ids"));
        let expected_means = float_matrix(case.get("hidden_state_means").expect("hidden means"));
        let expected_last = float_matrix(case.get("hidden_state_last").expect("hidden last"));
        let expected_embedding = floats(case.get("embedding").expect("embedding"));

        let output = model
            .forward(&input_ids, true)
            .expect("tiny forward pass must succeed");
        assert_eq!(
            output.hidden_states().len(),
            1 + model.config().num_hidden_layers(),
            "hidden state count for case {case_index}"
        );
        assert_eq!(output.hidden_states().len(), expected_means.len());
        assert_eq!(output.hidden_states().len(), expected_last.len());

        let hidden_size = model.config().hidden_size();
        let embedding_dim = model.config().embedding_dim();
        for (layer, (hidden, (expected_mean, expected_last))) in output
            .hidden_states()
            .iter()
            .zip(expected_means.iter().zip(&expected_last))
            .enumerate()
        {
            let width = hidden_size;
            assert_eq!(hidden.shape().dims(), &[input_ids.len(), width]);
            let rows = input_ids.len();
            let mut mean = vec![0.0f32; width];
            for row in 0..rows {
                for (column, value) in mean.iter_mut().enumerate() {
                    *value += hidden.data()[row * width + column];
                }
            }
            for value in &mut mean {
                *value /= rows as f32;
            }
            assert_close(
                &mean,
                expected_mean,
                2e-5,
                1e-4,
                &format!("case {case_index} layer {layer} mean"),
            );
            let last = &hidden.data()[(rows - 1) * width..rows * width];
            assert_close(
                last,
                expected_last,
                2e-5,
                1e-4,
                &format!("case {case_index} layer {layer} last token"),
            );
        }

        assert_eq!(
            output.projected().shape().dims(),
            &[input_ids.len(), embedding_dim]
        );
        let projected = output.projected().data();
        let mut projected_mean = vec![0.0f32; embedding_dim];
        for row in 0..input_ids.len() {
            for column in 0..embedding_dim {
                projected_mean[column] += projected[row * embedding_dim + column];
            }
        }
        for value in &mut projected_mean {
            *value /= input_ids.len() as f32;
        }
        assert_close(
            &projected_mean,
            &floats(case.get("projected_mean").expect("projected mean")),
            2e-4,
            1e-4,
            &format!("case {case_index} projected mean"),
        );
        assert_close(
            &projected[(input_ids.len() - 1) * embedding_dim..],
            &floats(case.get("projected_last").expect("projected last")),
            2e-4,
            1e-4,
            &format!("case {case_index} projected last"),
        );
        assert_embedding(
            output.normalized(),
            &expected_embedding,
            &format!("case {case_index} embedding"),
        );
        let truncated = output.embedding(4).expect("truncated embedding");
        assert_eq!(truncated.len(), 4);
        let norm: f32 = truncated.iter().map(|value| value * value).sum();
        assert!((norm - 1.0).abs() <= 1e-4);
    }
}

#[test]
fn real_model_parity_when_the_model_cache_is_present() {
    if std::env::var_os("SELDON_SKIP_MODEL_PARITY").is_some() {
        eprintln!("skipping real-model text parity: SELDON_SKIP_MODEL_PARITY is set");
        return;
    }
    let weights_path = Path::new(".model-cache")
        .join(REVISION)
        .join("model.safetensors");
    if !weights_path.is_file() {
        eprintln!(
            "skipping real-model text parity: {} is not present",
            weights_path.display()
        );
        return;
    }

    let manifest = json::parse(MANIFEST).expect("manifest must parse");
    let expected_weight_sha = manifest
        .get("files")
        .and_then(Value::as_array)
        .expect("manifest files")
        .iter()
        .find(|file| string(file, "path") == "model.safetensors")
        .map(|file| string(file, "sha256"))
        .expect("model.safetensors manifest entry");
    let expected_tokenizer_sha = string(
        manifest.get("tokenizer").expect("tokenizer section"),
        "sha256",
    );

    let tokenizer_path = Path::new(".model-cache")
        .join(REVISION)
        .join("tokenizer.json");
    let tokenizer_bytes = std::fs::read(&tokenizer_path).expect("read the frozen tokenizer");
    assert_eq!(
        sha256_hex(&tokenizer_bytes),
        expected_tokenizer_sha,
        "the frozen tokenizer SHA-256 must match the manifest"
    );

    let embedder = TextEmbedder::load(&Path::new(".model-cache").join(REVISION))
        .expect("load the frozen text embedder");
    assert_eq!(
        embedder.model().fingerprint_hex(),
        expected_weight_sha,
        "the frozen weights SHA-256 must match the manifest"
    );

    let golden = json::parse(TEXT_GOLDEN).expect("text golden fixture must parse");
    let cases = golden
        .get("cases")
        .and_then(Value::as_array)
        .expect("text golden cases");
    for case in cases {
        let name = string(case, "id");
        let prompt_name = string(case, "prompt");
        let prefix = string(case, "prefix");
        let text = string(case, "text");
        let expected_ids = ids(case.get("token_ids").expect("token_ids"));
        let expected_embeddings = case.get("embeddings").expect("embeddings");

        let full = prompt::apply(prompt_from_name(prompt_name), text);
        assert_eq!(prefix, prompt_from_name(prompt_name).prefix());
        let actual_ids = embedder
            .tokenizer()
            .encode(&full, true)
            .expect("tokenization");
        assert_eq!(actual_ids, expected_ids, "token ids for {name}");

        let prompt = prompt_from_name(prompt_name);
        for dimension in [768usize, 512, 256, 128] {
            let actual = embedder.embed(text, prompt, dimension).expect("embedding");
            let expected = floats(
                expected_embeddings
                    .get(&dimension.to_string())
                    .expect("dimension"),
            );
            assert_embedding(&actual, &expected, &format!("{name} {dimension}d"));
        }
    }

    let probe = golden.get("layer_probe").expect("layer probe");
    let probe_ids = ids(probe.get("token_ids").expect("probe ids"));
    let expected_means = float_matrix(probe.get("hidden_state_means").expect("probe means"));
    let expected_last = float_matrix(probe.get("hidden_state_last").expect("probe last"));
    let output = embedder
        .model()
        .forward(&probe_ids, true)
        .expect("probe forward pass");
    assert_eq!(output.hidden_states().len(), expected_means.len());
    assert_eq!(output.hidden_states().len(), expected_last.len());
    let hidden_size = embedder.model().config().hidden_size();
    let embedding_dim = embedder.model().config().embedding_dim();
    let rows = probe_ids.len();
    for (layer, (hidden, (expected_mean, expected_last))) in output
        .hidden_states()
        .iter()
        .zip(expected_means.iter().zip(&expected_last))
        .enumerate()
    {
        assert_eq!(hidden.shape().dims(), &[rows, hidden_size]);
        let mut mean = vec![0.0f32; hidden_size];
        for row in 0..rows {
            for (column, value) in mean.iter_mut().enumerate() {
                *value += hidden.data()[row * hidden_size + column];
            }
        }
        for value in &mut mean {
            *value /= rows as f32;
        }
        assert_close(
            &mean,
            expected_mean,
            2e-5,
            1e-4,
            &format!("probe layer {layer} mean"),
        );
        let last = &hidden.data()[(rows - 1) * hidden_size..rows * hidden_size];
        assert_close(
            last,
            expected_last,
            2e-5,
            1e-4,
            &format!("probe layer {layer} last"),
        );
    }
    let projected = output.projected();
    assert_eq!(projected.shape().dims(), &[rows, embedding_dim]);
    let mut projected_mean = vec![0.0f32; embedding_dim];
    for row in 0..rows {
        for (column, value) in projected_mean.iter_mut().enumerate() {
            *value += projected.data()[row * embedding_dim + column];
        }
    }
    for value in &mut projected_mean {
        *value /= rows as f32;
    }
    assert_close(
        &projected_mean,
        &floats(probe.get("projected_mean").expect("projected mean")),
        2e-4,
        1e-4,
        "probe projected mean",
    );
    assert_close(
        &projected.data()[(rows - 1) * embedding_dim..],
        &floats(probe.get("projected_last").expect("projected last")),
        2e-4,
        1e-4,
        "probe projected last",
    );
    let expected_embedding = floats(probe.get("embedding_768").expect("probe embedding"));
    assert_embedding(output.normalized(), &expected_embedding, "probe embedding");
}

#[test]
fn tiny_model_is_deterministic_and_rejects_invalid_dimensions() {
    let config = TextConfig::from_json_str(TINY_CONFIG).expect("tiny configuration");
    let weights = TextWeights::from_bytes(TINY_WEIGHTS, &config).expect("tiny weights");
    let model = TextModel::new(config, weights).expect("tiny model");
    // The tiny model is not a real tokenizer, so this test uses fixed ids and
    // checks that the model is deterministic rather than that prompts differ.
    let first = model.embed_ids(&[1, 5, 7, 9, 2], 8).expect("embedding");
    let second = model.embed_ids(&[1, 5, 7, 9, 2], 8).expect("embedding");
    assert_eq!(first, second, "the scalar path must be deterministic");
    let error = model
        .embed_ids(&[1, 5, 7, 9, 2], 9)
        .expect_err("dimension above the model output must fail");
    assert!(matches!(
        error,
        seldon_vault_lite::error::Error::Internal(_)
    ));
}

#[test]
fn invalid_token_sequences_are_rejected() {
    let config = TextConfig::from_json_str(TINY_CONFIG).expect("tiny configuration");
    let weights = TextWeights::from_bytes(TINY_WEIGHTS, &config).expect("tiny weights");
    let model = TextModel::new(config, weights).expect("tiny model");

    let error = model
        .forward(&[], false)
        .expect_err("empty sequence must fail");
    assert!(matches!(error, seldon_vault_lite::error::Error::Corrupt(_)));

    let error = model
        .forward(&[1, 64], false)
        .expect_err("out-of-vocabulary token must fail");
    assert!(matches!(error, seldon_vault_lite::error::Error::Corrupt(_)));

    let too_long = vec![1u32; 8193];
    let error = model
        .forward(&too_long, false)
        .expect_err("context cap must be enforced");
    assert!(matches!(
        error,
        seldon_vault_lite::error::Error::LimitExceeded(_)
    ));
}
#[test]
fn text_model_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<TextModel>();
    assert_send_sync::<TextEmbedder>();
}

#[test]
fn tiny_model_supports_bounded_concurrent_calls() {
    let config = TextConfig::from_json_str(TINY_CONFIG).expect("tiny configuration");
    let weights = TextWeights::from_bytes(TINY_WEIGHTS, &config).expect("tiny weights");
    let model = std::sync::Arc::new(TextModel::new(config, weights).expect("tiny model"));
    let expected = model.embed_ids(&[1, 5, 7, 9, 2], 8).expect("embedding");

    let mut handles = Vec::new();
    for _ in 0..4 {
        let model = std::sync::Arc::clone(&model);
        handles.push(std::thread::spawn(move || {
            model.embed_ids(&[1, 5, 7, 9, 2], 8)
        }));
    }
    for handle in handles {
        let embedding = handle.join().expect("worker thread").expect("embedding");
        assert_eq!(embedding, expected);
    }
}
