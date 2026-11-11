//! Tiny official audio fixture test.
//!
//! The fixture is a deterministic two-layer Gemma 4 audio tower with the
//! committed tiny text model. It exercises the audio configuration, weight
//! loader, SSCP, chunked attention, light convolution, feed-forward branches,
//! projection, and the text connection without the 1.5 GB frozen model.

use std::path::Path;

use seldon_vault_lite::json::{self, Value};
use seldon_vault_lite::media::audio_preprocess::log_mel;
use seldon_vault_lite::model::{
    AudioConfig, AudioModel, AudioWeights, TextConfig, TextModel, TextWeights,
};

const TINY_CONFIG: &str = include_str!("../fixtures/model/tiny_audio/config.json");
const TINY_WEIGHTS: &[u8] = include_bytes!("../fixtures/model/tiny_audio/model.safetensors");
const TINY_GOLDEN: &str = include_str!("../fixtures/model/tiny_audio/golden.json");

fn string<'a>(value: &'a Value, key: &str) -> &'a str {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("missing string field {key}"))
}

fn integer(value: &Value, key: &str) -> u64 {
    value
        .get(key)
        .and_then(Value::as_number)
        .and_then(|number| number.as_u64())
        .unwrap_or_else(|| panic!("missing integer field {key}"))
}

fn floats(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .expect("array of numbers")
        .iter()
        .map(|item| item.as_number().expect("number").as_f64() as f32)
        .collect()
}

fn matrix(value: &Value) -> Vec<Vec<f32>> {
    value
        .as_array()
        .expect("array of rows")
        .iter()
        .map(floats)
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], label: &str, atol: f32, rtol: f32) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        let tolerance = atol + rtol * e.abs();
        assert!(
            (a - e).abs() <= tolerance,
            "{label}: value {index} differs: {a} vs {e} (tolerance {tolerance})"
        );
    }
}

fn cosine(actual: &[f32], expected: &[f32]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    let mut dot = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut expected_norm = 0.0f64;
    for (a, e) in actual.iter().zip(expected) {
        dot += f64::from(*a) * f64::from(*e);
        actual_norm += f64::from(*a) * f64::from(*a);
        expected_norm += f64::from(*e) * f64::from(*e);
    }
    dot / (actual_norm.sqrt() * expected_norm.sqrt())
}

#[test]
fn tiny_audio_model_matches_the_official_fixture() {
    let golden = json::parse(TINY_GOLDEN).expect("tiny audio golden must parse");
    let text_config = TextConfig::from_json_str(TINY_CONFIG).expect("tiny text configuration");
    let audio_config = AudioConfig::from_json_str(TINY_CONFIG).expect("tiny audio configuration");
    let text_weights =
        TextWeights::from_bytes(TINY_WEIGHTS, &text_config).expect("tiny text weights");
    let audio_weights =
        AudioWeights::from_bytes(TINY_WEIGHTS, &audio_config, text_config.hidden_size())
            .expect("tiny audio weights");
    let text_model = TextModel::new(text_config, text_weights).expect("tiny text model");
    let audio_model = AudioModel::new(
        audio_config,
        audio_weights,
        text_model.config().hidden_size(),
    )
    .expect("tiny audio model");

    let samples = floats(golden.get("samples").expect("samples"));
    let mel = log_mel(&samples).expect("tiny audio mel");
    assert_eq!(mel.frames(), 12, "tiny mel frame count");
    assert_eq!(mel.valid_frames(), 12, "tiny valid frame count");
    assert_eq!(mel.frames() * 128, mel.features().len());

    let expected_mel = matrix(golden.get("mel_features").expect("mel features"));
    assert_eq!(expected_mel.len(), mel.frames(), "tiny mel rows");
    for (row, expected) in expected_mel.iter().enumerate() {
        assert_close(
            &mel.features()[row * 128..(row + 1) * 128],
            expected,
            &format!("tiny mel row {row}"),
            2e-5,
            2e-5,
        );
    }

    let output = audio_model
        .forward(&mel, true)
        .expect("tiny audio forward pass");
    assert_eq!(
        output.audio_tokens(),
        integer(&golden, "audio_tokens") as usize
    );
    assert_eq!(output.text_hidden(), 16);
    assert_eq!(output.hidden_states().len(), 2, "tiny encoder layers");

    let sscp = output.sscp_output().expect("tiny SSCP capture");
    let expected_sscp = matrix(golden.get("sscp_output").expect("SSCP output"));
    assert_eq!(sscp.data().len(), expected_sscp.len() * 16);
    for (row, expected) in expected_sscp.iter().enumerate() {
        assert_close(
            &sscp.data()[row * 16..(row + 1) * 16],
            expected,
            &format!("tiny SSCP row {row}"),
            2e-4,
            2e-4,
        );
    }

    let expected_layers = golden
        .get("layer_outputs")
        .and_then(Value::as_array)
        .expect("layer outputs");
    assert_eq!(expected_layers.len(), 2);
    for (index, expected) in expected_layers.iter().enumerate() {
        let expected = matrix(expected);
        let actual = output.hidden_states()[index].data();
        assert_eq!(actual.len(), expected.len() * 16);
        for (row, row_expected) in expected.iter().enumerate() {
            assert_close(
                &actual[row * 16..(row + 1) * 16],
                row_expected,
                &format!("tiny audio layer {index} row {row}"),
                2e-4,
                2e-4,
            );
        }
    }

    let expected_features = matrix(golden.get("audio_features").expect("audio features"));
    assert_eq!(output.pooled().len(), expected_features.len() * 16);
    for (row, expected) in expected_features.iter().enumerate() {
        assert_close(
            &output.pooled()[row * 16..(row + 1) * 16],
            expected,
            &format!("tiny audio feature row {row}"),
            2e-4,
            2e-4,
        );
    }

    let input_ids = floats(golden.get("input_ids").expect("input ids"))
        .into_iter()
        .map(|value| value as u32)
        .collect::<Vec<_>>();
    let forward = text_model
        .forward_with_features(&input_ids, output.pooled(), 2, output.audio_tokens(), true)
        .expect("tiny text forward with audio features");
    let embedding = forward.embedding(8).expect("tiny embedding");
    let expected_embedding = floats(golden.get("embedding").expect("embedding"));
    let similarity = cosine(&embedding, &expected_embedding);
    assert!(
        similarity >= 0.9999,
        "tiny audio embedding cosine {similarity} is below 0.9999"
    );
    let mut maximum = 0.0f32;
    for (actual, expected) in embedding.iter().zip(&expected_embedding) {
        maximum = maximum.max((actual - expected).abs());
    }
    assert!(
        maximum <= 1e-4,
        "tiny audio embedding maximum difference {maximum} exceeds 1e-4"
    );
    let norm: f32 = embedding.iter().map(|value| value * value).sum();
    assert!((norm - 1.0).abs() <= 1e-5, "tiny embedding norm {norm}");

    for dimension in [8usize, 4, 2, 1] {
        let truncated = forward.embedding(dimension).expect("tiny truncation");
        assert_eq!(truncated.len(), dimension);
        let norm: f32 = truncated.iter().map(|value| value * value).sum();
        assert!((norm - 1.0).abs() <= 1e-5, "tiny {dimension}d norm {norm}");
        assert!(cosine(&truncated, &embedding[..dimension]) >= 0.999999);
    }
}

#[test]
fn tiny_audio_fixture_is_deterministic() {
    let golden = json::parse(TINY_GOLDEN).expect("tiny audio golden must parse");
    let text_config = TextConfig::from_json_str(TINY_CONFIG).expect("tiny text configuration");
    let audio_config = AudioConfig::from_json_str(TINY_CONFIG).expect("tiny audio configuration");
    let text_weights =
        TextWeights::from_bytes(TINY_WEIGHTS, &text_config).expect("tiny text weights");
    let audio_weights =
        AudioWeights::from_bytes(TINY_WEIGHTS, &audio_config, text_config.hidden_size())
            .expect("tiny audio weights");
    let text_model = TextModel::new(text_config, text_weights).expect("tiny text model");
    let audio_model = AudioModel::new(
        audio_config,
        audio_weights,
        text_model.config().hidden_size(),
    )
    .expect("tiny audio model");
    let samples = floats(golden.get("samples").expect("samples"));
    let mel = log_mel(&samples).expect("tiny audio mel");
    let first = audio_model.forward(&mel, false).expect("first pass");
    let second = audio_model.forward(&mel, false).expect("second pass");
    assert_eq!(first.pooled(), second.pooled());
}

#[test]
fn tiny_audio_fixture_files_are_committed() {
    assert!(Path::new("fixtures/model/tiny_audio/config.json").is_file());
    assert!(Path::new("fixtures/model/tiny_audio/model.safetensors").is_file());
    assert!(Path::new("fixtures/model/tiny_audio/golden.json").is_file());
    assert_eq!(
        string(&json::parse(TINY_GOLDEN).expect("golden"), "model"),
        "google/embeddinggemma-2"
    );
}
