//! Audio embedding parity tests against the official reference.
//!
//! The full audio path loads the frozen text and audio towers. The default test
//! covers the short reference clip; the complete six-case corpus is available
//! as an ignored test for manual verification. Preprocessing itself is covered
//! by tests/audio.rs.

use std::path::Path;

use seldon_vault_lite::json::{self, Value};
use seldon_vault_lite::media::wav;
use seldon_vault_lite::model::{AudioEmbedder, AudioSpecialTokens};

const REVISION: &str = "f9972ac22e696a105c482202646979e33e4b5a1e";
const MANIFEST: &str = include_str!("../fixtures/model/manifest.json");
const AUDIO_GOLDEN: &str = include_str!("../fixtures/model/audio_golden.json");

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

fn model_cache() -> Option<&'static Path> {
    if std::env::var_os("SELDON_SKIP_MODEL_PARITY").is_some() {
        eprintln!("skipping real-model audio parity: SELDON_SKIP_MODEL_PARITY is set");
        return None;
    }
    let path = Path::new(".model-cache").join(REVISION);
    if !path.join("model.safetensors").is_file() {
        eprintln!(
            "skipping real-model audio parity: {} is not present",
            path.join("model.safetensors").display()
        );
        return None;
    }
    Some(Box::leak(path.into_boxed_path()))
}

fn load_embedder(cache: &Path) -> AudioEmbedder {
    let manifest = json::parse(MANIFEST).expect("manifest must parse");
    let expected_weight_sha = manifest
        .get("files")
        .and_then(Value::as_array)
        .expect("manifest files")
        .iter()
        .find(|file| string(file, "path") == "model.safetensors")
        .map(|file| string(file, "sha256"))
        .expect("model.safetensors manifest entry");
    let embedder = AudioEmbedder::load(cache).expect("load the frozen audio embedder");
    assert_eq!(
        embedder.audio().fingerprint_hex(),
        expected_weight_sha,
        "the frozen weights SHA-256 must match the manifest"
    );
    assert_eq!(
        embedder.special_tokens(),
        AudioSpecialTokens {
            bos: 2,
            eos: 1,
            pad: 0,
            boa: 256_000,
            audio: 258_881,
            eoa: 258_883,
        },
        "special token ids"
    );
    embedder
}

fn run_case(embedder: &AudioEmbedder, case: &Value) {
    let name = string(case, "name");
    let source = string(case, "source");
    let audio = wav::decode_file(Path::new(source))
        .unwrap_or_else(|error| panic!("{name}: source WAV must decode: {error}"));
    let output = embedder
        .embed_with_capture(&audio, 768)
        .unwrap_or_else(|error| panic!("{name}: audio embedding failed: {error}"));

    let expected_tokens = integer(case, "audio_tokens") as usize;
    assert_eq!(
        output.audio().audio_tokens(),
        expected_tokens,
        "{name}: audio soft tokens"
    );
    let expected_ids = floats(case.get("input_ids").expect("input ids"))
        .into_iter()
        .map(|value| value as u32)
        .collect::<Vec<_>>();
    let tokens = embedder.special_tokens();
    let mut input_ids = Vec::with_capacity(expected_tokens + 4);
    input_ids.push(tokens.bos);
    input_ids.push(tokens.boa);
    input_ids.extend(std::iter::repeat_n(tokens.audio, expected_tokens));
    input_ids.push(tokens.eoa);
    input_ids.push(tokens.eos);
    assert_eq!(input_ids, expected_ids, "{name}: input token layout");

    let sscp = output.audio().sscp_output().expect("SSCP capture");
    let expected = floats(case.get("sscp_output_first").expect("SSCP probe"));
    assert_close(
        &sscp.data()[..expected.len()],
        &expected,
        &format!("{name} SSCP output"),
        2e-4,
        2e-4,
    );

    let layers = case
        .get("layer_first")
        .and_then(Value::as_array)
        .expect("layer probes");
    assert_eq!(layers.len(), 12, "{name}: twelve encoder layers");
    assert_eq!(
        output.audio().hidden_states().len(),
        12,
        "{name}: captured layer count"
    );
    for (index, probe) in layers.iter().enumerate() {
        let expected = floats(probe);
        assert_close(
            &output.audio().hidden_states()[index].data()[..expected.len()],
            &expected,
            &format!("{name} audio layer {index}"),
            2e-4,
            2e-4,
        );
    }

    let features = floats(
        case.get("audio_features_first")
            .expect("audio feature probe"),
    );
    assert_close(
        &output.audio().pooled()[..features.len()],
        &features,
        &format!("{name} audio projection"),
        2e-4,
        2e-4,
    );

    let expected = floats(case.get("audio_embedding").expect("audio embedding"));
    let actual = output.embedding();
    assert_eq!(actual.len(), 768, "{name}: embedding dimension");
    let similarity = cosine(actual, &expected);
    assert!(
        similarity >= 0.999,
        "{name}: cosine similarity {similarity} is below 0.999"
    );
    let mut maximum = 0.0f32;
    for (a, e) in actual.iter().zip(&expected) {
        maximum = maximum.max((a - e).abs());
    }
    assert!(
        maximum <= 2e-3,
        "{name}: maximum absolute embedding difference {maximum} exceeds 2e-3"
    );

    let norm: f32 = actual.iter().map(|value| value * value).sum();
    assert!(
        (norm - 1.0).abs() <= 1e-5,
        "{name}: embedding norm {norm} is not unit length"
    );

    for dimension in [768usize, 512, 256, 128] {
        let truncated = output
            .forward()
            .embedding(dimension)
            .expect("Matryoshka truncation");
        assert_eq!(truncated.len(), dimension, "{name}: truncated dimension");
        let norm: f32 = truncated.iter().map(|value| value * value).sum();
        assert!(
            (norm - 1.0).abs() <= 1e-5,
            "{name}: truncated norm at {dimension}d is {norm}"
        );
        let prefix = &actual[..dimension];
        let prefix_similarity = cosine(&truncated, prefix);
        assert!(
            prefix_similarity >= 0.999999,
            "{name}: {dimension}d truncation is not the renormalized prefix ({prefix_similarity})"
        );
    }
}

#[test]
fn audio_embedding_matches_the_official_reference() {
    let Some(cache) = model_cache() else {
        return;
    };
    let embedder = load_embedder(cache);
    let golden = json::parse(AUDIO_GOLDEN).expect("audio golden fixture must parse");
    let cases = golden
        .get("cases")
        .and_then(Value::as_array)
        .expect("audio golden cases");
    let case = cases
        .iter()
        .find(|case| string(case, "name") == "short_tone_100ms")
        .expect("short tone reference case");
    run_case(&embedder, case);
}

#[test]
#[ignore = "full six-case audio parity run; execute manually"]
fn all_audio_embeddings_match_the_official_reference() {
    let Some(cache) = model_cache() else {
        return;
    };
    let embedder = load_embedder(cache);
    let golden = json::parse(AUDIO_GOLDEN).expect("audio golden fixture must parse");
    let cases = golden
        .get("cases")
        .and_then(Value::as_array)
        .expect("audio golden cases");
    for case in cases {
        run_case(&embedder, case);
    }
}

#[test]
fn rejects_an_out_of_range_embedding_dimension() {
    let Some(cache) = model_cache() else {
        return;
    };
    let embedder = load_embedder(cache);
    let audio = wav::decode_file(Path::new("fixtures/media/audio/short_tone_100ms.wav"))
        .expect("audio fixture");
    let error = embedder
        .embed(&audio, 0)
        .expect_err("zero dimension must fail");
    assert!(matches!(
        error,
        seldon_vault_lite::error::Error::Internal(_)
    ));
    let error = embedder
        .embed(&audio, 769)
        .expect_err("oversized dimension must fail");
    assert!(matches!(
        error,
        seldon_vault_lite::error::Error::Internal(_)
    ));
}

#[test]
#[ignore = "text-to-audio ranking over the six-case corpus; execute manually"]
fn text_queries_rank_the_expected_audio_clip_first() {
    let Some(cache) = model_cache() else {
        return;
    };
    let embedder = load_embedder(cache);
    let golden = json::parse(AUDIO_GOLDEN).expect("audio golden fixture must parse");
    let cases = golden
        .get("cases")
        .and_then(Value::as_array)
        .expect("audio golden cases");
    let mut clips = Vec::new();
    for case in cases {
        let name = string(case, "name");
        let source = string(case, "source");
        let audio = wav::decode_file(Path::new(source))
            .unwrap_or_else(|error| panic!("{name}: WAV must decode: {error}"));
        let embedding = embedder
            .embed(&audio, 768)
            .unwrap_or_else(|error| panic!("{name}: embedding failed: {error}"));
        clips.push((name.to_owned(), embedding));
    }

    for (query, expected) in [
        ("rain falling", "rain_noise_2s"),
        ("music", "music_chords_2s"),
        ("a test of audio search", "french_speech"),
    ] {
        let query_embedding = embedder
            .embed_text(
                query,
                seldon_vault_lite::model::prompt::Prompt::SearchQuery,
                768,
            )
            .expect("text query embedding");
        let mut ranked: Vec<(f64, &str)> = clips
            .iter()
            .map(|(name, embedding)| (cosine(&query_embedding, embedding), name.as_str()))
            .collect();
        ranked.sort_by(|left, right| right.0.total_cmp(&left.0));
        assert_eq!(ranked[0].1, expected, "query {query:?} ranking: {ranked:?}");
        assert!(
            ranked[0].0 > ranked[1].0,
            "query {query:?} has no strict top hit: {ranked:?}"
        );
        println!("query {query:?} ranking: {ranked:?}");
    }
}
