//! Image embedding parity tests against the official reference.
//!
//! The full image path is expensive: the official preprocessing upscales a
//! small image to 576 patches at the 70-token budget, and the scalar vision
//! tower then runs 16 layers. The default test covers one reference case; the
//! complete seven-case corpus is available as an ignored test for manual
//! verification. Preprocessing itself is covered by tests/preprocess.rs.

use std::path::Path;

use seldon_vault_lite::json::{self, Value};
use seldon_vault_lite::media::png;
use seldon_vault_lite::media::preprocess::PreprocessOptions;
use seldon_vault_lite::model::{ImageEmbedder, ImageSpecialTokens};

const REVISION: &str = "f9972ac22e696a105c482202646979e33e4b5a1e";
const MANIFEST: &str = include_str!("../fixtures/model/manifest.json");
const IMAGE_GOLDEN: &str = include_str!("../fixtures/model/image_golden.json");

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
        eprintln!("skipping real-model image parity: SELDON_SKIP_MODEL_PARITY is set");
        return None;
    }
    let path = Path::new(".model-cache").join(REVISION);
    if !path.join("model.safetensors").is_file() {
        eprintln!(
            "skipping real-model image parity: {} is not present",
            path.join("model.safetensors").display()
        );
        return None;
    }
    Some(Box::leak(path.into_boxed_path()))
}

fn load_embedder(cache: &Path) -> ImageEmbedder {
    let manifest = json::parse(MANIFEST).expect("manifest must parse");
    let expected_weight_sha = manifest
        .get("files")
        .and_then(Value::as_array)
        .expect("manifest files")
        .iter()
        .find(|file| string(file, "path") == "model.safetensors")
        .map(|file| string(file, "sha256"))
        .expect("model.safetensors manifest entry");
    let embedder = ImageEmbedder::load(cache).expect("load the frozen image embedder");
    assert_eq!(
        embedder.vision().fingerprint_hex(),
        expected_weight_sha,
        "the frozen weights SHA-256 must match the manifest"
    );
    assert_eq!(
        embedder.special_tokens(),
        ImageSpecialTokens {
            bos: 2,
            eos: 1,
            pad: 0,
            boi: 255_999,
            image: 258_880,
            eoi: 258_882,
        },
        "special token ids"
    );
    embedder
}

fn run_case(embedder: &ImageEmbedder, case: &Value) {
    let name = string(case, "name");
    let source = string(case, "source");
    let budget = integer(case, "max_soft_tokens") as usize;
    let image = png::decode_file(Path::new(source))
        .unwrap_or_else(|error| panic!("{name}: source PNG must decode: {error}"));
    let output = embedder
        .embed_with_capture(
            &image,
            768,
            PreprocessOptions::new(budget).expect("golden budget"),
        )
        .unwrap_or_else(|error| panic!("{name}: image embedding failed: {error}"));

    assert_eq!(
        output.vision().soft_tokens(),
        integer(case, "soft_tokens") as usize,
        "{name}: soft tokens"
    );

    let patch_first = floats(case.get("patch_output_first").expect("patch output probe"));
    assert_close(
        &output.vision().hidden_states()[0].data()[..patch_first.len()],
        &patch_first,
        &format!("{name} patch embedder"),
        2e-5,
        1e-4,
    );

    let layers = case
        .get("layer_first")
        .and_then(Value::as_array)
        .expect("layer probes");
    assert_eq!(layers.len(), 16, "{name}: sixteen encoder layers");
    for (index, probe) in layers.iter().enumerate() {
        let expected = floats(probe);
        assert_close(
            &output.vision().hidden_states()[index + 1].data()[..expected.len()],
            &expected,
            &format!("{name} vision layer {index}"),
            5e-5,
            5e-4,
        );
    }

    let features_first = floats(
        case.get("vision_features_first")
            .expect("vision feature probe"),
    );
    assert_close(
        &output.vision().pooled()[..features_first.len()],
        &features_first,
        &format!("{name} vision projection"),
        2e-5,
        1e-4,
    );

    let expected = floats(case.get("image_embedding").expect("image embedding"));
    let actual = output.embedding();
    assert_eq!(actual.len(), 768, "{name}: embedding dimension");
    let similarity = cosine(actual, &expected);
    assert!(
        similarity >= 0.99999,
        "{name}: cosine similarity {similarity} is below 0.99999"
    );
    let mut maximum = 0.0f32;
    for (a, e) in actual.iter().zip(&expected) {
        maximum = maximum.max((a - e).abs());
    }
    assert!(
        maximum <= 1e-4,
        "{name}: maximum absolute embedding difference {maximum} exceeds 1e-4"
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
fn image_embedding_matches_the_official_reference() {
    let Some(cache) = model_cache() else {
        return;
    };
    let embedder = load_embedder(cache);
    let golden = json::parse(IMAGE_GOLDEN).expect("image golden fixture must parse");
    let cases = golden
        .get("cases")
        .and_then(Value::as_array)
        .expect("image golden cases");
    let case = cases
        .iter()
        .find(|case| string(case, "name") == "quadrants_16x16")
        .expect("quadrants reference case");
    run_case(&embedder, case);
}

#[test]
#[ignore = "vision parity for the rotation and aspect-ratio cases"]
fn rotation_and_aspect_vision_parity() {
    let Some(cache) = model_cache() else {
        return;
    };
    let embedder = load_embedder(cache);
    let golden = json::parse(IMAGE_GOLDEN).expect("image golden fixture must parse");
    let cases = golden
        .get("cases")
        .and_then(Value::as_array)
        .expect("image golden cases");
    for name in ["wide_32x16", "quadrants_16x16_rot90"] {
        let case = cases
            .iter()
            .find(|case| string(case, "name") == name)
            .unwrap_or_else(|| panic!("missing reference case {name}"));
        run_case(&embedder, case);
    }
}

#[test]
#[ignore = "full seven-case image parity run; execute manually"]
fn all_image_embeddings_match_the_official_reference() {
    let Some(cache) = model_cache() else {
        return;
    };
    let embedder = load_embedder(cache);
    let golden = json::parse(IMAGE_GOLDEN).expect("image golden fixture must parse");
    let cases = golden
        .get("cases")
        .and_then(Value::as_array)
        .expect("image golden cases");
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
    let image =
        png::decode_file(Path::new("fixtures/media/rgb_2x2.png")).expect("RGB fixture must decode");
    let error = embedder
        .embed(&image, 0, PreprocessOptions::default())
        .expect_err("zero dimension must fail");
    assert!(matches!(
        error,
        seldon_vault_lite::error::Error::Internal(_)
    ));
    let error = embedder
        .embed(&image, 769, PreprocessOptions::default())
        .expect_err("oversized dimension must fail");
    assert!(matches!(
        error,
        seldon_vault_lite::error::Error::Internal(_)
    ));
}
