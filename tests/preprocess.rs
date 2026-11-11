//! Integration tests for the official image preprocessing pipeline.

use std::path::Path;

use seldon_vault_lite::binary::sha256::sha256_hex;
use seldon_vault_lite::json;
use seldon_vault_lite::media::png;
use seldon_vault_lite::media::preprocess::{
    PATCH_VALUES, PreprocessOptions, aspect_ratio_preserving_size, preprocess,
};

fn number(value: &json::Value, key: &str) -> u64 {
    value
        .get(key)
        .and_then(|value| value.as_number())
        .and_then(|number| number.as_u64())
        .unwrap_or_else(|| panic!("missing integer field {key}"))
}

fn string<'a>(value: &'a json::Value, key: &str) -> &'a str {
    value
        .get(key)
        .and_then(|value| value.as_str())
        .unwrap_or_else(|| panic!("missing string field {key}"))
}

fn hash_f32(values: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    sha256_hex(&bytes)
}

fn hash_positions(values: &[[i32; 2]]) -> String {
    // The reference stores positions as int64; widen the product's compact
    // int32 representation before hashing so the fingerprints are comparable.
    let mut bytes = Vec::with_capacity(values.len() * 16);
    for position in values {
        bytes.extend_from_slice(&i64::from(position[0]).to_le_bytes());
        bytes.extend_from_slice(&i64::from(position[1]).to_le_bytes());
    }
    sha256_hex(&bytes)
}

#[test]
fn reproduces_the_official_preprocessing_fixtures() {
    let document = json::parse(include_str!("../fixtures/media/preprocess_golden.json"))
        .expect("preprocessing golden must parse");
    let cases = document
        .get("cases")
        .and_then(|value| value.as_array())
        .expect("preprocessing golden must contain cases");
    assert!(!cases.is_empty());

    for case in cases {
        let name = string(case, "name");
        let source = string(case, "source");
        let image = png::decode_file(Path::new(source))
            .unwrap_or_else(|error| panic!("{name}: source PNG must decode: {error}"));
        let options = PreprocessOptions::new(number(case, "max_soft_tokens") as usize)
            .expect("golden budget must be supported");
        let patches = preprocess(&image, options)
            .unwrap_or_else(|error| panic!("{name}: preprocessing failed: {error}"));

        assert_eq!(
            patches.resized_size(),
            (
                number(case, "target_width") as usize,
                number(case, "target_height") as usize
            ),
            "{name}: resized size"
        );
        assert_eq!(
            patches.patch_grid(),
            (
                number(case, "patch_grid_width") as usize,
                number(case, "patch_grid_height") as usize
            ),
            "{name}: patch grid"
        );
        assert_eq!(
            patches.valid_patches(),
            number(case, "valid_patches") as usize,
            "{name}: valid patches"
        );
        assert_eq!(
            patches.soft_tokens(),
            number(case, "soft_tokens") as usize,
            "{name}: soft tokens"
        );
        assert_eq!(
            patches.pixel_values().len(),
            patches.max_patches() * PATCH_VALUES,
            "{name}: padded tensor length"
        );
        assert_eq!(
            hash_f32(patches.pixel_values()),
            string(case, "pixel_values_sha256"),
            "{name}: pixel values"
        );
        assert_eq!(
            hash_positions(patches.positions()),
            string(case, "positions_sha256"),
            "{name}: positions"
        );

        let first = case
            .get("first_patch")
            .and_then(|value| value.as_array())
            .expect("first patch");
        for (index, expected) in first.iter().enumerate() {
            let expected = expected
                .as_number()
                .expect("patch value must be a number")
                .as_f64();
            let actual = f64::from(patches.pixel_values()[index]);
            assert!(
                (actual - expected).abs() <= 1e-6,
                "{name}: first patch value {index} differs: {actual} vs {expected}"
            );
        }

        let first = case
            .get("first_position")
            .and_then(|value| value.as_array())
            .expect("first position");
        let first_position = [
            first[0]
                .as_number()
                .and_then(|number| number.as_i64())
                .expect("first position x") as i32,
            first[1]
                .as_number()
                .and_then(|number| number.as_i64())
                .expect("first position y") as i32,
        ];
        assert_eq!(
            patches.positions()[0],
            first_position,
            "{name}: first position"
        );
    }
}

#[test]
fn computes_the_official_aspect_ratio_preserving_size() {
    assert_eq!(
        aspect_ratio_preserving_size(2, 2, 16, 2520, 3).expect("2x2"),
        (768, 768)
    );
    assert_eq!(
        aspect_ratio_preserving_size(16, 16, 16, 630, 3).expect("16x16"),
        (384, 384)
    );
    assert_eq!(
        aspect_ratio_preserving_size(100, 50, 16, 630, 3).expect("2:1"),
        (528, 240)
    );
}

#[test]
fn rejects_an_unsupported_visual_token_budget() {
    let error = PreprocessOptions::new(100).expect_err("100 is not an official budget");
    assert!(matches!(error, seldon_vault_lite::error::Error::Usage(_)));
}
