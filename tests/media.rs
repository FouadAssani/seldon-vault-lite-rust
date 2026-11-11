//! Integration tests for the native media decoders.

use seldon_vault_lite::error::Error;
use seldon_vault_lite::media::deflate::inflate_zlib;
use seldon_vault_lite::media::{Image, jpeg, png};

fn assert_close(actual: &Image, expected: &Image, tolerance: i32, label: &str) {
    assert_eq!(actual.width(), expected.width(), "{label}: width");
    assert_eq!(actual.height(), expected.height(), "{label}: height");
    assert_eq!(actual.channels(), expected.channels(), "{label}: channels");
    for (index, (actual, expected)) in actual
        .pixels()
        .iter()
        .zip(expected.pixels().iter())
        .enumerate()
    {
        let difference = i32::from(*actual) - i32::from(*expected);
        assert!(
            difference.abs() <= tolerance,
            "{label}: byte {index} differs by {difference} (actual {actual}, expected {expected})"
        );
    }
}

#[test]
fn decodes_a_committed_png_fixture_through_the_public_api() {
    let image = png::decode_file(std::path::Path::new("fixtures/media/rgb_2x2.png"))
        .expect("RGB fixture must decode");
    assert_eq!(image.width(), 2);
    assert_eq!(image.height(), 2);
    assert_eq!(image.channels(), 3);
    assert_eq!(
        image.pixels(),
        &[255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]
    );
}

#[test]
fn rejects_a_truncated_png_fixture() {
    let bytes = include_bytes!("../fixtures/media/gradient_16x16.png");
    let error = png::decode(&bytes[..bytes.len() - 1]).expect_err("truncated PNG must fail");
    assert!(matches!(error, Error::Corrupt(_)), "{error}");
}

#[test]
fn inflates_a_zlib_stream_through_the_public_api() {
    let input = [
        0x78, 0x9c, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x07, 0x00, 0x06, 0x2c, 0x02, 0x15,
    ];
    assert_eq!(inflate_zlib(&input, 16).expect("zlib stream"), b"hello");
}

#[test]
fn inflate_enforces_the_output_cap_through_the_public_api() {
    let input = [
        0x78, 0x9c, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x07, 0x00, 0x06, 0x2c, 0x02, 0x15,
    ];
    let error = inflate_zlib(&input, 4).expect_err("output cap must be enforced");
    assert!(matches!(error, Error::LimitExceeded(_)), "{error}");
}

#[test]
fn decodes_a_committed_jpeg_fixture_through_the_public_api() {
    let image = jpeg::decode_file(std::path::Path::new(
        "fixtures/media/quadrants_16x16_444.jpg",
    ))
    .expect("4:4:4 JPEG fixture must decode");
    assert_eq!(image.width(), 16);
    assert_eq!(image.height(), 16);
    assert_eq!(image.channels(), 3);
    let reference = png::decode(include_bytes!(
        "../fixtures/media/quadrants_16x16_444_ref.png"
    ))
    .expect("reference PNG must decode");
    assert_close(&image, &reference, 3, "quadrants_16x16_444");
}

#[test]
fn rejects_a_truncated_jpeg_fixture() {
    let bytes = include_bytes!("../fixtures/media/quadrants_16x16_444.jpg");
    let error = jpeg::decode(&bytes[..bytes.len() - 1]).expect_err("truncated JPEG must fail");
    assert!(matches!(error, Error::Corrupt(_)), "{error}");
}

#[test]
fn jpeg_and_png_expose_the_same_image_type() {
    let jpeg_image = jpeg::decode(include_bytes!("../fixtures/media/quadrants_16x16_444.jpg"))
        .expect("JPEG fixture");
    let png_image =
        png::decode(include_bytes!("../fixtures/media/quadrants_16x16.png")).expect("PNG fixture");
    assert_eq!(jpeg_image.width(), png_image.width());
    assert_eq!(jpeg_image.height(), png_image.height());
    assert_eq!(jpeg_image.channels(), png_image.channels());
}
