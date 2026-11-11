//! Integration tests for the native media decoders.

use seldon_vault_lite::error::Error;
use seldon_vault_lite::media::deflate::inflate_zlib;
use seldon_vault_lite::media::png::{decode, decode_file};

#[test]
fn decodes_a_committed_png_fixture_through_the_public_api() {
    let image = decode_file(std::path::Path::new("fixtures/media/rgb_2x2.png"))
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
    let error = decode(&bytes[..bytes.len() - 1]).expect_err("truncated PNG must fail");
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
