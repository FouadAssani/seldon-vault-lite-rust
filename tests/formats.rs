//! Integration tests for the M1 formats and tensor core.
//!
//! They exercise the public API exactly as the future model loader will:
//! strict JSON, Safetensors parsing, BF16 decoding, and FP32 kernels.

use seldon_vault_lite::binary::{SafetensorsFile, read_file_bounded};
use seldon_vault_lite::error::Error;
use seldon_vault_lite::json::{self, Number, Value};
use seldon_vault_lite::tensor::{DType, Shape, Tensor, kernels};

const FIXTURE: &[u8] = include_bytes!("../fixtures/formats/mini.safetensors");

#[test]
fn fixture_decodes_through_the_public_api() {
    let file = SafetensorsFile::parse(FIXTURE).expect("the fixture must parse");
    assert_eq!(file.entries().len(), 3);
    assert_eq!(
        file.fingerprint_hex(),
        "caca3b030faeb4ded52275d54c2fca0f2ee0d6b1aeeb8ab1819e0785b1f89ca7"
    );
    assert_eq!(
        file.metadata().get("version").map(String::as_str),
        Some("1")
    );

    let alpha = file.tensor("alpha").expect("alpha");
    assert_eq!(alpha.dtype(), DType::F32);
    assert_eq!(alpha.shape().dims(), &[2, 2]);
    let alpha = alpha.to_f32().expect("decode alpha");
    assert_eq!(alpha.data(), &[1.0, 2.0, 3.0, 4.0]);

    let beta = file.tensor("beta").expect("beta");
    assert_eq!(beta.dtype(), DType::Bf16);
    let beta = beta.to_f32().expect("decode beta");
    assert_eq!(beta.data(), &[0.5, -1.0, 2.0]);

    let empty = file.tensor("empty").expect("empty");
    assert_eq!(empty.to_f32().expect("decode empty").numel(), 0);

    let decoded = file.decode_all().expect("decode all tensors");
    assert_eq!(decoded.len(), 3);
    assert_eq!(decoded[0].0, "alpha");
    assert_eq!(decoded[1].0, "beta");
    assert_eq!(decoded[2].0, "empty");
}

#[test]
fn truncated_and_corrupted_fixtures_are_rejected() {
    let truncated = &FIXTURE[..FIXTURE.len() - 1];
    let error = SafetensorsFile::parse(truncated).expect_err("truncation must fail");
    assert!(matches!(error, Error::Corrupt(_)));

    let mut corrupted = FIXTURE.to_vec();
    corrupted[8] = b'[';
    let error = SafetensorsFile::parse(&corrupted).expect_err("bad header must fail");
    assert!(matches!(error, Error::Corrupt(_)));
}

#[test]
fn strict_json_and_fp32_tensors_compose() {
    let config = json::parse(r#"{"hidden_size":4,"layers":[1,2],"dtype":"F32"}"#)
        .expect("config must parse");
    let hidden_size = config
        .get("hidden_size")
        .and_then(Value::as_number)
        .and_then(Number::as_u64)
        .expect("hidden_size");
    assert_eq!(hidden_size, 4);

    let shape = Shape::new(vec![hidden_size as usize, 2]).expect("shape");
    let tensor = Tensor::filled(shape, 1.0).expect("tensor");
    assert_eq!(tensor.numel(), 8);

    let identity = Tensor::filled(Shape::new(vec![2, 2]).expect("shape"), 1.0).expect("identity");
    let product = kernels::matmul(
        tensor.data(),
        identity.data(),
        tensor.shape().dim(0).expect("rows"),
        tensor.shape().dim(1).expect("inner"),
        identity.shape().dim(1).expect("columns"),
    )
    .expect("matmul");
    assert_eq!(product.len(), 8);
    assert!(product.iter().all(|value| *value == 2.0));
}

#[test]
fn read_file_bounded_accepts_and_rejects_by_cap() {
    let mut path = std::env::temp_dir();
    path.push(format!("seldon-vault-lite-m1-{}.tmp", std::process::id()));
    std::fs::write(&path, b"seldon").expect("write the temporary file");

    let bytes = read_file_bounded(&path, 6).expect("read within the cap");
    assert_eq!(bytes, b"seldon");

    let error = read_file_bounded(&path, 5).expect_err("the cap must be enforced");
    assert!(matches!(error, Error::LimitExceeded(_)));

    std::fs::remove_file(&path).expect("remove the temporary file");
}
