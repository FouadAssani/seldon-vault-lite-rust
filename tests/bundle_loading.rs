//! Bundle-backed model loading tests for milestone M7.
//!
//! These tests build a bundle from the committed tiny text fixture and prove
//! that the source abstraction produces the same model and tokenizer as the
//! development directory. The full frozen-model bundle is covered by the
//! ignored test at the end.

use std::path::Path;

use seldon_vault_lite::bundle::{Bundle, SectionInput, SectionRole};
use seldon_vault_lite::error::Error;
use seldon_vault_lite::model::ModelFiles;
use seldon_vault_lite::model::source::{load_audio_embedder, load_image_embedder, load_text_model};
use seldon_vault_lite::model::text::TextModel;
use seldon_vault_lite::model::tokenizer::Tokenizer;

const TINY_CONFIG: &[u8] = include_bytes!("../fixtures/model/tiny_text/config.json");
const TINY_WEIGHTS: &[u8] = include_bytes!("../fixtures/model/tiny_text/model.safetensors");
const MINI_TOKENIZER: &[u8] = include_bytes!("../fixtures/tokenizer/mini_tokenizer.json");

fn tiny_model_bundle() -> Vec<u8> {
    let sections = [
        SectionInput {
            path: "config.json",
            role: SectionRole::Config,
            data: TINY_CONFIG,
        },
        SectionInput {
            path: "model.safetensors",
            role: SectionRole::Weights,
            data: TINY_WEIGHTS,
        },
    ];
    Bundle::build(&sections).expect("build tiny model bundle")
}

fn tiny_tokenizer_bundle() -> Vec<u8> {
    let sections = [SectionInput {
        path: "tokenizer.json",
        role: SectionRole::Tokenizer,
        data: MINI_TOKENIZER,
    }];
    Bundle::build(&sections).expect("build tiny tokenizer bundle")
}

#[test]
fn text_model_from_bundle_matches_the_directory() {
    let bytes = tiny_model_bundle();
    let bundle = Bundle::parse(&bytes).expect("parse tiny model bundle");
    bundle.verify().expect("verify tiny model bundle");
    let from_bundle =
        TextModel::from_files(&ModelFiles::bundle(&bundle)).expect("load tiny model from bundle");
    let from_directory = TextModel::load(Path::new("fixtures/model/tiny_text"))
        .expect("load tiny model from directory");

    assert_eq!(
        from_bundle.config().vocab_size(),
        from_directory.config().vocab_size()
    );
    assert_eq!(
        from_bundle.config().hidden_size(),
        from_directory.config().hidden_size()
    );
    assert_eq!(
        from_bundle.fingerprint_hex(),
        from_directory.fingerprint_hex()
    );
    assert_eq!(from_bundle.parameters(), from_directory.parameters());

    let ids = [2u32, 5, 6, 7, 1];
    let bundle_embedding = from_bundle.embed_ids(&ids, 8).expect("bundle embedding");
    let directory_embedding = from_directory
        .embed_ids(&ids, 8)
        .expect("directory embedding");
    assert_eq!(bundle_embedding, directory_embedding);
}

#[test]
fn tokenizer_from_bundle_matches_the_directory() {
    let bytes = tiny_tokenizer_bundle();
    let bundle = Bundle::parse(&bytes).expect("parse tiny tokenizer bundle");
    let from_bundle =
        Tokenizer::from_files(&ModelFiles::bundle(&bundle)).expect("load tokenizer from bundle");
    let from_directory = Tokenizer::load(Path::new("fixtures/tokenizer/mini_tokenizer.json"))
        .expect("load tokenizer from directory");

    assert_eq!(from_bundle.vocab_size(), from_directory.vocab_size());
    assert_eq!(from_bundle.max_token_id(), from_directory.max_token_id());
    for text in ["Hello world", "Bonjour le monde", "日本語", ""] {
        let bundle_ids = from_bundle.encode(text, true).expect("bundle encode");
        let directory_ids = from_directory.encode(text, true).expect("directory encode");
        assert_eq!(bundle_ids, directory_ids, "token ids for {text:?}");
    }
}

#[test]
fn text_model_helper_loads_from_a_bundle_file() {
    let bytes = tiny_model_bundle();
    let mut path = std::env::temp_dir();
    path.push(format!(
        "seldon-vault-lite-bundle-{}.svlm",
        std::process::id()
    ));
    std::fs::write(&path, &bytes).expect("write temporary bundle");
    let model = load_text_model(&path).expect("load model from bundle file");
    std::fs::remove_file(&path).expect("remove temporary bundle");

    let directory =
        TextModel::load(Path::new("fixtures/model/tiny_text")).expect("directory model");
    assert_eq!(model.fingerprint_hex(), directory.fingerprint_hex());
    assert_eq!(model.parameters(), directory.parameters());
}

#[test]
fn missing_weights_section_is_a_typed_error() {
    let sections = [SectionInput {
        path: "config.json",
        role: SectionRole::Config,
        data: TINY_CONFIG,
    }];
    let bytes = Bundle::build(&sections).expect("build incomplete bundle");
    let bundle = Bundle::parse(&bytes).expect("parse incomplete bundle");
    let error =
        TextModel::from_files(&ModelFiles::bundle(&bundle)).expect_err("missing weights must fail");
    assert!(matches!(error, Error::Corrupt(_)), "{error}");
}

#[test]
fn corrupt_bundle_file_is_rejected_by_the_helper() {
    let mut bytes = tiny_model_bundle();
    bytes[0] ^= 0xFF;
    let mut path = std::env::temp_dir();
    path.push(format!(
        "seldon-vault-lite-corrupt-bundle-{}.svlm",
        std::process::id()
    ));
    std::fs::write(&path, &bytes).expect("write corrupt bundle");
    let error = load_text_model(&path).expect_err("corrupt bundle must fail");
    std::fs::remove_file(&path).expect("remove corrupt bundle");
    assert!(matches!(error, Error::Corrupt(_)), "{error}");
}

#[test]
#[ignore = "writes a tiny bundle for an embedded-model build"]
fn write_tiny_bundle_for_embedded_build() {
    let path = std::env::var("SELDON_TINY_BUNDLE")
        .unwrap_or_else(|_| ".model-cache/tiny_text.svlm".to_owned());
    let bytes = tiny_model_bundle();
    std::fs::write(&path, &bytes).expect("write tiny bundle");
    println!("wrote {} bytes to {path}", bytes.len());
}

#[test]
#[ignore = "loads the frozen vision tower from the development cache"]
fn frozen_image_bundle_loads() {
    let revision = seldon_vault_lite::MODEL_REVISION;
    let bundle_path = Path::new(".model-cache").join(revision).join("model.svlm");
    if !bundle_path.is_file() {
        eprintln!("skipping: {} is not present", bundle_path.display());
        return;
    }
    let embedder =
        load_image_embedder(&bundle_path).expect("load frozen image embedder from bundle");
    assert_eq!(
        embedder.model().fingerprint_hex(),
        "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79"
    );
    assert_eq!(embedder.vision().config().num_hidden_layers(), 16);
    assert_eq!(
        embedder.tokenizer().vocab_size(),
        embedder.model().config().vocab_size()
    );
}

#[test]
#[ignore = "loads the frozen audio tower from the development cache"]
fn frozen_audio_bundle_loads() {
    let revision = seldon_vault_lite::MODEL_REVISION;
    let bundle_path = Path::new(".model-cache").join(revision).join("model.svlm");
    if !bundle_path.is_file() {
        eprintln!("skipping: {} is not present", bundle_path.display());
        return;
    }
    let embedder =
        load_audio_embedder(&bundle_path).expect("load frozen audio embedder from bundle");
    assert_eq!(
        embedder.model().fingerprint_hex(),
        "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79"
    );
    assert_eq!(embedder.audio().config().num_hidden_layers(), 12);
    assert_eq!(
        embedder.tokenizer().map(|tokenizer| tokenizer.vocab_size()),
        Some(embedder.model().config().vocab_size())
    );
}

#[test]
#[ignore = "loads the frozen model bundle from the development cache"]
fn frozen_model_bundle_matches_the_directory() {
    let revision = seldon_vault_lite::MODEL_REVISION;
    let bundle_path = Path::new(".model-cache").join(revision).join("model.svlm");
    let model_dir = Path::new(".model-cache").join(revision);
    if !bundle_path.is_file() {
        eprintln!("skipping: {} is not present", bundle_path.display());
        return;
    }
    let from_bundle = seldon_vault_lite::model::load_text_model(&bundle_path)
        .expect("load frozen model from bundle");
    let from_directory = TextModel::load(&model_dir).expect("load frozen model from directory");
    assert_eq!(
        from_bundle.fingerprint_hex(),
        from_directory.fingerprint_hex()
    );
    assert_eq!(from_bundle.parameters(), from_directory.parameters());
    let ids = [2u32, 5, 6, 7, 1];
    let bundle_embedding = from_bundle.embed_ids(&ids, 768).expect("bundle embedding");
    let directory_embedding = from_directory
        .embed_ids(&ids, 768)
        .expect("directory embedding");
    assert_eq!(bundle_embedding, directory_embedding);
}
