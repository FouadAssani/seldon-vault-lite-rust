//! Tokenizer parity tests against the frozen official tokenizer.
//!
//! The reduced tokenizer fixture reproduces the full tokenizer on the golden
//! corpus. The full 32 MiB tokenizer.json is a development asset outside Git;
//! when it is present in .model-cache, this test also verifies parity against
//! it and its SHA-256 manifest entry.

use std::path::Path;

use seldon_vault_lite::binary::sha256::sha256_hex;
use seldon_vault_lite::json::{self, Number, Value};
use seldon_vault_lite::model::prompt;
use seldon_vault_lite::model::tokenizer::Tokenizer;

const GOLDEN: &str = include_str!("../fixtures/tokenizer/golden.json");
const MINI: &str = include_str!("../fixtures/tokenizer/mini_tokenizer.json");
const MANIFEST: &str = include_str!("../fixtures/model/manifest.json");
const REVISION: &str = "f9972ac22e696a105c482202646979e33e4b5a1e";

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

fn string<'a>(value: &'a Value, key: &str) -> &'a str {
    value
        .get(key)
        .and_then(Value::as_str)
        .expect("a string field")
}

fn unsigned(value: &Value, key: &str) -> u64 {
    value
        .get(key)
        .and_then(Value::as_number)
        .and_then(Number::as_u64)
        .expect("an unsigned field")
}

#[test]
fn mini_tokenizer_reproduces_the_golden_corpus() {
    let tokenizer = Tokenizer::from_json_str(MINI).expect("the reduced tokenizer must load");
    let golden = json::parse(GOLDEN).expect("the golden fixture must parse");
    let cases = golden
        .get("cases")
        .and_then(Value::as_array)
        .expect("the golden cases");

    for case in cases {
        let name = string(case, "name");
        let text = string(case, "text");
        let expected_ids = ids(case.get("ids").expect("ids"));
        let expected_ids_with_special =
            ids(case.get("ids_with_special").expect("ids_with_special"));
        let expected_decoded = string(case, "decoded");

        assert_eq!(
            tokenizer.encode(text, false).expect("encoding"),
            expected_ids,
            "plain ids for {name}"
        );
        assert_eq!(
            tokenizer.encode(text, true).expect("encoding"),
            expected_ids_with_special,
            "ids with special tokens for {name}"
        );
        assert_eq!(
            tokenizer.decode(&expected_ids, true).expect("decoding"),
            expected_decoded,
            "decoded text for {name}"
        );
        assert_eq!(
            tokenizer
                .decode(&expected_ids_with_special, true)
                .expect("decoding"),
            expected_decoded,
            "decoded text with special tokens for {name}"
        );
    }
}

#[test]
fn manifest_records_the_frozen_model_and_fixtures() {
    let manifest = json::parse(MANIFEST).expect("the manifest must parse");
    assert_eq!(string(&manifest, "model"), "google/embeddinggemma-2");
    assert_eq!(string(&manifest, "revision"), REVISION);
    assert_eq!(unsigned(&manifest, "schema_version"), 1);

    let tokenizer = manifest.get("tokenizer").expect("tokenizer section");
    assert_eq!(unsigned(tokenizer, "vocab_size"), 262_144);
    assert_eq!(unsigned(tokenizer, "merges"), 514_906);
    assert_eq!(
        tokenizer.get("byte_fallback").and_then(Value::as_bool),
        Some(true)
    );
    assert_eq!(
        tokenizer
            .get("special_ids")
            .and_then(|value| value.get("bos"))
            .and_then(Value::as_number)
            .and_then(Number::as_u64),
        Some(2)
    );
    assert_eq!(
        tokenizer
            .get("special_ids")
            .and_then(|value| value.get("eos"))
            .and_then(Value::as_number)
            .and_then(Number::as_u64),
        Some(1)
    );
    assert_eq!(
        tokenizer
            .get("special_ids")
            .and_then(|value| value.get("unk"))
            .and_then(Value::as_number)
            .and_then(Number::as_u64),
        Some(3)
    );

    let text_config = manifest.get("text_config").expect("text config");
    assert_eq!(unsigned(text_config, "hidden_size"), 512);
    assert_eq!(unsigned(text_config, "embedding_dim"), 768);
    assert_eq!(unsigned(text_config, "num_hidden_layers"), 24);
    assert_eq!(unsigned(text_config, "sliding_window"), 512);
    assert_eq!(unsigned(text_config, "vocab_size"), 262_144);

    let prompts = manifest.get("prompts").expect("prompts");
    assert_eq!(string(prompts, "SearchQuery"), prompt::SEARCH_QUERY);
    assert_eq!(string(prompts, "Document"), prompt::DOCUMENT);
    assert_eq!(string(prompts, "Classification"), prompt::CLASSIFICATION);
    assert_eq!(string(prompts, "Clustering"), prompt::CLUSTERING);
    assert_eq!(string(prompts, "CodeRetrieval"), prompt::CODE_RETRIEVAL);
    assert_eq!(string(prompts, "FactChecking"), prompt::FACT_CHECKING);
    assert_eq!(
        string(prompts, "SentenceSimilarity"),
        prompt::SENTENCE_SIMILARITY
    );
    assert_eq!(
        string(prompts, "QuestionAnswering"),
        prompt::QUESTION_ANSWERING
    );

    let fixtures = manifest.get("fixtures").expect("fixtures");
    assert_eq!(
        string(fixtures.get("golden").expect("golden fixture"), "sha256"),
        sha256_hex(GOLDEN.as_bytes())
    );
    assert_eq!(
        string(
            fixtures.get("mini_tokenizer").expect("mini fixture"),
            "sha256"
        ),
        sha256_hex(MINI.as_bytes())
    );
}

#[test]
fn full_tokenizer_parity_when_the_model_cache_is_present() {
    let manifest = json::parse(MANIFEST).expect("the manifest must parse");
    let expected_sha = string(
        manifest.get("tokenizer").expect("tokenizer section"),
        "sha256",
    );
    let path = Path::new(".model-cache")
        .join(REVISION)
        .join("tokenizer.json");
    if !path.is_file() {
        eprintln!(
            "skipping full tokenizer parity: {} is not present",
            path.display()
        );
        return;
    }

    let bytes = std::fs::read(&path).expect("read the frozen tokenizer");
    assert_eq!(
        sha256_hex(&bytes),
        expected_sha,
        "the frozen tokenizer SHA-256 must match the manifest"
    );

    let full = Tokenizer::load(&path).expect("load the full tokenizer");
    assert_eq!(full.vocab_size(), 262_144);
    assert_eq!(full.max_token_id(), 262_143);
    assert_eq!(full.bos_id(), 2);
    assert_eq!(full.eos_id(), 1);

    let golden = json::parse(GOLDEN).expect("the golden fixture must parse");
    let cases = golden
        .get("cases")
        .and_then(Value::as_array)
        .expect("the golden cases");
    for case in cases {
        let name = string(case, "name");
        let text = string(case, "text");
        let expected_ids = ids(case.get("ids").expect("ids"));
        let expected_ids_with_special =
            ids(case.get("ids_with_special").expect("ids_with_special"));
        assert_eq!(
            full.encode(text, false).expect("encoding"),
            expected_ids,
            "full tokenizer plain ids for {name}"
        );
        assert_eq!(
            full.encode(text, true).expect("encoding"),
            expected_ids_with_special,
            "full tokenizer ids with special tokens for {name}"
        );
        assert_eq!(
            full.decode(&expected_ids, true).expect("decoding"),
            string(case, "decoded"),
            "full tokenizer decoded text for {name}"
        );
    }
}
