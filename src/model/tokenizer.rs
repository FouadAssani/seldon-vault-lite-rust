//! Exact HuggingFace BPE tokenizer for the frozen EmbeddingGemma 2 model.
//!
//! The tokenizer is specialized for the model's tokenizer.json:
//!
//! - model type BPE, byte fallback enabled, fuse_unk enabled;
//! - normalizer replaces spaces with the SentencePiece marker;
//! - pre-tokenizer splits on spaces with MergedWithPrevious;
//! - post-processor adds <bos> and <eos>;
//! - decoder replaces the marker with spaces and applies byte fallback.
//!
//! The implementation is deliberately strict. A tokenizer document that does
//! not match this pipeline is rejected as unsupported rather than silently
//! tokenized differently. Every allocation derived from the document is
//! bounded, and every token id is checked before use.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::binary::read_file_bounded;
use crate::error::{Error, Result};
use crate::json::{self, Limits, Number, Value};

/// Maximum size of a tokenizer.json file in bytes.
pub const MAX_TOKENIZER_FILE_BYTES: u64 = 64 * 1024 * 1024;

/// Maximum number of vocabulary entries.
pub const MAX_VOCAB_ENTRIES: usize = 1_000_000;

/// Maximum number of BPE merges.
pub const MAX_MERGES: usize = 1_000_000;

/// Maximum number of added tokens.
pub const MAX_ADDED_TOKENS: usize = 10_000;

/// Maximum input size accepted by encode, in bytes.
pub const MAX_INPUT_BYTES: usize = 1024 * 1024;

/// Maximum number of tokens produced by one encode call.
///
/// The model declares 262,144 maximum positions. Tokenizing more than that
/// would produce an input the model cannot consume.
pub const MAX_TOKENS: usize = 262_144;

/// SentencePiece space marker used by the model tokenizer.
const SPACE_MARKER: char = '▁';

/// One BPE merge.
#[derive(Debug, Clone, Copy)]
struct Merge {
    /// Rank in the merge table; lower ranks are applied first.
    rank: u32,
    /// Token id produced by the merge.
    merged_id: u32,
}

/// One added token, matched in raw text before normalization.
#[derive(Debug, Clone)]
struct AddedToken {
    /// Raw text of the token.
    text: String,
    /// Token id.
    id: u32,
}

/// Frozen-model BPE tokenizer.
#[derive(Debug)]
pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    char_ids: HashMap<char, u32>,
    byte_ids: [u32; 256],
    merges: HashMap<(u32, u32), Merge>,
    added_tokens: Vec<AddedToken>,
    special_ids: HashSet<u32>,
    unk_id: u32,
    bos_id: u32,
    eos_id: u32,
    pad_id: Option<u32>,
    max_token_id: u32,
}

impl Tokenizer {
    /// Loads a tokenizer.json file, bounded by the tokenizer file cap.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = read_file_bounded(path, MAX_TOKENIZER_FILE_BYTES)?;
        let text = std::str::from_utf8(&bytes)
            .map_err(|_| Error::corrupt("tokenizer file is not valid UTF-8"))?;
        Self::from_json_str(text)
    }

    /// Parses a tokenizer JSON document from a string.
    pub fn from_json_str(text: &str) -> Result<Self> {
        let root = json::parse_with_limits(text, &json_limits())?;
        Self::from_json(&root)
    }

    /// Builds a tokenizer from an already parsed JSON value.
    ///
    /// # Errors
    ///
    /// Returns Error::Corrupt for an inconsistent document and
    /// Error::Unsupported for a pipeline this implementation does not cover.
    pub fn from_json(root: &Value) -> Result<Self> {
        let model = object(
            root.get("model")
                .ok_or_else(|| Error::corrupt("tokenizer is missing model"))?,
            "model",
        )?;

        let model_type = string(
            model
                .get("type")
                .ok_or_else(|| Error::corrupt("tokenizer model is missing type"))?,
            "model.type",
        )?;
        if model_type != "BPE" {
            return Err(Error::unsupported(format!(
                "tokenizer model {model_type}; only BPE is implemented"
            )));
        }

        let byte_fallback = boolean(
            model
                .get("byte_fallback")
                .ok_or_else(|| Error::corrupt("tokenizer model is missing byte_fallback"))?,
            "model.byte_fallback",
        )?;
        if !byte_fallback {
            return Err(Error::unsupported(
                "tokenizer without byte fallback is not supported",
            ));
        }
        if !boolean(
            model
                .get("fuse_unk")
                .ok_or_else(|| Error::corrupt("tokenizer model is missing fuse_unk"))?,
            "model.fuse_unk",
        )? {
            return Err(Error::unsupported(
                "tokenizer with fuse_unk disabled is not supported",
            ));
        }
        if boolean(
            model
                .get("ignore_merges")
                .ok_or_else(|| Error::corrupt("tokenizer model is missing ignore_merges"))?,
            "model.ignore_merges",
        )? {
            return Err(Error::unsupported(
                "tokenizer with ignore_merges enabled is not supported",
            ));
        }
        expect_null(
            model.get("continuing_subword_prefix").ok_or_else(|| {
                Error::corrupt("tokenizer model is missing continuing_subword_prefix")
            })?,
            "model.continuing_subword_prefix",
        )?;
        expect_null(
            model
                .get("end_of_word_suffix")
                .ok_or_else(|| Error::corrupt("tokenizer model is missing end_of_word_suffix"))?,
            "model.end_of_word_suffix",
        )?;
        expect_null(
            model
                .get("dropout")
                .ok_or_else(|| Error::corrupt("tokenizer model is missing dropout"))?,
            "model.dropout",
        )?;

        let vocab_value = object(
            model
                .get("vocab")
                .ok_or_else(|| Error::corrupt("tokenizer model is missing vocab"))?,
            "model.vocab",
        )?;
        if vocab_value.len() > MAX_VOCAB_ENTRIES {
            return Err(Error::limit_exceeded(format!(
                "tokenizer vocabulary has {} entries, above the {MAX_VOCAB_ENTRIES}-entry cap",
                vocab_value.len()
            )));
        }

        let mut vocab = HashMap::with_capacity(vocab_value.len());
        let mut id_to_token = HashMap::with_capacity(vocab_value.len());
        for (token, id_value) in vocab_value {
            let id = unsigned(id_value, "model.vocab id")?;
            let id = u32::try_from(id)
                .map_err(|_| Error::limit_exceeded("tokenizer vocabulary id does not fit u32"))?;
            if vocab.insert(token.clone(), id).is_some() {
                return Err(Error::corrupt("duplicate tokenizer vocabulary token"));
            }
            if id_to_token.insert(id, token.clone()).is_some() {
                return Err(Error::corrupt("duplicate tokenizer vocabulary id"));
            }
        }

        let mut char_ids: HashMap<char, u32> = HashMap::new();
        for (token, id) in &vocab {
            let mut chars = token.chars();
            if let Some(character) = chars.next() {
                if chars.next().is_none() {
                    char_ids.insert(character, *id);
                }
            }
        }
        if !char_ids.contains_key(&SPACE_MARKER) {
            return Err(Error::corrupt(
                "tokenizer vocabulary is missing the space marker token",
            ));
        }

        let mut byte_ids = [0u32; 256];
        for (byte, slot) in byte_ids.iter_mut().enumerate() {
            let token = format!("<0x{byte:02X}>");
            *slot = *vocab.get(&token).ok_or_else(|| {
                Error::corrupt(format!(
                    "tokenizer vocabulary is missing byte token {token}"
                ))
            })?;
        }

        let merges_value = array(
            model
                .get("merges")
                .ok_or_else(|| Error::corrupt("tokenizer model is missing merges"))?,
            "model.merges",
        )?;
        if merges_value.len() > MAX_MERGES {
            return Err(Error::limit_exceeded(format!(
                "tokenizer has {} merges, above the {MAX_MERGES}-merge cap",
                merges_value.len()
            )));
        }

        let mut merges: HashMap<(u32, u32), Merge> = HashMap::with_capacity(merges_value.len());
        for (rank, merge_value) in merges_value.iter().enumerate() {
            let pair = array(merge_value, "model.merges entry")?;
            if pair.len() != 2 {
                return Err(Error::corrupt(
                    "tokenizer merge must contain exactly two strings",
                ));
            }
            let left = string(&pair[0], "merge left")?;
            let right = string(&pair[1], "merge right")?;
            let left_id = *vocab.get(left).ok_or_else(|| {
                Error::corrupt(format!("merge left {left:?} is not in the vocabulary"))
            })?;
            let right_id = *vocab.get(right).ok_or_else(|| {
                Error::corrupt(format!("merge right {right:?} is not in the vocabulary"))
            })?;
            let merged = format!("{left}{right}");
            let merged_id = *vocab.get(&merged).ok_or_else(|| {
                Error::corrupt(format!("merge result {merged:?} is not in the vocabulary"))
            })?;
            let rank = u32::try_from(rank)
                .map_err(|_| Error::limit_exceeded("tokenizer merge rank does not fit u32"))?;
            if merges
                .insert((left_id, right_id), Merge { rank, merged_id })
                .is_some()
            {
                return Err(Error::corrupt("duplicate tokenizer merge pair"));
            }
        }

        let unk_token = string(
            model
                .get("unk_token")
                .ok_or_else(|| Error::corrupt("tokenizer model is missing unk_token"))?,
            "model.unk_token",
        )?;
        let unk_id = *vocab.get(unk_token).ok_or_else(|| {
            Error::corrupt(format!(
                "unknown token {unk_token:?} is not in the vocabulary"
            ))
        })?;

        let added_value = array(
            root.get("added_tokens")
                .ok_or_else(|| Error::corrupt("tokenizer is missing added_tokens"))?,
            "added_tokens",
        )?;
        if added_value.len() > MAX_ADDED_TOKENS {
            return Err(Error::limit_exceeded(format!(
                "tokenizer has {} added tokens, above the {MAX_ADDED_TOKENS}-token cap",
                added_value.len()
            )));
        }

        let mut added_tokens = Vec::with_capacity(added_value.len());
        let mut special_ids = HashSet::new();
        let mut seen_added_ids = HashSet::new();
        for value in added_value {
            let descriptor = object(value, "added token")?;
            let text = string(
                descriptor
                    .get("content")
                    .ok_or_else(|| Error::corrupt("added token is missing content"))?,
                "added token content",
            )?;
            if text.is_empty() {
                return Err(Error::corrupt("added token content must not be empty"));
            }
            let id = unsigned(
                descriptor
                    .get("id")
                    .ok_or_else(|| Error::corrupt("added token is missing id"))?,
                "added token id",
            )?;
            let id = u32::try_from(id)
                .map_err(|_| Error::limit_exceeded("added token id does not fit u32"))?;
            if !seen_added_ids.insert(id) {
                return Err(Error::corrupt("duplicate added token id"));
            }
            if !id_to_token.contains_key(&id) {
                return Err(Error::corrupt(format!(
                    "added token {text:?} has id {id} outside the vocabulary"
                )));
            }
            let special = boolean(
                descriptor
                    .get("special")
                    .ok_or_else(|| Error::corrupt("added token is missing special"))?,
                "added token special",
            )?;
            if special {
                special_ids.insert(id);
            }
            added_tokens.push(AddedToken {
                text: text.to_owned(),
                id,
            });
        }
        added_tokens.sort_by(|left, right| {
            right
                .text
                .len()
                .cmp(&left.text.len())
                .then_with(|| left.id.cmp(&right.id))
        });

        validate_normalizer(
            root.get("normalizer")
                .ok_or_else(|| Error::corrupt("tokenizer is missing normalizer"))?,
        )?;
        validate_pre_tokenizer(
            root.get("pre_tokenizer")
                .ok_or_else(|| Error::corrupt("tokenizer is missing pre_tokenizer"))?,
        )?;
        let (bos_id, eos_id) = validate_post_processor(
            root.get("post_processor")
                .ok_or_else(|| Error::corrupt("tokenizer is missing post_processor"))?,
            &vocab,
        )?;
        validate_decoder(
            root.get("decoder")
                .ok_or_else(|| Error::corrupt("tokenizer is missing decoder"))?,
        )?;

        let pad_id = vocab.get("<pad>").copied();
        let max_token_id = id_to_token.keys().copied().max().unwrap_or(0);

        Ok(Self {
            vocab,
            id_to_token,
            char_ids,
            byte_ids,
            merges,
            added_tokens,
            special_ids,
            unk_id,
            bos_id,
            eos_id,
            pad_id,
            max_token_id,
        })
    }

    /// Encodes text to token ids.
    ///
    /// When add_special_tokens is true, <bos> is prepended and <eos> appended,
    /// matching the model post-processor. Added tokens are matched in the raw
    /// text before normalization, as in the official tokenizer.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        if text.len() > MAX_INPUT_BYTES {
            return Err(Error::limit_exceeded(format!(
                "tokenizer input is {} bytes, above the {MAX_INPUT_BYTES}-byte cap",
                text.len()
            )));
        }

        let mut ids = Vec::with_capacity(16);
        if add_special_tokens {
            ids.push(self.bos_id);
        }

        let mut segment = String::new();
        let mut index = 0;
        while index < text.len() {
            let rest = &text[index..];
            if let Some(added) = self.match_added_token(rest) {
                if !segment.is_empty() {
                    self.encode_segment(&segment, &mut ids)?;
                    segment.clear();
                }
                ids.push(added.id);
                self.check_token_limit(ids.len())?;
                index += added.text.len();
            } else {
                let character = rest
                    .chars()
                    .next()
                    .ok_or_else(|| Error::internal("tokenizer input ended mid-character"))?;
                segment.push(character);
                index += character.len_utf8();
            }
        }

        if !segment.is_empty() {
            self.encode_segment(&segment, &mut ids)?;
        }
        if add_special_tokens {
            ids.push(self.eos_id);
        }
        self.check_token_limit(ids.len())?;
        Ok(ids)
    }

    /// Decodes token ids to text.
    ///
    /// Special tokens are skipped when skip_special_tokens is true. Byte
    /// fallback sequences are reassembled, and the space marker is replaced by
    /// a space. Invalid UTF-8 is rejected rather than silently replaced.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            let token = self
                .id_to_token
                .get(&id)
                .ok_or_else(|| Error::corrupt(format!("cannot decode unknown token id {id}")))?;
            if skip_special_tokens && self.special_ids.contains(&id) {
                continue;
            }
            if let Some(byte) = parse_byte_token(token) {
                bytes.push(byte);
            } else {
                for character in token.chars() {
                    if character == SPACE_MARKER {
                        bytes.push(b' ');
                    } else {
                        let mut buffer = [0u8; 4];
                        bytes.extend_from_slice(character.encode_utf8(&mut buffer).as_bytes());
                    }
                }
            }
        }
        String::from_utf8(bytes)
            .map_err(|_| Error::corrupt("decoded token sequence is not valid UTF-8"))
    }

    /// Looks up a token id by its string.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    /// Looks up a token string by its id.
    pub fn token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(String::as_str)
    }

    /// Number of vocabulary entries.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Largest token id present in the vocabulary.
    pub fn max_token_id(&self) -> u32 {
        self.max_token_id
    }

    /// Beginning-of-sequence token id.
    pub fn bos_id(&self) -> u32 {
        self.bos_id
    }

    /// End-of-sequence token id.
    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }

    /// Unknown token id.
    pub fn unk_id(&self) -> u32 {
        self.unk_id
    }

    /// Padding token id, when the vocabulary defines one.
    pub fn pad_id(&self) -> Option<u32> {
        self.pad_id
    }

    /// Returns true when the id is a special added token.
    pub fn is_special_id(&self, id: u32) -> bool {
        self.special_ids.contains(&id)
    }

    fn match_added_token(&self, rest: &str) -> Option<&AddedToken> {
        self.added_tokens
            .iter()
            .find(|token| rest.starts_with(&token.text))
    }

    fn encode_segment(&self, segment: &str, ids: &mut Vec<u32>) -> Result<()> {
        let mut symbols = Vec::with_capacity(segment.len());
        for character in segment.chars() {
            let normalized = if character == ' ' {
                SPACE_MARKER
            } else {
                character
            };
            if let Some(id) = self.char_ids.get(&normalized) {
                symbols.push(*id);
            } else {
                let mut buffer = [0u8; 4];
                for byte in normalized.encode_utf8(&mut buffer).as_bytes() {
                    symbols.push(self.byte_ids[*byte as usize]);
                }
            }
        }

        if symbols.len() > MAX_TOKENS {
            return Err(Error::limit_exceeded(format!(
                "tokenization exceeds the {MAX_TOKENS}-token cap"
            )));
        }

        while let Some((left, right, merged_id)) = self.best_merge(&symbols) {
            let mut next = Vec::with_capacity(symbols.len());
            let mut index = 0;
            while index < symbols.len() {
                if index + 1 < symbols.len()
                    && symbols[index] == left
                    && symbols[index + 1] == right
                {
                    next.push(merged_id);
                    index += 2;
                } else {
                    next.push(symbols[index]);
                    index += 1;
                }
            }
            symbols = next;
        }

        ids.extend_from_slice(&symbols);
        self.check_token_limit(ids.len())
    }

    fn best_merge(&self, symbols: &[u32]) -> Option<(u32, u32, u32)> {
        let mut best: Option<(u32, u32, u32, u32)> = None;
        for window in symbols.windows(2) {
            if let Some(merge) = self.merges.get(&(window[0], window[1])) {
                if best.is_none_or(|current| merge.rank < current.0) {
                    best = Some((merge.rank, window[0], window[1], merge.merged_id));
                }
            }
        }
        best.map(|(_, left, right, merged_id)| (left, right, merged_id))
    }

    fn check_token_limit(&self, count: usize) -> Result<()> {
        if count > MAX_TOKENS {
            return Err(Error::limit_exceeded(format!(
                "tokenization exceeds the {MAX_TOKENS}-token cap"
            )));
        }
        Ok(())
    }
}

/// JSON limits used when loading tokenizer.json.
pub fn json_limits() -> Limits {
    Limits {
        max_input_bytes: MAX_TOKENIZER_FILE_BYTES as usize,
        max_depth: 64,
        max_string_bytes: 1024 * 1024,
        max_container_entries: 600_000,
        max_total_values: 3_000_000,
    }
}

/// Validates the model normalizer.
fn validate_normalizer(value: &Value) -> Result<()> {
    let normalizer = object(value, "normalizer")?;
    if string(
        normalizer
            .get("type")
            .ok_or_else(|| Error::corrupt("normalizer is missing type"))?,
        "normalizer.type",
    )? != "Replace"
    {
        return Err(Error::unsupported(
            "tokenizer normalizer is not a Replace normalizer",
        ));
    }
    let pattern = object(
        normalizer
            .get("pattern")
            .ok_or_else(|| Error::corrupt("normalizer is missing pattern"))?,
        "normalizer.pattern",
    )?;
    if string(
        pattern
            .get("String")
            .ok_or_else(|| Error::corrupt("normalizer pattern is missing String"))?,
        "normalizer.pattern.String",
    )? != " "
    {
        return Err(Error::unsupported(
            "tokenizer normalizer does not replace a space",
        ));
    }
    if string(
        normalizer
            .get("content")
            .ok_or_else(|| Error::corrupt("normalizer is missing content"))?,
        "normalizer.content",
    )? != "▁"
    {
        return Err(Error::unsupported(
            "tokenizer normalizer does not use the expected space marker",
        ));
    }
    Ok(())
}

/// Validates the model pre-tokenizer.
fn validate_pre_tokenizer(value: &Value) -> Result<()> {
    let pre_tokenizer = object(value, "pre_tokenizer")?;
    if string(
        pre_tokenizer
            .get("type")
            .ok_or_else(|| Error::corrupt("pre_tokenizer is missing type"))?,
        "pre_tokenizer.type",
    )? != "Split"
    {
        return Err(Error::unsupported(
            "tokenizer pre_tokenizer is not a Split pre-tokenizer",
        ));
    }
    let pattern = object(
        pre_tokenizer
            .get("pattern")
            .ok_or_else(|| Error::corrupt("pre_tokenizer is missing pattern"))?,
        "pre_tokenizer.pattern",
    )?;
    if string(
        pattern
            .get("String")
            .ok_or_else(|| Error::corrupt("pre_tokenizer pattern is missing String"))?,
        "pre_tokenizer.pattern.String",
    )? != " "
    {
        return Err(Error::unsupported(
            "tokenizer pre_tokenizer does not split on a space",
        ));
    }
    if string(
        pre_tokenizer
            .get("behavior")
            .ok_or_else(|| Error::corrupt("pre_tokenizer is missing behavior"))?,
        "pre_tokenizer.behavior",
    )? != "MergedWithPrevious"
    {
        return Err(Error::unsupported(
            "tokenizer pre_tokenizer does not use MergedWithPrevious",
        ));
    }
    if boolean(
        pre_tokenizer
            .get("invert")
            .ok_or_else(|| Error::corrupt("pre_tokenizer is missing invert"))?,
        "pre_tokenizer.invert",
    )? {
        return Err(Error::unsupported("tokenizer pre_tokenizer is inverted"));
    }
    Ok(())
}

/// Validates the post-processor and returns the BOS and EOS ids.
fn validate_post_processor(value: &Value, vocab: &HashMap<String, u32>) -> Result<(u32, u32)> {
    let post_processor = object(value, "post_processor")?;
    if string(
        post_processor
            .get("type")
            .ok_or_else(|| Error::corrupt("post_processor is missing type"))?,
        "post_processor.type",
    )? != "TemplateProcessing"
    {
        return Err(Error::unsupported(
            "tokenizer post_processor is not TemplateProcessing",
        ));
    }

    let single = array(
        post_processor
            .get("single")
            .ok_or_else(|| Error::corrupt("post_processor is missing single"))?,
        "post_processor.single",
    )?;
    if single.len() != 3 {
        return Err(Error::unsupported(
            "tokenizer post_processor single template must have three parts",
        ));
    }

    let first = parse_special_token(&single[0], "post_processor.single[0]")?;
    if first != "<bos>" {
        return Err(Error::unsupported(
            "tokenizer post_processor does not prepend <bos>",
        ));
    }
    let second = object(&single[1], "post_processor.single[1]")?;
    if second.get("Sequence").is_none() {
        return Err(Error::unsupported(
            "tokenizer post_processor middle part is not a Sequence",
        ));
    }
    let third = parse_special_token(&single[2], "post_processor.single[2]")?;
    if third != "<eos>" {
        return Err(Error::unsupported(
            "tokenizer post_processor does not append <eos>",
        ));
    }

    let bos_id = *vocab
        .get("<bos>")
        .ok_or_else(|| Error::corrupt("tokenizer vocabulary is missing <bos>"))?;
    let eos_id = *vocab
        .get("<eos>")
        .ok_or_else(|| Error::corrupt("tokenizer vocabulary is missing <eos>"))?;
    Ok((bos_id, eos_id))
}

/// Validates the decoder chain used by decode.
fn validate_decoder(value: &Value) -> Result<()> {
    let decoder = object(value, "decoder")?;
    if string(
        decoder
            .get("type")
            .ok_or_else(|| Error::corrupt("decoder is missing type"))?,
        "decoder.type",
    )? != "Sequence"
    {
        return Err(Error::unsupported(
            "tokenizer decoder is not a Sequence decoder",
        ));
    }
    let decoders = array(
        decoder
            .get("decoders")
            .ok_or_else(|| Error::corrupt("decoder is missing decoders"))?,
        "decoder.decoders",
    )?;
    if decoders.len() != 3 {
        return Err(Error::unsupported(
            "tokenizer decoder chain must have three stages",
        ));
    }

    let replace = object(&decoders[0], "decoder.decoders[0]")?;
    if string(
        replace
            .get("type")
            .ok_or_else(|| Error::corrupt("decoder stage is missing type"))?,
        "decoder.decoders[0].type",
    )? != "Replace"
    {
        return Err(Error::unsupported(
            "tokenizer decoder first stage is not Replace",
        ));
    }
    let pattern = object(
        replace
            .get("pattern")
            .ok_or_else(|| Error::corrupt("decoder Replace is missing pattern"))?,
        "decoder.decoders[0].pattern",
    )?;
    if string(
        pattern
            .get("String")
            .ok_or_else(|| Error::corrupt("decoder Replace pattern is missing String"))?,
        "decoder.decoders[0].pattern.String",
    )? != "▁"
    {
        return Err(Error::unsupported(
            "tokenizer decoder does not replace the space marker",
        ));
    }
    if string(
        replace
            .get("content")
            .ok_or_else(|| Error::corrupt("decoder Replace is missing content"))?,
        "decoder.decoders[0].content",
    )? != " "
    {
        return Err(Error::unsupported(
            "tokenizer decoder does not replace the marker with a space",
        ));
    }

    let byte_fallback = object(&decoders[1], "decoder.decoders[1]")?;
    if string(
        byte_fallback
            .get("type")
            .ok_or_else(|| Error::corrupt("decoder stage is missing type"))?,
        "decoder.decoders[1].type",
    )? != "ByteFallback"
    {
        return Err(Error::unsupported(
            "tokenizer decoder second stage is not ByteFallback",
        ));
    }

    let fuse = object(&decoders[2], "decoder.decoders[2]")?;
    if string(
        fuse.get("type")
            .ok_or_else(|| Error::corrupt("decoder stage is missing type"))?,
        "decoder.decoders[2].type",
    )? != "Fuse"
    {
        return Err(Error::unsupported(
            "tokenizer decoder third stage is not Fuse",
        ));
    }
    Ok(())
}

/// Extracts the special-token name from a TemplateProcessing part.
fn parse_special_token<'a>(value: &'a Value, context: &str) -> Result<&'a str> {
    let part = object(value, context)?;
    let special = object(
        part.get("SpecialToken")
            .ok_or_else(|| Error::corrupt(format!("{context} is missing SpecialToken")))?,
        context,
    )?;
    string(
        special
            .get("id")
            .ok_or_else(|| Error::corrupt(format!("{context} is missing id")))?,
        context,
    )
}

/// Parses a byte-fallback token such as <0x0A>.
fn parse_byte_token(token: &str) -> Option<u8> {
    let bytes = token.as_bytes();
    if bytes.len() != 6 || !token.starts_with("<0x") || !token.ends_with('>') {
        return None;
    }
    u8::from_str_radix(&token[3..5], 16).ok()
}

/// Extracts a JSON object.
fn object<'a>(value: &'a Value, context: &str) -> Result<&'a json::Object> {
    value
        .as_object()
        .ok_or_else(|| Error::corrupt(format!("tokenizer {context} must be a JSON object")))
}

/// Extracts a JSON array.
fn array<'a>(value: &'a Value, context: &str) -> Result<&'a [Value]> {
    value
        .as_array()
        .ok_or_else(|| Error::corrupt(format!("tokenizer {context} must be a JSON array")))
}

/// Extracts a JSON string.
fn string<'a>(value: &'a Value, context: &str) -> Result<&'a str> {
    value
        .as_str()
        .ok_or_else(|| Error::corrupt(format!("tokenizer {context} must be a string")))
}

/// Extracts a JSON boolean.
fn boolean(value: &Value, context: &str) -> Result<bool> {
    value
        .as_bool()
        .ok_or_else(|| Error::corrupt(format!("tokenizer {context} must be a boolean")))
}

/// Extracts a JSON unsigned integer.
fn unsigned(value: &Value, context: &str) -> Result<u64> {
    value
        .as_number()
        .and_then(Number::as_u64)
        .ok_or_else(|| Error::corrupt(format!("tokenizer {context} must be an unsigned integer")))
}

/// Requires a JSON null.
fn expect_null(value: &Value, context: &str) -> Result<()> {
    if matches!(value, Value::Null) {
        Ok(())
    } else {
        Err(Error::unsupported(format!(
            "tokenizer {context} is not null"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINI: &str = include_str!("../../fixtures/tokenizer/mini_tokenizer.json");

    fn mini() -> Tokenizer {
        Tokenizer::from_json_str(MINI).expect("the reduced tokenizer fixture must load")
    }

    #[test]
    fn encodes_reference_texts() {
        let tokenizer = mini();
        let cases = [
            ("hello", vec![23391]),
            ("Hello, world!", vec![9259, 236764, 1902, 236888]),
            ("Bonjour, le monde !", vec![53406, 236764, 674, 22150, 1717]),
            ("café", vec![123125, 236859]),
            ("😀 emoji", vec![242398, 64334]),
            ("  leading spaces", vec![138, 26016, 9952]),
            ("multiple   spaces", vec![43819, 139, 35220]),
            ("a\nb\tc", vec![236746, 107, 236763, 255968, 236755]),
            ("<bos>hello<eos>", vec![2, 23391, 1]),
            ("<|image|> a photo", vec![258880, 496, 4429]),
            ("\u{10FFFF}", vec![482, 381, 429, 429]),
        ];
        for (text, expected) in cases {
            assert_eq!(
                tokenizer.encode(text, false).expect("encoding"),
                expected,
                "text {text:?}"
            );
        }
    }

    #[test]
    fn adds_bos_and_eos() {
        let tokenizer = mini();
        assert_eq!(tokenizer.encode("", true).expect("encoding"), vec![2, 1]);
        assert_eq!(
            tokenizer.encode("hello", true).expect("encoding"),
            vec![2, 23391, 1]
        );
        assert_eq!(
            tokenizer.encode("<bos>hello<eos>", true).expect("encoding"),
            vec![2, 2, 23391, 1, 1]
        );
    }

    #[test]
    fn decodes_reference_texts() {
        let tokenizer = mini();
        for (text, ids) in [
            ("hello", vec![23391]),
            ("café", vec![123125, 236859]),
            ("Bonjour, le monde !", vec![53406, 236764, 674, 22150, 1717]),
            ("  leading spaces", vec![138, 26016, 9952]),
            ("a\nb\tc", vec![236746, 107, 236763, 255968, 236755]),
        ] {
            assert_eq!(
                tokenizer.decode(&ids, true).expect("decoding"),
                text,
                "ids {ids:?}"
            );
        }
    }

    #[test]
    fn skips_special_tokens_when_requested() {
        let tokenizer = mini();
        assert_eq!(
            tokenizer.decode(&[2, 23391, 1], true).expect("decoding"),
            "hello"
        );
        assert_eq!(
            tokenizer.decode(&[2, 23391, 1], false).expect("decoding"),
            "<bos>hello<eos>"
        );
        assert_eq!(
            tokenizer
                .decode(&[258880, 496, 4429], true)
                .expect("decoding"),
            " a photo"
        );
    }

    #[test]
    fn exposes_special_ids() {
        let tokenizer = mini();
        assert_eq!(tokenizer.bos_id(), 2);
        assert_eq!(tokenizer.eos_id(), 1);
        assert_eq!(tokenizer.unk_id(), 3);
        assert_eq!(tokenizer.pad_id(), Some(0));
        assert!(tokenizer.is_special_id(2));
        assert!(!tokenizer.is_special_id(23391));
        assert_eq!(tokenizer.token_id("<bos>"), Some(2));
        assert_eq!(tokenizer.token(2), Some("<bos>"));
        assert_eq!(tokenizer.vocab_size(), 694);
        assert_eq!(tokenizer.max_token_id(), 258884);
    }

    #[test]
    fn rejects_unsupported_or_inconsistent_documents() {
        for (document, label) in [
            ("{}", "missing model"),
            (r#"{"model":{"type":"Unigram"}}"#, "Unigram model"),
        ] {
            let error = Tokenizer::from_json_str(document).expect_err("document must be rejected");
            assert!(
                matches!(error, Error::Corrupt(_) | Error::Unsupported(_)),
                "{label}: {error}"
            );
        }

        let unigram = MINI.replace("\"type\":\"BPE\"", "\"type\":\"Unigram\"");
        let error = Tokenizer::from_json_str(&unigram).expect_err("Unigram must be rejected");
        assert!(matches!(error, Error::Unsupported(_)));

        let no_byte_fallback = MINI.replace("\"byte_fallback\":true", "\"byte_fallback\":false");
        let error = Tokenizer::from_json_str(&no_byte_fallback)
            .expect_err("disabled byte fallback must be rejected");
        assert!(matches!(error, Error::Unsupported(_)));

        let no_fuse = MINI.replace("\"fuse_unk\":true", "\"fuse_unk\":false");
        let error =
            Tokenizer::from_json_str(&no_fuse).expect_err("disabled fuse_unk must be rejected");
        assert!(matches!(error, Error::Unsupported(_)));
    }

    #[test]
    fn rejects_oversized_input() {
        let tokenizer = mini();
        let input = "a".repeat(MAX_INPUT_BYTES + 1);
        let error = tokenizer
            .encode(&input, false)
            .expect_err("input cap must be enforced");
        assert!(matches!(error, Error::LimitExceeded(_)));
    }

    #[test]
    fn rejects_unknown_ids_when_decoding() {
        let tokenizer = mini();
        let error = tokenizer
            .decode(&[u32::MAX], true)
            .expect_err("unknown id must fail");
        assert!(matches!(error, Error::Corrupt(_)));
    }

    #[test]
    fn byte_fallback_round_trips() {
        let tokenizer = mini();
        let ids = tokenizer
            .encode("\u{10FFFF}", false)
            .expect("encoding rare scalar");
        assert_eq!(ids, vec![482, 381, 429, 429]);
        assert_eq!(
            tokenizer.decode(&ids, true).expect("decoding"),
            "\u{10FFFF}"
        );
    }
}
