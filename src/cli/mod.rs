//! Argument parsing and command execution for the CLI.
//!
//! M2 added the embed command. M3 adds index, search, and verify:
//!
//! - index ingests a strict JSONL document file into the persistent store;
//! - search runs BM25, exact vector search, or RRF hybrid ranking;
//! - verify decodes the committed journal prefix without modifying it.
//!
//! Text embedding uses the frozen model tokenizer and official Document or
//! SearchQuery prompt. A diagnostic --ids option bypasses the tokenizer for
//! reproducible tests against the tiny official model; it never fakes an
//! embedding.

use std::collections::BTreeSet;
use std::ffi::{OsStr, OsString};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::binary::reader::read_file_bounded;
use crate::error::{Error, Result};
use crate::index::{Filters, Index, SearchHit, SearchMode, SearchRequest};
use crate::json::{self, Number, Object, Value};
use crate::media::preprocess::PreprocessOptions;
use crate::media::{Audio, Image, jpeg, png, wav};
use crate::model::prompt::Prompt;
use crate::model::text::{TextEmbedder, TextModel};
use crate::model::{AudioEmbedder, ImageEmbedder};
use crate::storage::model::{ChunkInput, Document, DocumentId, Metadata, Modality};
use crate::storage::{ReadOnlyStore, Store};
use crate::{MODEL_CACHE_DIR, MODEL_REVISION, PROGRAM_NAME, TARGET_TRIPLE, VERSION};

/// Maximum number of bytes read from standard input.
const MAX_STDIN_BYTES: usize = 1024 * 1024;
/// Maximum size of an index JSONL file.
const MAX_INDEX_FILE_BYTES: u64 = 64 * 1024 * 1024;
/// Maximum size of one JSONL document line.
const MAX_INDEX_LINE_BYTES: usize = 1024 * 1024;
/// Maximum number of documents ingested by one index invocation.
const MAX_INDEX_DOCUMENTS: usize = 10_000;
/// Maximum number of diagnostic token ids accepted by one command.
const MAX_IDS: usize = 8192;

/// Help text displayed by the help command.
const HELP: &str = concat!(
    env!("CARGO_PKG_NAME"),
    r#" — local multimodal embedding and search engine

USAGE:
    "#,
    env!("CARGO_PKG_NAME"),
    r#" <COMMAND>

COMMANDS:
    embed      Embeds text, token ids, an image, or a WAV clip and prints a JSON vector
    index      Ingests JSONL documents into the persistent store
    search     Runs hybrid, lexical, or exact vector search
    verify     Verifies a data directory without modifying it
    compact    Materializes a snapshot and replaces the journal
    version    Prints the version and build target
    help       Prints this help

EMBED:
    embed [--model-dir <DIR>] [--prompt <NAME>] [--dim <N>]
          [--text <TEXT> | --ids <ID,ID,...> | --image <PATH> | --audio <PATH>]
          [TEXT...]

    --model-dir   Model directory (default: .model-cache/<revision>)
    --prompt      SearchQuery, Document, Classification, Clustering,
                  CodeRetrieval, FactChecking, SentenceSimilarity, or
                  QuestionAnswering (default: SearchQuery)
    --dim         Output dimension, from 1 to the model embedding dimension
                  (default: 768; MRL dimensions are 768, 512, 256, 128)
    --vision-tokens  Visual token budget for images: 70, 140, 280, 560, or
                  1120 (default: 280)
    --text        Text to embed; if omitted, positional TEXT is joined with
                  spaces, and if no text is given standard input is read
    --ids         Comma-separated token ids; bypasses the tokenizer
    --image       PNG or JPEG image to embed; mutually exclusive with text,
                  --ids, and --audio. Image embeddings use the official
                  preprocessing and vision tower, not a caption or file name.
    --audio       RIFF/WAVE PCM or IEEE-float audio to embed; mutually
                  exclusive with text, --ids, and --image. Audio is converted
                  to 16 kHz mono, resampled natively, and embedded through the
                  official log-mel processor and audio tower. The 30-second
                  cap applies; no transcription is performed.

INDEX:
    index --data-dir <DIR> [--model-dir <DIR>] [--dim <N>] [--ids]
          <FILE|->

    --data-dir    Data directory holding the journal and writer lock
    --model-dir   Model directory (default: .model-cache/<revision>)
    --dim         Embedding dimension (default: 768)
    --vision-tokens  Visual token budget for image documents: 70, 140, 280,
                  560, or 1120 (default: 280)
    --ids         Diagnostic mode: JSONL lines carry an ids array used
                  instead of tokenizing text; text is still stored for BM25

    Each JSONL line is an object with id and exactly one of text, image, or
    audio, plus optional title, source, and metadata object. Image lines are
    decoded with the native PNG/JPEG decoders and embedded through the vision
    tower; audio lines are decoded with the native WAV decoder and embedded
    through the audio tower. Their chunk text is empty, so they participate in
    vector search but not BM25. When source is omitted for an image or audio
    document, its path becomes the source. Blank lines are ignored. Duplicate
    ids and unknown fields are rejected before any write.

SEARCH:
    search --data-dir <DIR> [--model-dir <DIR>] [--dim <N>]
           [--mode hybrid|lexical|vector] [--limit <N>] [--offset <N>]
           [--filter KEY=VALUE]... [--ids <ID,ID,...>] [--image <PATH>]
           [--audio <PATH>] [QUERY...]

    --data-dir    Data directory holding the journal and writer lock
    --model-dir   Model directory (default: .model-cache/<revision>)
    --dim         Query embedding dimension (default: 768)
    --mode        hybrid (default), lexical (BM25 only), or vector
    --limit       Maximum hits, from 1 to 1000 (default: 10)
    --offset      Number of ranked hits to skip (default: 0)
    --filter      Exact metadata equality filter; repeatable
    --ids         Diagnostic query ids for vector or hybrid mode
    --image       PNG or JPEG image query; requires vector mode and cannot be
                  combined with query text, --ids, or --audio
    --audio       RIFF/WAVE audio query; requires vector mode and cannot be
                  combined with query text, --ids, or --image
    --vision-tokens  Visual token budget for an image query: 70, 140, 280,
                  560, or 1120 (default: 280)

    QUERY is joined with spaces; with no query, standard input is read with
    a 1 MiB cap. Lexical mode does not load the model.

VERIFY:
    verify --data-dir <DIR>

    Decodes and validates the committed snapshot and journal prefix, reporting
    records, snapshot records, documents, chunks, and any uncommitted trailing
    bytes. Neither file is modified.

COMPACT:
    compact --data-dir <DIR>

    Materializes the current documents and chunks into snapshot.sv, then
    replaces journal.log with an empty file. The snapshot is written and
    synchronized before the journal is replaced, so a crash leaves a
    recoverable state.

OPTIONS:
    -V, --version    Equivalent to version
    -h, --help       Equivalent to help

EXIT CODES:
    0    success
    1    execution error
    2    usage error

The serve command will be added with milestone M6. No command is simulated
in the meantime.
"#
);

/// Command recognized by the CLI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// Prints the program version and build target.
    Version,
    /// Prints help and exit codes.
    Help,
    /// Embeds text or token ids.
    Embed(EmbedArgs),
    /// Ingests JSONL documents.
    Index(IndexArgs),
    /// Runs a search.
    Search(SearchArgs),
    /// Verifies a data directory.
    Verify(VerifyArgs),
    /// Materializes a snapshot and compacts the journal.
    Compact(CompactArgs),
}

/// Arguments of the embed command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedArgs {
    /// Directory containing config.json and model.safetensors, and for text
    /// input also tokenizer.json.
    pub model_dir: PathBuf,
    /// Official prompt role applied before tokenization.
    pub prompt: Prompt,
    /// Requested Matryoshka dimension.
    pub dimension: usize,
    /// Visual token budget for image input.
    pub vision_budget: usize,
    /// Input source.
    pub input: EmbedInput,
}

/// Input source of the embed command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbedInput {
    /// Raw text, prefixed with the official prompt before tokenization.
    Text(String),
    /// Token ids used directly, without the tokenizer or prompt prefix.
    Ids(Vec<u32>),
    /// PNG or JPEG image read from the given path.
    Image(PathBuf),
    /// RIFF/WAVE audio read from the given path.
    Audio(PathBuf),
    /// Text read from standard input.
    Stdin,
}

/// Arguments of the index command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexArgs {
    /// Data directory.
    pub data_dir: PathBuf,
    /// Model directory.
    pub model_dir: PathBuf,
    /// Embedding dimension.
    pub dimension: usize,
    /// Visual token budget for image documents.
    pub vision_budget: usize,
    /// JSONL file, or - for standard input.
    pub input: PathBuf,
    /// Diagnostic mode that uses an ids array instead of the tokenizer.
    pub ids_mode: bool,
}

/// Arguments of the search command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchArgs {
    /// Data directory.
    pub data_dir: PathBuf,
    /// Model directory, used by vector and hybrid modes.
    pub model_dir: PathBuf,
    /// Query embedding dimension.
    pub dimension: usize,
    /// Ranking strategy.
    pub mode: SearchMode,
    /// Maximum number of hits.
    pub limit: usize,
    /// Pagination offset.
    pub offset: usize,
    /// Modality and metadata filters.
    pub filters: Filters,
    /// Query text or standard input.
    pub input: SearchInput,
    /// Optional image query for vector mode.
    pub image: Option<PathBuf>,
    /// Optional audio query for vector mode.
    pub audio: Option<PathBuf>,
    /// Visual token budget for an image query.
    pub vision_budget: usize,
    /// Diagnostic query token ids for vector or hybrid mode.
    pub ids: Option<Vec<u32>>,
}

/// Query source of the search command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchInput {
    /// Query text from positional arguments.
    Text(String),
    /// Query text read from standard input.
    Stdin,
}

/// Arguments of the verify command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyArgs {
    /// Data directory.
    pub data_dir: PathBuf,
}

/// Arguments of the compact command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactArgs {
    /// Data directory.
    pub data_dir: PathBuf,
}

/// Default model directory for the frozen revision.
pub fn default_model_dir() -> PathBuf {
    Path::new(MODEL_CACHE_DIR).join(MODEL_REVISION)
}

/// Parses the arguments following the program name.
///
/// Non-UTF-8 arguments are rejected with a usage error rather than causing a
/// panic: the CLI must never abort on hostile input.
pub fn parse<I, S>(args: I) -> Result<Command>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    let mut args = args.into_iter();
    let Some(first) = args.next() else {
        return Err(Error::usage(
            "no command provided; use --help to display help",
        ));
    };
    let first = first
        .into()
        .into_string()
        .map_err(|_| Error::usage("non-UTF-8 argument"))?;
    let rest = args
        .map(|argument| {
            argument
                .into()
                .into_string()
                .map_err(|_| Error::usage("non-UTF-8 argument"))
        })
        .collect::<Result<Vec<_>>>()?;

    match first.as_str() {
        "version" | "--version" | "-V" => {
            reject_extra(&rest, "version")?;
            Ok(Command::Version)
        }
        "help" | "--help" | "-h" => {
            reject_extra(&rest, "help")?;
            Ok(Command::Help)
        }
        "embed" => parse_embed(rest),
        "index" => parse_index(rest),
        "search" => parse_search(rest),
        "verify" => parse_verify(rest),
        "compact" => parse_compact(rest),
        other => Err(Error::usage(format!("unknown command {other}; use --help"))),
    }
}

fn reject_extra(args: &[String], command: &str) -> Result<()> {
    if args.is_empty() {
        Ok(())
    } else {
        Err(Error::usage(format!("extra arguments after {command}")))
    }
}

fn next_value<'a>(args: &'a [String], index: &mut usize, option: &str) -> Result<&'a str> {
    *index += 1;
    args.get(*index)
        .map(String::as_str)
        .ok_or_else(|| Error::usage(format!("{option} requires a value")))
}

fn parse_dimension(value: &str) -> Result<usize> {
    let dimension = value
        .parse::<usize>()
        .map_err(|_| Error::usage(format!("invalid dimension {value}")))?;
    if dimension == 0 {
        return Err(Error::usage("dimension must be positive"));
    }
    Ok(dimension)
}

fn parse_vision_budget(value: &str) -> Result<usize> {
    let budget = value
        .parse::<usize>()
        .map_err(|_| Error::usage(format!("invalid vision token budget {value}")))?;
    PreprocessOptions::new(budget)?;
    Ok(budget)
}

fn parse_positive_usize(value: &str, name: &str) -> Result<usize> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| Error::usage(format!("invalid {name} {value}")))?;
    if parsed == 0 {
        return Err(Error::usage(format!("{name} must be positive")));
    }
    Ok(parsed)
}

fn parse_embed(args: Vec<String>) -> Result<Command> {
    let mut model_dir = default_model_dir();
    let mut prompt = Prompt::SearchQuery;
    let mut dimension = 768usize;
    let mut vision_budget = crate::media::preprocess::DEFAULT_MAX_SOFT_TOKENS;
    let mut ids = None;
    let mut explicit_text = None;
    let mut image = None;
    let mut audio = None;
    let mut text_parts = Vec::new();
    let mut index = 0;

    while index < args.len() {
        let argument = args[index].as_str();
        match argument {
            "--model-dir" => {
                let value = next_value(&args, &mut index, "--model-dir")?;
                model_dir = PathBuf::from(value);
            }
            "--prompt" => {
                let value = next_value(&args, &mut index, "--prompt")?;
                prompt = Prompt::from_name(value).ok_or_else(|| {
                    Error::usage(format!(
                        "unknown prompt {value}; use SearchQuery, Document, Classification, Clustering, CodeRetrieval, FactChecking, SentenceSimilarity, or QuestionAnswering"
                    ))
                })?;
            }
            "--dim" => {
                let value = next_value(&args, &mut index, "--dim")?;
                dimension = parse_dimension(value)?;
            }
            "--ids" => {
                let value = next_value(&args, &mut index, "--ids")?;
                if ids.is_some() {
                    return Err(Error::usage("--ids was provided more than once"));
                }
                ids = Some(parse_ids(value)?);
            }
            "--vision-tokens" => {
                let value = next_value(&args, &mut index, "--vision-tokens")?;
                vision_budget = parse_vision_budget(value)?;
            }
            "--text" => {
                let value = next_value(&args, &mut index, "--text")?;
                if explicit_text.is_some() {
                    return Err(Error::usage("--text was provided more than once"));
                }
                explicit_text = Some(value.to_owned());
            }
            "--image" => {
                let value = next_value(&args, &mut index, "--image")?;
                if image.is_some() {
                    return Err(Error::usage("--image was provided more than once"));
                }
                image = Some(PathBuf::from(value));
            }
            "--audio" => {
                let value = next_value(&args, &mut index, "--audio")?;
                if audio.is_some() {
                    return Err(Error::usage("--audio was provided more than once"));
                }
                audio = Some(PathBuf::from(value));
            }
            "-h" | "--help" => return Ok(Command::Help),
            other if other.starts_with('-') => {
                return Err(Error::usage(format!("unknown option {other}")));
            }
            other => text_parts.push(other.to_owned()),
        }
        index += 1;
    }

    if ids.is_some() && (explicit_text.is_some() || !text_parts.is_empty()) {
        return Err(Error::usage("--ids cannot be combined with text input"));
    }
    if explicit_text.is_some() && !text_parts.is_empty() {
        return Err(Error::usage(
            "--text cannot be combined with positional text",
        ));
    }
    if image.is_some() && (ids.is_some() || explicit_text.is_some() || !text_parts.is_empty()) {
        return Err(Error::usage(
            "--image cannot be combined with text or token ids",
        ));
    }
    if audio.is_some()
        && (ids.is_some() || explicit_text.is_some() || !text_parts.is_empty() || image.is_some())
    {
        return Err(Error::usage(
            "--audio cannot be combined with text, token ids, or an image",
        ));
    }

    let input = if let Some(path) = audio {
        EmbedInput::Audio(path)
    } else if let Some(path) = image {
        EmbedInput::Image(path)
    } else if let Some(ids) = ids {
        EmbedInput::Ids(ids)
    } else if let Some(text) = explicit_text {
        EmbedInput::Text(text)
    } else if text_parts.is_empty() {
        EmbedInput::Stdin
    } else {
        EmbedInput::Text(text_parts.join(" "))
    };

    Ok(Command::Embed(EmbedArgs {
        model_dir,
        prompt,
        dimension,
        vision_budget,
        input,
    }))
}

fn parse_ids(value: &str) -> Result<Vec<u32>> {
    let mut ids = Vec::new();
    for part in value.split(',') {
        if part.is_empty() {
            return Err(Error::usage("--ids contains an empty id"));
        }
        let id = part
            .parse::<u32>()
            .map_err(|_| Error::usage(format!("invalid token id {part}")))?;
        ids.push(id);
    }
    if ids.is_empty() {
        return Err(Error::usage("--ids requires at least one id"));
    }
    if ids.len() > MAX_IDS {
        return Err(Error::limit_exceeded(format!(
            "--ids has {} ids, above the {MAX_IDS}-id cap",
            ids.len()
        )));
    }
    Ok(ids)
}

fn parse_index(args: Vec<String>) -> Result<Command> {
    let mut data_dir = None;
    let mut model_dir = default_model_dir();
    let mut dimension = 768usize;
    let mut vision_budget = crate::media::preprocess::DEFAULT_MAX_SOFT_TOKENS;
    let mut input = None;
    let mut ids_mode = false;
    let mut index = 0;

    while index < args.len() {
        let argument = args[index].as_str();
        match argument {
            "--data-dir" => {
                let value = next_value(&args, &mut index, "--data-dir")?;
                data_dir = Some(PathBuf::from(value));
            }
            "--model-dir" => {
                let value = next_value(&args, &mut index, "--model-dir")?;
                model_dir = PathBuf::from(value);
            }
            "--dim" => {
                let value = next_value(&args, &mut index, "--dim")?;
                dimension = parse_dimension(value)?;
            }
            "--ids" => ids_mode = true,
            "--vision-tokens" => {
                let value = next_value(&args, &mut index, "--vision-tokens")?;
                vision_budget = parse_vision_budget(value)?;
            }
            "-h" | "--help" => return Ok(Command::Help),
            "-" => {
                if input.is_some() {
                    return Err(Error::usage("index accepts exactly one JSONL file or -"));
                }
                input = Some(PathBuf::from("-"));
            }
            other if other.starts_with('-') => {
                return Err(Error::usage(format!("unknown option {other}")));
            }
            other => {
                if input.is_some() {
                    return Err(Error::usage("index accepts exactly one JSONL file or -"));
                }
                input = Some(PathBuf::from(other));
            }
        }
        index += 1;
    }

    let data_dir = data_dir.ok_or_else(|| Error::usage("index requires --data-dir <DIR>"))?;
    let input = input.ok_or_else(|| Error::usage("index requires a JSONL file or -"))?;
    Ok(Command::Index(IndexArgs {
        data_dir,
        model_dir,
        dimension,
        vision_budget,
        input,
        ids_mode,
    }))
}

fn parse_search(args: Vec<String>) -> Result<Command> {
    let mut data_dir = None;
    let mut model_dir = default_model_dir();
    let mut dimension = 768usize;
    let mut mode = SearchMode::Hybrid;
    let mut limit = 10usize;
    let mut offset = 0usize;
    let mut filters = Filters::new();
    let mut ids = None;
    let mut image = None;
    let mut audio = None;
    let mut vision_budget = crate::media::preprocess::DEFAULT_MAX_SOFT_TOKENS;
    let mut query_parts = Vec::new();
    let mut index = 0;

    while index < args.len() {
        let argument = args[index].as_str();
        match argument {
            "--data-dir" => {
                let value = next_value(&args, &mut index, "--data-dir")?;
                data_dir = Some(PathBuf::from(value));
            }
            "--model-dir" => {
                let value = next_value(&args, &mut index, "--model-dir")?;
                model_dir = PathBuf::from(value);
            }
            "--dim" => {
                let value = next_value(&args, &mut index, "--dim")?;
                dimension = parse_dimension(value)?;
            }
            "--mode" => {
                let value = next_value(&args, &mut index, "--mode")?;
                mode = SearchMode::from_name(value).ok_or_else(|| {
                    Error::usage(format!(
                        "unknown search mode {value}; use hybrid, lexical, or vector"
                    ))
                })?;
            }
            "--limit" => {
                let value = next_value(&args, &mut index, "--limit")?;
                limit = parse_positive_usize(value, "limit")?;
            }
            "--offset" => {
                let value = next_value(&args, &mut index, "--offset")?;
                offset = value
                    .parse::<usize>()
                    .map_err(|_| Error::usage(format!("invalid offset {value}")))?;
            }
            "--filter" => {
                let value = next_value(&args, &mut index, "--filter")?;
                parse_filter(value, &mut filters)?;
            }
            "--ids" => {
                let value = next_value(&args, &mut index, "--ids")?;
                if ids.is_some() {
                    return Err(Error::usage("--ids was provided more than once"));
                }
                ids = Some(parse_ids(value)?);
            }
            "--image" => {
                let value = next_value(&args, &mut index, "--image")?;
                if image.is_some() {
                    return Err(Error::usage("--image was provided more than once"));
                }
                image = Some(PathBuf::from(value));
            }
            "--audio" => {
                let value = next_value(&args, &mut index, "--audio")?;
                if audio.is_some() {
                    return Err(Error::usage("--audio was provided more than once"));
                }
                audio = Some(PathBuf::from(value));
            }
            "--vision-tokens" => {
                let value = next_value(&args, &mut index, "--vision-tokens")?;
                vision_budget = parse_vision_budget(value)?;
            }
            "-h" | "--help" => return Ok(Command::Help),
            other if other.starts_with('-') => {
                return Err(Error::usage(format!("unknown option {other}")));
            }
            other => query_parts.push(other.to_owned()),
        }
        index += 1;
    }

    let data_dir = data_dir.ok_or_else(|| Error::usage("search requires --data-dir <DIR>"))?;
    if mode == SearchMode::Lexical && ids.is_some() {
        return Err(Error::usage(
            "--ids cannot be used with lexical search mode",
        ));
    }
    if image.is_some() {
        if ids.is_some() || !query_parts.is_empty() || audio.is_some() {
            return Err(Error::usage(
                "--image cannot be combined with query text, --ids, or --audio",
            ));
        }
        if mode != SearchMode::Vector {
            return Err(Error::usage("--image requires vector search mode"));
        }
    }
    if audio.is_some() {
        if ids.is_some() || !query_parts.is_empty() || image.is_some() {
            return Err(Error::usage(
                "--audio cannot be combined with query text, --ids, or an image",
            ));
        }
        if mode != SearchMode::Vector {
            return Err(Error::usage("--audio requires vector search mode"));
        }
    }
    let input = if query_parts.is_empty() {
        SearchInput::Stdin
    } else {
        SearchInput::Text(query_parts.join(" "))
    };

    Ok(Command::Search(SearchArgs {
        data_dir,
        model_dir,
        dimension,
        mode,
        limit,
        offset,
        filters,
        input,
        image,
        audio,
        vision_budget,
        ids,
    }))
}

fn parse_filter(value: &str, filters: &mut Filters) -> Result<()> {
    let (key, value) = value
        .split_once('=')
        .ok_or_else(|| Error::usage("--filter requires KEY=VALUE"))?;
    if key.is_empty() || value.is_empty() {
        return Err(Error::usage("--filter key and value must not be empty"));
    }
    filters.add_metadata(key, value)
}

fn parse_verify(args: Vec<String>) -> Result<Command> {
    let mut data_dir = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--data-dir" => {
                let value = next_value(&args, &mut index, "--data-dir")?;
                data_dir = Some(PathBuf::from(value));
            }
            "-h" | "--help" => return Ok(Command::Help),
            other if other.starts_with('-') => {
                return Err(Error::usage(format!("unknown option {other}")));
            }
            other => {
                return Err(Error::usage(format!(
                    "unexpected argument {other}; verify accepts only --data-dir"
                )));
            }
        }
        index += 1;
    }
    let data_dir = data_dir.ok_or_else(|| Error::usage("verify requires --data-dir <DIR>"))?;
    Ok(Command::Verify(VerifyArgs { data_dir }))
}

fn parse_compact(args: Vec<String>) -> Result<Command> {
    let mut data_dir = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--data-dir" => {
                let value = next_value(&args, &mut index, "--data-dir")?;
                data_dir = Some(PathBuf::from(value));
            }
            "-h" | "--help" => return Ok(Command::Help),
            other if other.starts_with('-') => {
                return Err(Error::usage(format!("unknown option {other}")));
            }
            other => {
                return Err(Error::usage(format!(
                    "unexpected argument {other}; compact accepts only --data-dir"
                )));
            }
        }
        index += 1;
    }
    let data_dir = data_dir.ok_or_else(|| Error::usage("compact requires --data-dir <DIR>"))?;
    Ok(Command::Compact(CompactArgs { data_dir }))
}

/// Executes a command and writes its output to stdout.
///
/// # Errors
///
/// Returns a typed error for a usage problem, a model or storage failure, a
/// numerical failure, or a write failure such as a closed pipe.
pub fn run<W: Write>(command: Command, stdout: &mut W) -> Result<()> {
    match command {
        Command::Version => writeln!(stdout, "{PROGRAM_NAME} {VERSION} ({TARGET_TRIPLE})")?,
        Command::Help => stdout.write_all(HELP.as_bytes())?,
        Command::Embed(args) => run_embed(args, stdout)?,
        Command::Index(args) => run_index(args, stdout)?,
        Command::Search(args) => run_search(args, stdout)?,
        Command::Verify(args) => run_verify(args, stdout)?,
        Command::Compact(args) => run_compact(args, stdout)?,
    }
    stdout.flush()?;
    Ok(())
}

fn run_embed<W: Write>(args: EmbedArgs, stdout: &mut W) -> Result<()> {
    match args.input {
        EmbedInput::Ids(ids) => {
            let model = TextModel::load(&args.model_dir)?;
            validate_dimension(args.dimension, model.config().embedding_dim())?;
            let embedding = model.embed_ids(&ids, args.dimension)?;
            write_vector(&embedding, stdout)?;
        }
        EmbedInput::Text(text) => {
            let embedder = TextEmbedder::load(&args.model_dir)?;
            validate_dimension(args.dimension, embedder.model().config().embedding_dim())?;
            let embedding = embedder.embed(&text, args.prompt, args.dimension)?;
            write_vector(&embedding, stdout)?;
        }
        EmbedInput::Image(path) => {
            let image = decode_image(&path)?;
            let embedder = ImageEmbedder::load(&args.model_dir)?;
            validate_dimension(args.dimension, embedder.model().config().embedding_dim())?;
            let embedding = embedder.embed(
                &image,
                args.dimension,
                PreprocessOptions::new(args.vision_budget)?,
            )?;
            write_vector(&embedding, stdout)?;
        }
        EmbedInput::Audio(path) => {
            let audio = decode_audio(&path)?;
            let embedder = AudioEmbedder::load(&args.model_dir)?;
            validate_dimension(args.dimension, embedder.model().config().embedding_dim())?;
            let embedding = embedder.embed(&audio, args.dimension)?;
            write_vector(&embedding, stdout)?;
        }
        EmbedInput::Stdin => {
            let text = read_stdin_bounded(MAX_STDIN_BYTES)?;
            let embedder = TextEmbedder::load(&args.model_dir)?;
            validate_dimension(args.dimension, embedder.model().config().embedding_dim())?;
            let embedding = embedder.embed(&text, args.prompt, args.dimension)?;
            write_vector(&embedding, stdout)?;
        }
    }
    Ok(())
}

fn decode_audio(path: &Path) -> Result<Audio> {
    let extension = path
        .extension()
        .and_then(OsStr::to_str)
        .map(|value| value.to_ascii_lowercase());
    match extension.as_deref() {
        Some("wav") => wav::decode_file(path),
        _ => Err(Error::usage("audio input must have a .wav extension")),
    }
}

fn decode_image(path: &Path) -> Result<Image> {
    let extension = path
        .extension()
        .and_then(OsStr::to_str)
        .map(|value| value.to_ascii_lowercase());
    match extension.as_deref() {
        Some("png") => png::decode_file(path),
        Some("jpg" | "jpeg") => jpeg::decode_file(path),
        _ => Err(Error::usage(
            "image input must have a .png, .jpg, or .jpeg extension",
        )),
    }
}

fn run_index<W: Write>(args: IndexArgs, stdout: &mut W) -> Result<()> {
    let input = read_index_input(&args.input)?;
    let pending = parse_index_documents(&input, args.ids_mode)?;

    let mut ingested = 0usize;
    let mut chunks = 0usize;

    if args.ids_mode {
        let model = TextModel::load(&args.model_dir)?;
        validate_dimension(args.dimension, model.config().embedding_dim())?;
        let mut store = Store::open(&args.data_dir)?;
        for pending in pending {
            let ids = pending
                .ids
                .as_ref()
                .ok_or_else(|| Error::internal("ids mode document is missing its ids array"))?;
            let embedding = model.embed_ids(ids, args.dimension)?;
            let chunk_ids = store.put_document(
                pending.document,
                vec![ChunkInput::new(pending.text, embedding)],
            )?;
            ingested += 1;
            chunks += chunk_ids.len();
        }
        write_index_summary(stdout, &args.data_dir, &store, ingested, chunks)?;
    } else if pending
        .iter()
        .any(|document| document.image.is_some() || document.audio.is_some())
    {
        let has_image = pending.iter().any(|document| document.image.is_some());
        let has_audio = pending.iter().any(|document| document.audio.is_some());
        let image_embedder = if has_image {
            Some(ImageEmbedder::load(&args.model_dir)?)
        } else {
            None
        };
        let audio_embedder = if has_audio {
            Some(AudioEmbedder::load(&args.model_dir)?)
        } else {
            None
        };
        if let Some(embedder) = &image_embedder {
            validate_dimension(args.dimension, embedder.model().config().embedding_dim())?;
        }
        if let Some(embedder) = &audio_embedder {
            validate_dimension(args.dimension, embedder.model().config().embedding_dim())?;
        }
        let mut store = Store::open(&args.data_dir)?;
        for pending in pending {
            let embedding = if let Some(path) = &pending.image {
                let image = decode_image(path)?;
                image_embedder
                    .as_ref()
                    .ok_or_else(|| Error::internal("image embedder was not loaded"))?
                    .embed(
                        &image,
                        args.dimension,
                        PreprocessOptions::new(args.vision_budget)?,
                    )?
            } else if let Some(path) = &pending.audio {
                let audio = decode_audio(path)?;
                audio_embedder
                    .as_ref()
                    .ok_or_else(|| Error::internal("audio embedder was not loaded"))?
                    .embed(&audio, args.dimension)?
            } else if let Some(embedder) = &image_embedder {
                embedder.embed_text(&pending.text, Prompt::Document, args.dimension)?
            } else if let Some(embedder) = &audio_embedder {
                embedder.embed_text(&pending.text, Prompt::Document, args.dimension)?
            } else {
                return Err(Error::internal(
                    "multimodal index branch has no loaded embedder",
                ));
            };
            let chunk_ids = store.put_document(
                pending.document,
                vec![ChunkInput::new(pending.text, embedding)],
            )?;
            ingested += 1;
            chunks += chunk_ids.len();
        }
        write_index_summary(stdout, &args.data_dir, &store, ingested, chunks)?;
    } else {
        let embedder = TextEmbedder::load(&args.model_dir)?;
        validate_dimension(args.dimension, embedder.model().config().embedding_dim())?;
        let mut store = Store::open(&args.data_dir)?;
        for pending in pending {
            let embedding = embedder.embed(&pending.text, Prompt::Document, args.dimension)?;
            let chunk_ids = store.put_document(
                pending.document,
                vec![ChunkInput::new(pending.text, embedding)],
            )?;
            ingested += 1;
            chunks += chunk_ids.len();
        }
        write_index_summary(stdout, &args.data_dir, &store, ingested, chunks)?;
    }
    Ok(())
}

fn write_index_summary<W: Write>(
    stdout: &mut W,
    data_dir: &Path,
    store: &Store,
    documents: usize,
    chunks: usize,
) -> Result<()> {
    let mut object = Object::new();
    object.insert(
        "documents".to_owned(),
        Value::Number(Number::Unsigned(documents as u64)),
    );
    object.insert(
        "chunks".to_owned(),
        Value::Number(Number::Unsigned(chunks as u64)),
    );
    object.insert(
        "dimension".to_owned(),
        Value::Number(Number::Unsigned(
            store.embedding_dimension().unwrap_or(0) as u64
        )),
    );
    object.insert(
        "data_dir".to_owned(),
        Value::String(data_dir.display().to_string()),
    );
    object.insert(
        "recovery_records".to_owned(),
        Value::Number(Number::Unsigned(store.recovery().replayed_records)),
    );
    object.insert(
        "recovery_truncated_bytes".to_owned(),
        Value::Number(Number::Unsigned(store.recovery().truncated_bytes)),
    );
    write_json(stdout, &Value::Object(object))
}

fn run_search<W: Write>(args: SearchArgs, stdout: &mut W) -> Result<()> {
    if args.mode == SearchMode::Lexical && args.ids.is_some() {
        return Err(Error::usage(
            "--ids cannot be used with lexical search mode",
        ));
    }
    if let Some(path) = &args.image {
        if args.mode != SearchMode::Vector {
            return Err(Error::usage("--image requires vector search mode"));
        }
        let image = decode_image(path)?;
        let embedder = ImageEmbedder::load(&args.model_dir)?;
        validate_dimension(args.dimension, embedder.model().config().embedding_dim())?;
        let embedding = embedder.embed(
            &image,
            args.dimension,
            PreprocessOptions::new(args.vision_budget)?,
        )?;
        let store = ReadOnlyStore::open(&args.data_dir)?;
        let index = Index::build_read_only(&store)?;
        let mut request = SearchRequest::new(SearchMode::Vector, args.limit);
        request.offset = args.offset;
        request.filters = args.filters;
        request.text = None;
        request.embedding = Some(&embedding);
        let hits = index.search(&request)?;
        return write_hits(stdout, &hits);
    }
    if let Some(path) = &args.audio {
        if args.mode != SearchMode::Vector {
            return Err(Error::usage("--audio requires vector search mode"));
        }
        let audio = decode_audio(path)?;
        let embedder = AudioEmbedder::load(&args.model_dir)?;
        validate_dimension(args.dimension, embedder.model().config().embedding_dim())?;
        let embedding = embedder.embed(&audio, args.dimension)?;
        let store = ReadOnlyStore::open(&args.data_dir)?;
        let index = Index::build_read_only(&store)?;
        let mut request = SearchRequest::new(SearchMode::Vector, args.limit);
        request.offset = args.offset;
        request.filters = args.filters;
        request.text = None;
        request.embedding = Some(&embedding);
        let hits = index.search(&request)?;
        return write_hits(stdout, &hits);
    }
    let query_text = match args.input {
        SearchInput::Text(text) => text,
        SearchInput::Stdin => read_stdin_bounded(MAX_STDIN_BYTES)?,
    };
    if query_text.len() > MAX_STDIN_BYTES {
        return Err(Error::limit_exceeded(format!(
            "query text is {} bytes, above the {MAX_STDIN_BYTES}-byte cap",
            query_text.len()
        )));
    }

    let store = ReadOnlyStore::open(&args.data_dir)?;
    let index = Index::build_read_only(&store)?;

    let embedding = if args.mode == SearchMode::Lexical {
        None
    } else if let Some(ids) = &args.ids {
        let model = TextModel::load(&args.model_dir)?;
        validate_dimension(args.dimension, model.config().embedding_dim())?;
        Some(model.embed_ids(ids, args.dimension)?)
    } else {
        let embedder = TextEmbedder::load(&args.model_dir)?;
        validate_dimension(args.dimension, embedder.model().config().embedding_dim())?;
        Some(embedder.embed(&query_text, Prompt::SearchQuery, args.dimension)?)
    };

    let mut request = SearchRequest::new(args.mode, args.limit);
    request.offset = args.offset;
    request.filters = args.filters;
    request.text = if args.mode == SearchMode::Vector {
        None
    } else {
        Some(query_text.as_str())
    };
    request.embedding = embedding.as_deref();

    let hits = index.search(&request)?;
    write_hits(stdout, &hits)
}

fn run_verify<W: Write>(args: VerifyArgs, stdout: &mut W) -> Result<()> {
    let verification = Store::verify(&args.data_dir)?;
    let mut object = Object::new();
    object.insert(
        "records".to_owned(),
        Value::Number(Number::Unsigned(verification.records)),
    );
    object.insert(
        "snapshot_records".to_owned(),
        Value::Number(Number::Unsigned(verification.snapshot_records)),
    );
    object.insert(
        "documents".to_owned(),
        Value::Number(Number::Unsigned(verification.documents as u64)),
    );
    object.insert(
        "chunks".to_owned(),
        Value::Number(Number::Unsigned(verification.chunks as u64)),
    );
    object.insert(
        "truncated_bytes".to_owned(),
        Value::Number(Number::Unsigned(verification.truncated_bytes)),
    );
    object.insert(
        "data_dir".to_owned(),
        Value::String(args.data_dir.display().to_string()),
    );
    write_json(stdout, &Value::Object(object))
}

fn run_compact<W: Write>(args: CompactArgs, stdout: &mut W) -> Result<()> {
    let mut store = Store::open(&args.data_dir)?;
    let compaction = store.compact()?;
    let mut object = Object::new();
    object.insert(
        "snapshot_records".to_owned(),
        Value::Number(Number::Unsigned(compaction.snapshot_records)),
    );
    object.insert(
        "snapshot_bytes".to_owned(),
        Value::Number(Number::Unsigned(compaction.snapshot_bytes)),
    );
    object.insert(
        "documents".to_owned(),
        Value::Number(Number::Unsigned(compaction.documents as u64)),
    );
    object.insert(
        "chunks".to_owned(),
        Value::Number(Number::Unsigned(compaction.chunks as u64)),
    );
    object.insert(
        "data_dir".to_owned(),
        Value::String(args.data_dir.display().to_string()),
    );
    write_json(stdout, &Value::Object(object))
}

fn validate_dimension(requested: usize, model_dimension: usize) -> Result<()> {
    if requested > model_dimension {
        return Err(Error::usage(format!(
            "dimension {requested} exceeds the model embedding dimension {model_dimension}"
        )));
    }
    Ok(())
}

fn write_vector<W: Write>(values: &[f32], stdout: &mut W) -> Result<()> {
    stdout.write_all(b"[")?;
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            stdout.write_all(b",")?;
        }
        write!(stdout, "{value}")?;
    }
    stdout.write_all(b"]\n")?;
    Ok(())
}

fn write_hits<W: Write>(stdout: &mut W, hits: &[SearchHit<'_>]) -> Result<()> {
    let mut values = Vec::with_capacity(hits.len());
    for hit in hits {
        let mut object = Object::new();
        object.insert(
            "chunk_id".to_owned(),
            Value::Number(Number::Unsigned(hit.chunk_id.get())),
        );
        object.insert(
            "document_id".to_owned(),
            Value::String(hit.document_id.as_str().to_owned()),
        );
        object.insert(
            "ordinal".to_owned(),
            Value::Number(Number::Unsigned(u64::from(hit.ordinal))),
        );
        object.insert("score".to_owned(), Value::Number(Number::Float(hit.score)));
        object.insert("lexical_rank".to_owned(), optional_rank(hit.lexical_rank));
        object.insert("vector_rank".to_owned(), optional_rank(hit.vector_rank));
        object.insert("title".to_owned(), optional_string(hit.title));
        object.insert("source".to_owned(), optional_string(hit.source));
        object.insert(
            "modality".to_owned(),
            Value::String(hit.modality.as_str().to_owned()),
        );
        object.insert("text".to_owned(), Value::String(hit.text.to_owned()));
        let mut metadata = Object::new();
        for (key, value) in hit.metadata.iter() {
            metadata.insert(key.to_owned(), Value::String(value.to_owned()));
        }
        object.insert("metadata".to_owned(), Value::Object(metadata));
        values.push(Value::Object(object));
    }
    write_json(stdout, &Value::Array(values))
}

fn optional_rank(value: Option<usize>) -> Value {
    match value {
        Some(value) => Value::Number(Number::Unsigned(value as u64)),
        None => Value::Null,
    }
}

fn optional_string(value: Option<&str>) -> Value {
    match value {
        Some(value) => Value::String(value.to_owned()),
        None => Value::Null,
    }
}

fn write_json<W: Write>(stdout: &mut W, value: &Value) -> Result<()> {
    let encoded = json::to_string(value)?;
    writeln!(stdout, "{encoded}")?;
    Ok(())
}

fn read_stdin_bounded(max_bytes: usize) -> Result<String> {
    let mut bytes = Vec::new();
    std::io::stdin()
        .lock()
        .take(max_bytes as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > max_bytes {
        return Err(Error::limit_exceeded(format!(
            "standard input exceeds the {max_bytes}-byte cap"
        )));
    }
    String::from_utf8(bytes).map_err(|_| Error::corrupt("standard input is not valid UTF-8"))
}

fn read_index_input(path: &Path) -> Result<String> {
    let bytes = if path == Path::new("-") {
        let mut bytes = Vec::new();
        std::io::stdin()
            .lock()
            .take(MAX_INDEX_FILE_BYTES + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() as u64 > MAX_INDEX_FILE_BYTES {
            return Err(Error::limit_exceeded(format!(
                "index input exceeds the {MAX_INDEX_FILE_BYTES}-byte cap"
            )));
        }
        bytes
    } else {
        read_file_bounded(path, MAX_INDEX_FILE_BYTES)?
    };
    String::from_utf8(bytes).map_err(|_| Error::corrupt("index input is not valid UTF-8"))
}

#[derive(Debug)]
struct PendingDocument {
    document: Document,
    text: String,
    image: Option<PathBuf>,
    audio: Option<PathBuf>,
    ids: Option<Vec<u32>>,
}

fn parse_index_documents(input: &str, ids_mode: bool) -> Result<Vec<PendingDocument>> {
    let mut documents = Vec::new();
    let mut seen = BTreeSet::new();

    for (line_index, raw_line) in input.split('\n').enumerate() {
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        if line.trim().is_empty() {
            continue;
        }
        if line.len() > MAX_INDEX_LINE_BYTES {
            return Err(Error::limit_exceeded(format!(
                "line {} is {} bytes, above the {MAX_INDEX_LINE_BYTES}-byte cap",
                line_index + 1,
                line.len()
            )));
        }
        if documents.len() >= MAX_INDEX_DOCUMENTS {
            return Err(Error::limit_exceeded(format!(
                "index input has more than {MAX_INDEX_DOCUMENTS} documents"
            )));
        }

        let value = json::parse(line)
            .map_err(|error| Error::corrupt(format!("line {}: {error}", line_index + 1)))?;
        let object = value.as_object().ok_or_else(|| {
            Error::usage(format!("line {} must be a JSON object", line_index + 1))
        })?;

        for key in object.keys() {
            if !allowed_index_field(key, ids_mode) {
                return Err(Error::usage(format!(
                    "line {} has unknown field {key}",
                    line_index + 1
                )));
            }
        }

        let id = required_string(object, "id", line_index)?;
        let document_id = DocumentId::new(id)?;
        if !seen.insert(document_id.clone()) {
            return Err(Error::usage(format!(
                "duplicate document id {}",
                document_id
            )));
        }

        let text = match object.get("text") {
            Some(value) => Some(
                value
                    .as_str()
                    .ok_or_else(|| {
                        Error::usage(format!(
                            "line {} field text must be a string",
                            line_index + 1
                        ))
                    })?
                    .to_owned(),
            ),
            None => None,
        };
        let image = optional_string_field(object, "image", line_index)?.map(PathBuf::from);
        let audio = optional_string_field(object, "audio", line_index)?.map(PathBuf::from);
        let modality_count = usize::from(text.is_some())
            + usize::from(image.is_some())
            + usize::from(audio.is_some());
        if modality_count != 1 {
            return Err(Error::usage(format!(
                "line {} must have exactly one of text, image, or audio",
                line_index + 1
            )));
        }
        if ids_mode && (image.is_some() || audio.is_some()) {
            return Err(Error::usage(format!(
                "line {} cannot combine --ids with an image or audio",
                line_index + 1
            )));
        }

        let (mut document, text) = match (&image, &audio, text) {
            (Some(_), None, None) => (Document::new(document_id, Modality::Image), String::new()),
            (None, Some(_), None) => (Document::new(document_id, Modality::Audio), String::new()),
            (None, None, Some(text)) => (Document::new(document_id, Modality::Text), text),
            _ => {
                return Err(Error::internal(
                    "index parser accepted an invalid modality combination",
                ));
            }
        };

        if let Some(title) = optional_string_field(object, "title", line_index)? {
            document = document.with_title(title)?;
        }
        let source = optional_string_field(object, "source", line_index)?;
        if let Some(source) = source {
            document = document.with_source(source)?;
        } else if let Some(image) = &image {
            document = document.with_source(
                image
                    .to_str()
                    .ok_or_else(|| Error::usage("image path is not valid UTF-8"))?,
            )?;
        } else if let Some(audio) = &audio {
            document = document.with_source(
                audio
                    .to_str()
                    .ok_or_else(|| Error::usage("audio path is not valid UTF-8"))?,
            )?;
        }
        let metadata = parse_metadata(object.get("metadata"), line_index)?;
        document = document.with_metadata(metadata);

        let ids = if ids_mode {
            Some(parse_json_ids(object.get("ids"), line_index)?)
        } else {
            None
        };

        documents.push(PendingDocument {
            document,
            text,
            image,
            audio,
            ids,
        });
    }

    if documents.is_empty() {
        return Err(Error::usage("index input contains no documents"));
    }
    Ok(documents)
}

fn allowed_index_field(key: &str, ids_mode: bool) -> bool {
    matches!(
        key,
        "id" | "text" | "image" | "audio" | "title" | "source" | "metadata"
    ) || (ids_mode && key == "ids")
}

fn required_string<'a>(object: &'a Object, key: &str, line_index: usize) -> Result<&'a str> {
    object.get(key).and_then(Value::as_str).ok_or_else(|| {
        Error::usage(format!(
            "line {} field {key} must be a string",
            line_index + 1
        ))
    })
}

fn optional_string_field<'a>(
    object: &'a Object,
    key: &str,
    line_index: usize,
) -> Result<Option<&'a str>> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value)),
        Some(other) => Err(Error::usage(format!(
            "line {} field {key} must be a string or null, got {}",
            line_index + 1,
            other.type_name()
        ))),
    }
}

fn parse_metadata(value: Option<&Value>, line_index: usize) -> Result<Metadata> {
    let mut metadata = Metadata::new();
    let Some(value) = value else {
        return Ok(metadata);
    };
    let object = value.as_object().ok_or_else(|| {
        Error::usage(format!(
            "line {} field metadata must be an object",
            line_index + 1
        ))
    })?;
    for (key, value) in object {
        let value = value.as_str().ok_or_else(|| {
            Error::usage(format!(
                "line {} metadata value for {key} must be a string",
                line_index + 1
            ))
        })?;
        metadata.insert(key.clone(), value.to_owned())?;
    }
    Ok(metadata)
}

fn parse_json_ids(value: Option<&Value>, line_index: usize) -> Result<Vec<u32>> {
    let value = value.ok_or_else(|| {
        Error::usage(format!(
            "line {} requires an ids array in --ids mode",
            line_index + 1
        ))
    })?;
    let array = value.as_array().ok_or_else(|| {
        Error::usage(format!(
            "line {} field ids must be an array",
            line_index + 1
        ))
    })?;
    if array.is_empty() {
        return Err(Error::usage(format!(
            "line {} ids array must not be empty",
            line_index + 1
        )));
    }
    if array.len() > MAX_IDS {
        return Err(Error::limit_exceeded(format!(
            "line {} has {} ids, above the {MAX_IDS}-id cap",
            line_index + 1,
            array.len()
        )));
    }
    let mut ids = Vec::with_capacity(array.len());
    for value in array {
        let number = value.as_number().ok_or_else(|| {
            Error::usage(format!(
                "line {} ids entries must be integers",
                line_index + 1
            ))
        })?;
        let raw = number.as_u64().ok_or_else(|| {
            Error::usage(format!(
                "line {} ids entries must be non-negative integers",
                line_index + 1
            ))
        })?;
        let id = u32::try_from(raw).map_err(|_| {
            Error::limit_exceeded(format!(
                "line {} token id {raw} does not fit u32",
                line_index + 1
            ))
        })?;
        ids.push(id);
    }
    Ok(ids)
}

/// CLI entry point from the process's raw arguments.
///
/// The first argument is the program name, as with std::env::args_os; it is
/// ignored. The rest is parsed by parse then executed by run.
pub fn main_entry<I, S>(args: I, stdout: &mut impl Write) -> Result<()>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    let mut args = args.into_iter();
    let _program_name = args.next();
    let command = parse(args)?;
    run(command, stdout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json;
    use std::ffi::OsStr;
    use std::fs;
    use std::os::unix::ffi::OsStrExt;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir(name: &str) -> PathBuf {
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "seldon-cli-{}-{}-{unique}",
            std::process::id(),
            name
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create temp dir");
        path
    }

    fn parse_args(args: &[&str]) -> Result<Command> {
        parse(args.iter().copied())
    }

    fn parse_embed_args(args: &[&str]) -> EmbedArgs {
        match parse_args(args).expect("embed arguments must parse") {
            Command::Embed(args) => args,
            other => panic!("expected embed, got {other:?}"),
        }
    }

    fn parse_index_args(args: &[&str]) -> IndexArgs {
        match parse_args(args).expect("index arguments must parse") {
            Command::Index(args) => args,
            other => panic!("expected index, got {other:?}"),
        }
    }

    fn parse_search_args(args: &[&str]) -> SearchArgs {
        match parse_args(args).expect("search arguments must parse") {
            Command::Search(args) => args,
            other => panic!("expected search, got {other:?}"),
        }
    }

    #[test]
    fn accepts_version_aliases() {
        for args in [&["version"][..], &["--version"], &["-V"]] {
            assert_eq!(
                parse_args(args).expect("version must be accepted"),
                Command::Version
            );
        }
    }

    #[test]
    fn accepts_help_aliases() {
        for args in [&["help"][..], &["--help"], &["-h"]] {
            assert_eq!(
                parse_args(args).expect("help must be accepted"),
                Command::Help
            );
        }
    }

    #[test]
    fn rejects_missing_command() {
        let error = parse_args(&[]).expect_err("a missing command must fail");
        assert!(matches!(error, Error::Usage(_)));
        assert_eq!(error.exit_code(), 2);
        assert!(error.to_string().contains("no command"));
    }

    #[test]
    fn rejects_unknown_command() {
        let error = parse_args(&["serve"]).expect_err("serve does not exist yet");
        assert!(matches!(error, Error::Usage(_)));
        assert!(error.to_string().contains("unknown command"));
    }

    #[test]
    fn rejects_extra_arguments() {
        let error = parse_args(&["version", "extra"]).expect_err("extra arguments must fail");
        assert!(matches!(error, Error::Usage(_)));
        assert!(error.to_string().contains("extra arguments"));
    }

    #[test]
    fn rejects_non_utf8_argument() {
        let args = [OsStr::from_bytes(b"\xff")];
        let error = parse(args.iter().copied()).expect_err("a non-UTF-8 argument must fail");
        assert!(matches!(error, Error::Usage(_)));
        assert!(error.to_string().contains("non-UTF-8"));
    }

    #[test]
    fn parses_embed_defaults_and_ids() {
        let args = parse_embed_args(&["embed", "--ids", "1,5,7,9,2"]);
        assert_eq!(args.model_dir, default_model_dir());
        assert_eq!(args.prompt, Prompt::SearchQuery);
        assert_eq!(args.dimension, 768);
        assert_eq!(
            args.vision_budget,
            crate::media::preprocess::DEFAULT_MAX_SOFT_TOKENS
        );
        assert_eq!(args.input, EmbedInput::Ids(vec![1, 5, 7, 9, 2]));
    }

    #[test]
    fn parses_embed_text_options() {
        let args = parse_embed_args(&[
            "embed",
            "--model-dir",
            "fixtures/model/tiny_text",
            "--prompt",
            "Document",
            "--dim",
            "8",
            "hello",
            "world",
        ]);
        assert_eq!(args.model_dir, PathBuf::from("fixtures/model/tiny_text"));
        assert_eq!(args.prompt, Prompt::Document);
        assert_eq!(args.dimension, 8);
        assert_eq!(args.input, EmbedInput::Text("hello world".to_owned()));

        let explicit = parse_embed_args(&["embed", "--text", "hello"]);
        assert_eq!(explicit.input, EmbedInput::Text("hello".to_owned()));

        let stdin = parse_embed_args(&["embed"]);
        assert_eq!(stdin.input, EmbedInput::Stdin);

        let image = parse_embed_args(&[
            "embed",
            "--image",
            "fixtures/media/rgb_2x2.png",
            "--dim",
            "128",
            "--vision-tokens",
            "70",
        ]);
        assert_eq!(
            image.input,
            EmbedInput::Image(PathBuf::from("fixtures/media/rgb_2x2.png"))
        );
        assert_eq!(image.dimension, 128);
        assert_eq!(image.vision_budget, 70);
    }

    #[test]
    fn rejects_image_combined_with_text_or_ids() {
        for args in [
            &["embed", "--image", "image.png", "--text", "hello"][..],
            &["embed", "--image", "image.png", "hello"],
            &["embed", "--image", "image.png", "--ids", "1,2"],
        ] {
            let error = parse_args(args).expect_err("image input cannot be combined");
            assert!(matches!(error, Error::Usage(_)), "{error}");
            assert!(error.to_string().contains("--image"), "{error}");
        }
    }

    #[test]
    fn parses_image_documents_and_defaults_the_source() {
        let input = "{\"id\":\"img-1\",\"image\":\"fixtures/media/quadrants_16x16.png\",\"title\":\"Quadrants\"}\n";
        let documents = parse_index_documents(input, false).expect("image document");
        assert_eq!(documents.len(), 1);
        assert_eq!(documents[0].document.modality(), Modality::Image);
        assert_eq!(
            documents[0].document.source(),
            Some("fixtures/media/quadrants_16x16.png")
        );
        assert_eq!(documents[0].document.title(), Some("Quadrants"));
        assert_eq!(documents[0].text, "");
        assert!(documents[0].image.is_some());
        assert!(documents[0].ids.is_none());
    }

    #[test]
    fn rejects_invalid_image_documents() {
        for input in [
            "{\"id\":\"img\",\"text\":\"hello\",\"image\":\"x.png\"}",
            "{\"id\":\"img\"}",
        ] {
            let error =
                parse_index_documents(input, false).expect_err("invalid image document must fail");
            assert!(matches!(error, Error::Usage(_)), "{error}");
        }
        let error = parse_index_documents("{\"id\":\"img\",\"image\":\"x.png\"}", true)
            .expect_err("ids mode cannot combine with an image");
        assert!(matches!(error, Error::Usage(_)), "{error}");
    }

    #[test]
    fn parses_audio_embed_and_search_queries() {
        let command = parse_args(&[
            "embed",
            "--audio",
            "fixtures/media/audio/short_tone_100ms.wav",
            "--dim",
            "128",
        ])
        .expect("audio embed arguments");
        let Command::Embed(args) = command else {
            panic!("expected embed command");
        };
        assert_eq!(args.dimension, 128);
        assert_eq!(
            args.input,
            EmbedInput::Audio(PathBuf::from("fixtures/media/audio/short_tone_100ms.wav"))
        );

        let command = parse_args(&[
            "search",
            "--data-dir",
            "data",
            "--mode",
            "vector",
            "--audio",
            "fixtures/media/audio/rain_noise_2s.wav",
        ])
        .expect("audio search arguments");
        let Command::Search(args) = command else {
            panic!("expected search command");
        };
        assert_eq!(args.mode, SearchMode::Vector);
        assert_eq!(
            args.audio,
            Some(PathBuf::from("fixtures/media/audio/rain_noise_2s.wav"))
        );
        assert_eq!(args.image, None);
    }

    #[test]
    fn rejects_audio_combined_with_text_image_or_ids() {
        for args in [
            &["embed", "--audio", "a.wav", "--text", "hello"][..],
            &["embed", "--audio", "a.wav", "hello"],
            &["embed", "--audio", "a.wav", "--ids", "1,2"],
            &["embed", "--audio", "a.wav", "--image", "image.png"],
            &[
                "search",
                "--data-dir",
                "data",
                "--mode",
                "vector",
                "--audio",
                "a.wav",
                "hello",
            ],
            &[
                "search",
                "--data-dir",
                "data",
                "--mode",
                "vector",
                "--audio",
                "a.wav",
                "--ids",
                "1,2",
            ],
            &[
                "search",
                "--data-dir",
                "data",
                "--mode",
                "hybrid",
                "--audio",
                "a.wav",
            ],
        ] {
            let error = parse_args(args).expect_err("audio input cannot be combined");
            assert!(matches!(error, Error::Usage(_)), "args {args:?}: {error}");
            assert!(
                error.to_string().contains("--audio"),
                "args {args:?}: {error}"
            );
        }
    }

    #[test]
    fn parses_audio_documents_and_defaults_the_source() {
        let input = "{\"id\":\"aud-1\",\"audio\":\"fixtures/media/audio/rain_noise_2s.wav\",\"title\":\"Rain\"}\n";
        let documents = parse_index_documents(input, false).expect("audio document");
        assert_eq!(documents.len(), 1);
        assert_eq!(documents[0].document.modality(), Modality::Audio);
        assert_eq!(
            documents[0].document.source(),
            Some("fixtures/media/audio/rain_noise_2s.wav")
        );
        assert_eq!(documents[0].document.title(), Some("Rain"));
        assert_eq!(documents[0].text, "");
        assert!(documents[0].audio.is_some());
        assert!(documents[0].image.is_none());
        assert!(documents[0].ids.is_none());
    }

    #[test]
    fn rejects_invalid_audio_documents() {
        for input in [
            "{\"id\":\"aud\",\"text\":\"hello\",\"audio\":\"x.wav\"}",
            "{\"id\":\"aud\",\"image\":\"x.png\",\"audio\":\"x.wav\"}",
            "{\"id\":\"aud\"}",
        ] {
            let error =
                parse_index_documents(input, false).expect_err("invalid audio document must fail");
            assert!(matches!(error, Error::Usage(_)), "{error}");
        }
        let error = parse_index_documents("{\"id\":\"aud\",\"audio\":\"x.wav\"}", true)
            .expect_err("ids mode cannot combine with audio");
        assert!(matches!(error, Error::Usage(_)), "{error}");
    }

    #[test]
    fn rejects_invalid_embed_options() {
        for args in [
            &["embed", "--dim", "0"][..],
            &["embed", "--dim", "abc"],
            &["embed", "--prompt", "Unknown"],
            &["embed", "--ids", "1,,2"],
            &["embed", "--ids", "x"],
            &["embed", "--ids", "1", "--text", "hello"],
            &["embed", "--text", "hello", "world"],
            &["embed", "--unknown"],
            &["embed", "--model-dir"],
        ] {
            let error = parse_args(args).expect_err("invalid embed arguments must fail");
            assert!(matches!(error, Error::Usage(_)), "args {args:?}: {error}");
        }
    }

    #[test]
    fn parses_index_defaults_and_ids_mode() {
        let args = parse_index_args(&["index", "--data-dir", "data", "docs.jsonl"]);
        assert_eq!(args.data_dir, PathBuf::from("data"));
        assert_eq!(args.model_dir, default_model_dir());
        assert_eq!(args.dimension, 768);
        assert_eq!(
            args.vision_budget,
            crate::media::preprocess::DEFAULT_MAX_SOFT_TOKENS
        );
        assert_eq!(args.input, PathBuf::from("docs.jsonl"));
        assert!(!args.ids_mode);

        let args = parse_index_args(&[
            "index",
            "--data-dir",
            "data",
            "--model-dir",
            "fixtures/model/tiny_text",
            "--dim",
            "8",
            "--vision-tokens",
            "70",
            "--ids",
            "-",
        ]);
        assert_eq!(args.model_dir, PathBuf::from("fixtures/model/tiny_text"));
        assert_eq!(args.dimension, 8);
        assert_eq!(args.vision_budget, 70);
        assert_eq!(args.input, PathBuf::from("-"));
        assert!(args.ids_mode);
    }

    #[test]
    fn rejects_invalid_index_options() {
        for args in [
            &["index", "docs.jsonl"][..],
            &["index", "--data-dir", "data"],
            &["index", "--data-dir", "data", "a", "b"],
            &[
                "index",
                "--data-dir",
                "data",
                "--vision-tokens",
                "100",
                "docs.jsonl",
            ],
            &["index", "--data-dir", "data", "--dim", "0", "docs.jsonl"],
            &["index", "--data-dir", "data", "--unknown", "docs.jsonl"],
        ] {
            let error = parse_args(args).expect_err("invalid index arguments must fail");
            assert!(matches!(error, Error::Usage(_)), "args {args:?}: {error}");
        }
    }

    #[test]
    fn parses_search_defaults_and_filters() {
        let args = parse_search_args(&[
            "search",
            "--data-dir",
            "data",
            "--mode",
            "lexical",
            "--limit",
            "5",
            "--offset",
            "2",
            "--filter",
            "lang=en",
            "--filter",
            "kind=note",
            "rust",
            "language",
        ]);
        assert_eq!(args.data_dir, PathBuf::from("data"));
        assert_eq!(args.mode, SearchMode::Lexical);
        assert_eq!(args.limit, 5);
        assert_eq!(args.offset, 2);
        assert_eq!(
            args.filters.metadata(),
            &[
                ("lang".to_owned(), "en".to_owned()),
                ("kind".to_owned(), "note".to_owned())
            ]
        );
        assert_eq!(args.input, SearchInput::Text("rust language".to_owned()));
        assert_eq!(args.image, None);
        assert_eq!(
            args.vision_budget,
            crate::media::preprocess::DEFAULT_MAX_SOFT_TOKENS
        );
        assert_eq!(args.ids, None);

        let args = parse_search_args(&[
            "search",
            "--data-dir",
            "data",
            "--mode",
            "vector",
            "--ids",
            "1,2,3",
        ]);
        assert_eq!(args.mode, SearchMode::Vector);
        assert_eq!(args.input, SearchInput::Stdin);
        assert_eq!(args.image, None);
        assert_eq!(args.ids, Some(vec![1, 2, 3]));

        let args = parse_search_args(&[
            "search",
            "--data-dir",
            "data",
            "--mode",
            "vector",
            "--image",
            "fixtures/media/quadrants_16x16.png",
            "--vision-tokens",
            "70",
        ]);
        assert_eq!(
            args.image,
            Some(PathBuf::from("fixtures/media/quadrants_16x16.png"))
        );
        assert_eq!(args.vision_budget, 70);
        assert_eq!(args.input, SearchInput::Stdin);
    }

    #[test]
    fn rejects_invalid_search_options() {
        for args in [
            &["search", "query"][..],
            &["search", "--data-dir", "data", "--mode", "unknown", "query"],
            &["search", "--data-dir", "data", "--limit", "0", "query"],
            &["search", "--data-dir", "data", "--offset", "x", "query"],
            &["search", "--data-dir", "data", "--filter", "lang", "query"],
            &["search", "--data-dir", "data", "--filter", "=en", "query"],
            &[
                "search",
                "--data-dir",
                "data",
                "--mode",
                "lexical",
                "--ids",
                "1,2",
                "query",
            ],
            &["search", "--data-dir", "data", "--unknown", "query"],
            &[
                "search",
                "--data-dir",
                "data",
                "--mode",
                "vector",
                "--image",
                "image.png",
                "query",
            ],
            &[
                "search",
                "--data-dir",
                "data",
                "--mode",
                "vector",
                "--image",
                "image.png",
                "--ids",
                "1,2",
            ],
            &[
                "search",
                "--data-dir",
                "data",
                "--mode",
                "lexical",
                "--image",
                "image.png",
            ],
            &[
                "search",
                "--data-dir",
                "data",
                "--mode",
                "hybrid",
                "--image",
                "image.png",
            ],
            &[
                "search",
                "--data-dir",
                "data",
                "--mode",
                "vector",
                "--image",
                "image.png",
                "--vision-tokens",
                "100",
            ],
        ] {
            let error = parse_args(args).expect_err("invalid search arguments must fail");
            assert!(matches!(error, Error::Usage(_)), "args {args:?}: {error}");
        }
    }

    #[test]
    fn parses_verify_arguments() {
        match parse_args(&["verify", "--data-dir", "data"]).expect("verify") {
            Command::Verify(args) => assert_eq!(args.data_dir, PathBuf::from("data")),
            other => panic!("expected verify, got {other:?}"),
        }
        let error = parse_args(&["verify"]).expect_err("data dir required");
        assert!(matches!(error, Error::Usage(_)));
        let error =
            parse_args(&["verify", "--data-dir", "data", "extra"]).expect_err("extra argument");
        assert!(matches!(error, Error::Usage(_)));
    }

    #[test]
    fn parses_compact_arguments() {
        match parse_args(&["compact", "--data-dir", "data"]).expect("compact") {
            Command::Compact(args) => assert_eq!(args.data_dir, PathBuf::from("data")),
            other => panic!("expected compact, got {other:?}"),
        }
        let error = parse_args(&["compact"]).expect_err("data dir required");
        assert!(matches!(error, Error::Usage(_)));
        let error =
            parse_args(&["compact", "--data-dir", "data", "extra"]).expect_err("extra argument");
        assert!(matches!(error, Error::Usage(_)));
    }

    #[test]
    fn version_output_is_stable() {
        let mut output = Vec::new();
        run(Command::Version, &mut output).expect("writing must succeed");
        assert_eq!(
            String::from_utf8(output).expect("output must be UTF-8"),
            format!("{PROGRAM_NAME} {VERSION} ({TARGET_TRIPLE})\n")
        );
    }

    #[test]
    fn help_contains_usage_and_exit_codes() {
        let mut output = Vec::new();
        run(Command::Help, &mut output).expect("writing must succeed");
        let text = String::from_utf8(output).expect("output must be UTF-8");
        assert!(text.contains("USAGE:"));
        assert!(text.contains("EXIT CODES:"));
        assert!(text.contains("embed"));
        assert!(text.contains("index"));
        assert!(text.contains("search"));
        assert!(text.contains("verify"));
    }

    #[test]
    fn embed_ids_runs_the_tiny_model() {
        let model_dir = Path::new("fixtures/model/tiny_text");
        let model = TextModel::load(model_dir).expect("tiny model must load");
        let expected = model.embed_ids(&[1, 5, 7, 9, 2], 8).expect("embedding");

        let mut output = Vec::new();
        run(
            Command::Embed(EmbedArgs {
                model_dir: model_dir.to_path_buf(),
                prompt: Prompt::SearchQuery,
                dimension: 8,
                vision_budget: crate::media::preprocess::DEFAULT_MAX_SOFT_TOKENS,
                input: EmbedInput::Ids(vec![1, 5, 7, 9, 2]),
            }),
            &mut output,
        )
        .expect("embed must succeed");

        let text = String::from_utf8(output).expect("output must be UTF-8");
        let value = json::parse(text.trim()).expect("output must be JSON");
        let array = value.as_array().expect("output must be an array");
        assert_eq!(array.len(), expected.len());
        for (actual, expected) in array.iter().zip(&expected) {
            let actual = actual.as_number().expect("a number").as_f64() as f32;
            assert!((actual - expected).abs() <= 1e-6);
        }
    }

    #[test]
    fn main_entry_ignores_program_name() {
        let mut output = Vec::new();
        main_entry(["seldon-vault-lite", "version"], &mut output)
            .expect("the command must succeed");
        assert_eq!(
            String::from_utf8(output).expect("output must be UTF-8"),
            format!("{PROGRAM_NAME} {VERSION} ({TARGET_TRIPLE})\n")
        );
    }

    #[test]
    fn index_search_and_verify_work_with_the_tiny_model() {
        let dir = temp_dir("m3");
        let jsonl = dir.join("documents.jsonl");
        let documents = concat!(
            "{\"id\":\"doc-1\",\"text\":\"rust language systems\",\"ids\":[1,5,7,9,2]}\n",
            "{\"id\":\"doc-2\",\"text\":\"python language scripting\",\"ids\":[1,6,7,9,2]}\n",
            "{\"id\":\"doc-3\",\"text\":\"rust rust rust\",\"ids\":[1,5,7,9,3]}\n"
        );
        fs::write(&jsonl, documents).expect("write JSONL");
        let data_dir = dir.join("data");

        let mut output = Vec::new();
        run(
            Command::Index(IndexArgs {
                data_dir: data_dir.clone(),
                model_dir: PathBuf::from("fixtures/model/tiny_text"),
                dimension: 8,
                vision_budget: crate::media::preprocess::DEFAULT_MAX_SOFT_TOKENS,
                input: jsonl,
                ids_mode: true,
            }),
            &mut output,
        )
        .expect("index must succeed");
        let summary =
            json::parse(String::from_utf8(output).expect("UTF-8").trim()).expect("summary JSON");
        assert_eq!(
            summary
                .get("documents")
                .and_then(Value::as_number)
                .and_then(Number::as_u64),
            Some(3)
        );
        assert_eq!(
            summary
                .get("chunks")
                .and_then(Value::as_number)
                .and_then(Number::as_u64),
            Some(3)
        );
        assert_eq!(
            summary
                .get("dimension")
                .and_then(Value::as_number)
                .and_then(Number::as_u64),
            Some(8)
        );

        let mut output = Vec::new();
        run(
            Command::Search(SearchArgs {
                data_dir: data_dir.clone(),
                model_dir: PathBuf::from("fixtures/model/tiny_text"),
                dimension: 8,
                mode: SearchMode::Lexical,
                limit: 10,
                offset: 0,
                filters: Filters::new(),
                input: SearchInput::Text("rust".to_owned()),
                image: None,
                audio: None,
                vision_budget: crate::media::preprocess::DEFAULT_MAX_SOFT_TOKENS,
                ids: None,
            }),
            &mut output,
        )
        .expect("lexical search must succeed");
        let hits =
            json::parse(String::from_utf8(output).expect("UTF-8").trim()).expect("hits JSON");
        let hits = hits.as_array().expect("array");
        assert_eq!(hits.len(), 2);
        assert_eq!(
            hits[0]
                .get("chunk_id")
                .and_then(Value::as_number)
                .and_then(Number::as_u64),
            Some(3)
        );
        assert_eq!(
            hits[1]
                .get("chunk_id")
                .and_then(Value::as_number)
                .and_then(Number::as_u64),
            Some(1)
        );

        let mut output = Vec::new();
        run(
            Command::Search(SearchArgs {
                data_dir: data_dir.clone(),
                model_dir: PathBuf::from("fixtures/model/tiny_text"),
                dimension: 8,
                mode: SearchMode::Vector,
                limit: 10,
                offset: 0,
                filters: Filters::new(),
                input: SearchInput::Text(String::new()),
                image: None,
                audio: None,
                vision_budget: crate::media::preprocess::DEFAULT_MAX_SOFT_TOKENS,
                ids: Some(vec![1, 5, 7, 9, 2]),
            }),
            &mut output,
        )
        .expect("vector search must succeed");
        let hits =
            json::parse(String::from_utf8(output).expect("UTF-8").trim()).expect("hits JSON");
        assert_eq!(hits.as_array().expect("array").len(), 3);

        let mut output = Vec::new();
        run(
            Command::Search(SearchArgs {
                data_dir: data_dir.clone(),
                model_dir: PathBuf::from("fixtures/model/tiny_text"),
                dimension: 8,
                mode: SearchMode::Hybrid,
                limit: 10,
                offset: 0,
                filters: Filters::new(),
                input: SearchInput::Text("rust".to_owned()),
                image: None,
                audio: None,
                vision_budget: crate::media::preprocess::DEFAULT_MAX_SOFT_TOKENS,
                ids: Some(vec![1, 5, 7, 9, 2]),
            }),
            &mut output,
        )
        .expect("hybrid search must succeed");
        let hits =
            json::parse(String::from_utf8(output).expect("UTF-8").trim()).expect("hits JSON");
        assert!(!hits.as_array().expect("array").is_empty());

        let mut output = Vec::new();
        run(
            Command::Verify(VerifyArgs {
                data_dir: data_dir.clone(),
            }),
            &mut output,
        )
        .expect("verify must succeed");
        let verification = json::parse(String::from_utf8(output).expect("UTF-8").trim())
            .expect("verification JSON");
        assert_eq!(
            verification
                .get("records")
                .and_then(Value::as_number)
                .and_then(Number::as_u64),
            Some(3)
        );
        assert_eq!(
            verification
                .get("documents")
                .and_then(Value::as_number)
                .and_then(Number::as_u64),
            Some(3)
        );
        assert_eq!(
            verification
                .get("chunks")
                .and_then(Value::as_number)
                .and_then(Number::as_u64),
            Some(3)
        );
        assert_eq!(
            verification
                .get("truncated_bytes")
                .and_then(Value::as_number)
                .and_then(Number::as_u64),
            Some(0)
        );
        assert_eq!(
            verification
                .get("snapshot_records")
                .and_then(Value::as_number)
                .and_then(Number::as_u64),
            Some(0)
        );

        let mut output = Vec::new();
        run(
            Command::Compact(CompactArgs {
                data_dir: data_dir.clone(),
            }),
            &mut output,
        )
        .expect("compact must succeed");
        let compaction =
            json::parse(String::from_utf8(output).expect("UTF-8").trim()).expect("compaction JSON");
        assert_eq!(
            compaction
                .get("snapshot_records")
                .and_then(Value::as_number)
                .and_then(Number::as_u64),
            Some(3)
        );
        assert_eq!(
            compaction
                .get("documents")
                .and_then(Value::as_number)
                .and_then(Number::as_u64),
            Some(3)
        );
        assert_eq!(
            compaction
                .get("chunks")
                .and_then(Value::as_number)
                .and_then(Number::as_u64),
            Some(3)
        );
        assert!(
            compaction
                .get("snapshot_bytes")
                .and_then(Value::as_number)
                .and_then(Number::as_u64)
                .unwrap_or(0)
                > 0
        );

        let mut output = Vec::new();
        run(
            Command::Verify(VerifyArgs {
                data_dir: data_dir.clone(),
            }),
            &mut output,
        )
        .expect("verify after compact");
        let verification = json::parse(String::from_utf8(output).expect("UTF-8").trim())
            .expect("verification JSON");
        assert_eq!(
            verification
                .get("records")
                .and_then(Value::as_number)
                .and_then(Number::as_u64),
            Some(0)
        );
        assert_eq!(
            verification
                .get("snapshot_records")
                .and_then(Value::as_number)
                .and_then(Number::as_u64),
            Some(3)
        );

        let writer = Store::open(&data_dir).expect("writer stays open");
        let mut output = Vec::new();
        run(
            Command::Search(SearchArgs {
                data_dir: data_dir.clone(),
                model_dir: PathBuf::from("fixtures/model/tiny_text"),
                dimension: 8,
                mode: SearchMode::Lexical,
                limit: 10,
                offset: 0,
                filters: Filters::new(),
                input: SearchInput::Text("rust".to_owned()),
                image: None,
                audio: None,
                vision_budget: crate::media::preprocess::DEFAULT_MAX_SOFT_TOKENS,
                ids: None,
            }),
            &mut output,
        )
        .expect("read-only search while a writer is open");
        let hits =
            json::parse(String::from_utf8(output).expect("UTF-8").trim()).expect("hits JSON");
        assert_eq!(hits.as_array().expect("array").len(), 2);
        drop(writer);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn index_rejects_malformed_jsonl_before_writing() {
        let dir = temp_dir("malformed");
        let jsonl = dir.join("documents.jsonl");
        fs::write(&jsonl, "{\"id\":\"doc-1\"}\n").expect("write");
        let data_dir = dir.join("data");
        let mut output = Vec::new();
        let error = run(
            Command::Index(IndexArgs {
                data_dir: data_dir.clone(),
                model_dir: PathBuf::from("fixtures/model/tiny_text"),
                dimension: 8,
                vision_budget: crate::media::preprocess::DEFAULT_MAX_SOFT_TOKENS,
                input: jsonl,
                ids_mode: true,
            }),
            &mut output,
        )
        .expect_err("missing text must fail");
        assert!(matches!(error, Error::Usage(_)), "{error}");
        assert!(!data_dir.exists());
        let _ = fs::remove_dir_all(&dir);
    }
}
