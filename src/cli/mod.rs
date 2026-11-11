//! Argument parsing and command execution for the CLI.
//!
//! M2 adds the `embed` command. It loads the frozen text model from a model
//! directory and prints one JSON vector. Text input is tokenized with the
//! official prompt prefix; token ids can also be supplied directly for
//! diagnostics and tests. No stub simulates a modality that is not
//! implemented.

use std::ffi::OsString;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::model::prompt::Prompt;
use crate::model::text::{TextEmbedder, TextModel};
use crate::{MODEL_CACHE_DIR, MODEL_REVISION, PROGRAM_NAME, TARGET_TRIPLE, VERSION};

/// Maximum number of bytes read from standard input by `embed`.
const MAX_STDIN_BYTES: usize = 1024 * 1024;

/// Help text displayed by the `help` command.
const HELP: &str = concat!(
    env!("CARGO_PKG_NAME"),
    r#" — local multimodal embedding and search engine

USAGE:
    "#,
    env!("CARGO_PKG_NAME"),
    r#" <COMMAND>

COMMANDS:
    embed      Embeds text or token ids and prints a JSON vector
    version    Prints the version and build target
    help       Prints this help

EMBED:
    embed [--model-dir <DIR>] [--prompt <NAME>] [--dim <N>]
          [--text <TEXT> | --ids <ID,ID,...>] [TEXT...]

    --model-dir   Model directory (default: .model-cache/<revision>)
    --prompt      SearchQuery, Document, Classification, Clustering,
                  CodeRetrieval, FactChecking, SentenceSimilarity, or
                  QuestionAnswering (default: SearchQuery)
    --dim         Output dimension, from 1 to the model embedding dimension
                  (default: 768; MRL dimensions are 768, 512, 256, 128)
    --text        Text to embed; if omitted, positional TEXT is joined with
                  spaces, and if no text is given standard input is read
    --ids         Comma-separated token ids; bypasses the tokenizer

OPTIONS:
    -V, --version    Equivalent to `version`
    -h, --help       Equivalent to `help`

EXIT CODES:
    0    success
    1    execution error
    2    usage error

The `serve`, `index`, `search`, and `verify` commands will be added
with milestones M3 to M6. No command is simulated in the meantime.
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
}

/// Arguments of the `embed` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedArgs {
    /// Directory containing config.json and model.safetensors, and for text
    /// input also tokenizer.json.
    pub model_dir: PathBuf,
    /// Official prompt role applied before tokenization.
    pub prompt: Prompt,
    /// Requested Matryoshka dimension.
    pub dimension: usize,
    /// Input source.
    pub input: EmbedInput,
}

/// Input source of the `embed` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbedInput {
    /// Raw text, prefixed with the official prompt before tokenization.
    Text(String),
    /// Token ids used directly, without the tokenizer or prompt prefix.
    Ids(Vec<u32>),
    /// Text read from standard input.
    Stdin,
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
            "no command provided; use `--help` to display help",
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
        other => Err(Error::usage(format!(
            "unknown command `{other}`; use `--help`"
        ))),
    }
}

fn reject_extra(args: &[String], command: &str) -> Result<()> {
    if args.is_empty() {
        Ok(())
    } else {
        Err(Error::usage(format!("extra arguments after `{command}`")))
    }
}

fn parse_embed(args: Vec<String>) -> Result<Command> {
    let mut model_dir = default_model_dir();
    let mut prompt = Prompt::SearchQuery;
    let mut dimension = 768usize;
    let mut ids = None;
    let mut explicit_text = None;
    let mut text_parts = Vec::new();
    let mut index = 0;

    while index < args.len() {
        let argument = args[index].as_str();
        match argument {
            "--model-dir" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| Error::usage("--model-dir requires a value"))?;
                model_dir = PathBuf::from(value);
            }
            "--prompt" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| Error::usage("--prompt requires a value"))?;
                prompt = Prompt::from_name(value).ok_or_else(|| {
                    Error::usage(format!(
                        "unknown prompt `{value}`; use SearchQuery, Document, Classification, Clustering, CodeRetrieval, FactChecking, SentenceSimilarity, or QuestionAnswering"
                    ))
                })?;
            }
            "--dim" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| Error::usage("--dim requires a value"))?;
                dimension = value
                    .parse()
                    .map_err(|_| Error::usage(format!("invalid dimension `{value}`")))?;
                if dimension == 0 {
                    return Err(Error::usage("dimension must be positive"));
                }
            }
            "--ids" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| Error::usage("--ids requires a value"))?;
                if ids.is_some() {
                    return Err(Error::usage("--ids was provided more than once"));
                }
                ids = Some(parse_ids(value)?);
            }
            "--text" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| Error::usage("--text requires a value"))?;
                if explicit_text.is_some() {
                    return Err(Error::usage("--text was provided more than once"));
                }
                explicit_text = Some(value.clone());
            }
            "-h" | "--help" => return Ok(Command::Help),
            other if other.starts_with('-') => {
                return Err(Error::usage(format!("unknown option `{other}`")));
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

    let input = if let Some(ids) = ids {
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
            .map_err(|_| Error::usage(format!("invalid token id `{part}`")))?;
        ids.push(id);
    }
    if ids.is_empty() {
        return Err(Error::usage("--ids requires at least one id"));
    }
    Ok(ids)
}

/// Executes a command and writes its output to `stdout`.
///
/// # Errors
///
/// Returns a typed error for a usage problem, a model loading failure, a
/// numerical failure, or a write failure such as a closed pipe.
pub fn run<W: Write>(command: Command, stdout: &mut W) -> Result<()> {
    match command {
        Command::Version => writeln!(stdout, "{PROGRAM_NAME} {VERSION} ({TARGET_TRIPLE})")?,
        Command::Help => stdout.write_all(HELP.as_bytes())?,
        Command::Embed(args) => run_embed(args, stdout)?,
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
        EmbedInput::Stdin => {
            let mut text = String::new();
            std::io::stdin()
                .lock()
                .take(MAX_STDIN_BYTES as u64 + 1)
                .read_to_string(&mut text)?;
            if text.len() > MAX_STDIN_BYTES {
                return Err(Error::limit_exceeded(format!(
                    "standard input exceeds the {MAX_STDIN_BYTES}-byte cap"
                )));
            }
            let embedder = TextEmbedder::load(&args.model_dir)?;
            validate_dimension(args.dimension, embedder.model().config().embedding_dim())?;
            let embedding = embedder.embed(&text, args.prompt, args.dimension)?;
            write_vector(&embedding, stdout)?;
        }
    }
    Ok(())
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
    stdout.write_all(
        b"]
",
    )?;
    Ok(())
}

/// CLI entry point from the process's raw arguments.
///
/// The first argument is the program name, as with
/// [`std::env::args_os`]; it is ignored. The rest is parsed by
/// [`parse`] then executed by [`run`].
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
    use std::os::unix::ffi::OsStrExt;

    fn parse_args(args: &[&str]) -> Result<Command> {
        parse(args.iter().copied())
    }

    fn parse_embed_args(args: &[&str]) -> EmbedArgs {
        match parse_args(args).expect("embed arguments must parse") {
            Command::Embed(args) => args,
            other => panic!("expected embed, got {other:?}"),
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
}
