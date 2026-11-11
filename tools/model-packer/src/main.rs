//! Rust std packer for the frozen Seldon Vault Lite model bundle.
//!
//! The packer reads the frozen manifest, validates every selected file's size
//! and SHA-256, writes the versioned container described in `src/bundle`, and
//! verifies the result before reporting success. It is a development tool, not
//! part of the delivered runtime binary.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use seldon_vault_lite::binary::sha256::Sha256;
use seldon_vault_lite::bundle::{Bundle, BundleSource, SectionRole, write_bundle};
use seldon_vault_lite::error::{Error, Result};
use seldon_vault_lite::json::{self, Number, Value};

/// Files required for the text, image, and audio embedders.
const REQUIRED_FILES: &[&str] = &["config.json", "model.safetensors", "tokenizer.json"];

/// Small files that document the official processors and prompts.
const OPTIONAL_FILES: &[&str] = &[
    "config_sentence_transformers.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.model",
    "1_Pooling/config.json",
    "2_Normalize/config.json",
];

/// One manifest file entry.
struct ManifestEntry {
    path: String,
    size: u64,
    sha256: String,
}

/// The frozen manifest fields used by the packer.
struct Manifest {
    revision: String,
    files: Vec<ManifestEntry>,
}

/// One selected file and its validated fingerprint.
struct PackFile {
    path: String,
    role: SectionRole,
    size: u64,
    sha256: [u8; 32],
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(i32::from(error.exit_code()));
    }
}

fn run() -> Result<()> {
    let mut arguments = std::env::args().skip(1);
    let command = arguments
        .next()
        .ok_or_else(|| Error::usage("missing command; use pack, verify, list, or version"))?;
    match command.as_str() {
        "pack" => {
            let mut manifest_path: Option<PathBuf> = None;
            let mut model_dir: Option<PathBuf> = None;
            let mut output: Option<PathBuf> = None;
            while let Some(argument) = arguments.next() {
                match argument.as_str() {
                    "--manifest" => {
                        manifest_path =
                            Some(PathBuf::from(next_value(&mut arguments, "--manifest")?));
                    }
                    "--model-dir" => {
                        model_dir = Some(PathBuf::from(next_value(&mut arguments, "--model-dir")?));
                    }
                    "--output" => {
                        output = Some(PathBuf::from(next_value(&mut arguments, "--output")?));
                    }
                    other => {
                        return Err(Error::usage(format!("unknown pack argument {other}")));
                    }
                }
            }
            let manifest_path =
                manifest_path.ok_or_else(|| Error::usage("pack requires --manifest <path>"))?;
            let model_dir =
                model_dir.ok_or_else(|| Error::usage("pack requires --model-dir <path>"))?;
            let output = output.ok_or_else(|| Error::usage("pack requires --output <path>"))?;
            pack(&manifest_path, &model_dir, &output)
        }
        "verify" => {
            let path = arguments
                .next()
                .ok_or_else(|| Error::usage("verify requires a bundle path"))?;
            if arguments.next().is_some() {
                return Err(Error::usage("verify accepts exactly one bundle path"));
            }
            verify(Path::new(&path))
        }
        "list" => {
            let path = arguments
                .next()
                .ok_or_else(|| Error::usage("list requires a bundle path"))?;
            if arguments.next().is_some() {
                return Err(Error::usage("list accepts exactly one bundle path"));
            }
            list(Path::new(&path))
        }
        "version" | "--version" | "-V" => {
            println!("model-packer {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        "help" | "--help" | "-h" => {
            print_help();
            Ok(())
        }
        other => Err(Error::usage(format!("unknown command {other}"))),
    }
}

fn next_value(arguments: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    arguments
        .next()
        .ok_or_else(|| Error::usage(format!("{flag} requires a value")))
}

fn print_help() {
    println!(
        "model-packer {}\n\n\
         Usage:\n  \
         model-packer pack --manifest <manifest.json> --model-dir <dir> --output <bundle>\n  \
         model-packer verify <bundle>\n  \
         model-packer list <bundle>\n  \
         model-packer version\n\n\
         The packer validates the frozen manifest size and SHA-256 for every\n\
         selected file, writes the versioned bundle, and verifies it.",
        env!("CARGO_PKG_VERSION")
    );
}

fn pack(manifest_path: &Path, model_dir: &Path, output: &Path) -> Result<()> {
    let manifest = load_manifest(manifest_path)?;
    if manifest.revision != seldon_vault_lite::MODEL_REVISION {
        return Err(Error::corrupt(format!(
            "manifest revision {} does not match the frozen revision {}",
            manifest.revision,
            seldon_vault_lite::MODEL_REVISION
        )));
    }

    let mut selected = Vec::new();
    for path in REQUIRED_FILES.iter().chain(OPTIONAL_FILES.iter()) {
        let entry = manifest
            .files
            .iter()
            .find(|entry| entry.path == *path)
            .ok_or_else(|| {
                if REQUIRED_FILES.contains(path) {
                    Error::corrupt(format!("manifest is missing required file {path}"))
                } else {
                    Error::unsupported(format!("manifest has no optional file {path}"))
                }
            });
        let entry = match entry {
            Ok(entry) => entry,
            Err(Error::Unsupported(_)) => continue,
            Err(error) => return Err(error),
        };
        let file_path = model_dir.join(&entry.path);
        let metadata = std::fs::metadata(&file_path).map_err(|error| {
            Error::corrupt(format!("cannot read {}: {error}", file_path.display()))
        })?;
        if !metadata.is_file() {
            return Err(Error::corrupt(format!(
                "{} is not a regular file",
                file_path.display()
            )));
        }
        if metadata.len() != entry.size {
            return Err(Error::corrupt(format!(
                "{} has {} bytes, manifest declares {}",
                file_path.display(),
                metadata.len(),
                entry.size
            )));
        }
        let digest = file_sha256(&file_path, entry.size)?;
        let digest_hex = hex(&digest);
        if !digest_hex.eq_ignore_ascii_case(&entry.sha256) {
            return Err(Error::corrupt(format!(
                "{} SHA-256 {} does not match manifest {}",
                file_path.display(),
                digest_hex,
                entry.sha256
            )));
        }
        selected.push(PackFile {
            path: entry.path.clone(),
            role: role_for(&entry.path),
            size: entry.size,
            sha256: digest,
        });
    }
    selected.sort_by(|left, right| left.path.cmp(&right.path));

    let mut handles = Vec::with_capacity(selected.len());
    for file in &selected {
        let path = model_dir.join(&file.path);
        handles.push(
            File::open(&path).map_err(|error| {
                Error::corrupt(format!("cannot open {}: {error}", path.display()))
            })?,
        );
    }

    let mut temp = output.to_path_buf();
    let file_name = temp
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| Error::usage("output path has no UTF-8 file name"))?;
    temp.set_file_name(format!("{file_name}.tmp"));
    let mut output_file = File::create(&temp)
        .map_err(|error| Error::corrupt(format!("cannot create {}: {error}", temp.display())))?;

    {
        let mut sources: Vec<BundleSource<'_>> = selected
            .iter()
            .zip(handles.iter_mut())
            .map(|(file, handle)| BundleSource {
                path: &file.path,
                role: file.role,
                size: file.size,
                sha256: file.sha256,
                reader: handle,
            })
            .collect();
        write_bundle(&mut sources, &mut output_file)?;
    }
    output_file
        .sync_all()
        .map_err(|error| Error::corrupt(format!("cannot sync {}: {error}", temp.display())))?;
    drop(output_file);
    std::fs::rename(&temp, output).map_err(|error| {
        Error::corrupt(format!(
            "cannot rename {} to {}: {error}",
            temp.display(),
            output.display()
        ))
    })?;

    let bytes = std::fs::read(output)
        .map_err(|error| Error::corrupt(format!("cannot read {}: {error}", output.display())))?;
    let bundle = Bundle::parse(&bytes)?;
    bundle.verify()?;
    println!(
        "packed {} sections ({} bytes) into {}",
        bundle.sections().len(),
        bytes.len(),
        output.display()
    );
    for section in bundle.sections() {
        println!(
            "  {:?} {:>10} {}",
            section.role(),
            section.len(),
            section.path()
        );
    }
    Ok(())
}

fn verify(path: &Path) -> Result<()> {
    let bytes = std::fs::read(path)
        .map_err(|error| Error::corrupt(format!("cannot read {}: {error}", path.display())))?;
    let bundle = Bundle::parse(&bytes)?;
    bundle.verify()?;
    println!(
        "verified {} sections ({} bytes) in {}",
        bundle.sections().len(),
        bytes.len(),
        path.display()
    );
    Ok(())
}

fn list(path: &Path) -> Result<()> {
    let bytes = std::fs::read(path)
        .map_err(|error| Error::corrupt(format!("cannot read {}: {error}", path.display())))?;
    let bundle = Bundle::parse(&bytes)?;
    for section in bundle.sections() {
        println!(
            "{:?}\t{}\t{}",
            section.role(),
            section.len(),
            section.path()
        );
    }
    Ok(())
}

fn load_manifest(path: &Path) -> Result<Manifest> {
    let bytes = std::fs::read(path)
        .map_err(|error| Error::corrupt(format!("cannot read {}: {error}", path.display())))?;
    let text = std::str::from_utf8(&bytes)
        .map_err(|_| Error::corrupt(format!("{} is not valid UTF-8", path.display())))?;
    let value = json::parse(text).map_err(|error| {
        Error::corrupt(format!("{} is not valid JSON: {error}", path.display()))
    })?;
    let revision = value
        .get("revision")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::corrupt("manifest has no revision string"))?
        .to_owned();
    let files = value
        .get("files")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::corrupt("manifest has no files array"))?;
    let mut entries = Vec::with_capacity(files.len());
    for file in files {
        let path = file
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::corrupt("manifest file has no path string"))?
            .to_owned();
        let size = file
            .get("size")
            .and_then(Value::as_number)
            .and_then(Number::as_u64)
            .ok_or_else(|| Error::corrupt(format!("manifest file {path} has no size")))?;
        let sha256 = file
            .get("sha256")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::corrupt(format!("manifest file {path} has no sha256")))?;
        if sha256.len() != 64 {
            return Err(Error::corrupt(format!(
                "manifest file {path} sha256 is not 64 characters"
            )));
        }
        entries.push(ManifestEntry {
            path,
            size,
            sha256: sha256.to_owned(),
        });
    }
    Ok(Manifest {
        revision,
        files: entries,
    })
}

fn file_sha256(path: &Path, expected_size: u64) -> Result<[u8; 32]> {
    let mut file = File::open(path)
        .map_err(|error| Error::corrupt(format!("cannot open {}: {error}", path.display())))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut total = 0u64;
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| Error::corrupt(format!("cannot read {}: {error}", path.display())))?;
        if read == 0 {
            break;
        }
        total = total
            .checked_add(read as u64)
            .ok_or_else(|| Error::limit_exceeded("file length overflow"))?;
        hasher.update(&buffer[..read]);
    }
    if total != expected_size {
        return Err(Error::corrupt(format!(
            "{} has {total} bytes, expected {expected_size}",
            path.display()
        )));
    }
    Ok(hasher.finalize())
}

fn role_for(path: &str) -> SectionRole {
    if path.ends_with("model.safetensors") {
        SectionRole::Weights
    } else if path.contains("tokenizer") {
        SectionRole::Tokenizer
    } else if path.contains("processor") || path.contains("preprocessor") {
        SectionRole::Processor
    } else if path.ends_with("config.json") || path.contains("config_sentence") {
        SectionRole::Config
    } else {
        SectionRole::Other
    }
}

fn hex(digest: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in digest {
        output.push_str(&format!("{byte:02x}"));
    }
    output
}
