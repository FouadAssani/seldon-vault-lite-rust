//! Versioned, bounded model container for the final single-executable build.
//!
//! The container freezes the model files (configuration, tokenizer, processors,
//! and weights) into one deterministic byte stream with a header CRC-32, a
//! sorted section table, per-section SHA-256 digests, and a catalog digest.
//! The development loaders still read a model directory; the final binary will
//! embed this container with `include_bytes!` and read sections from the
//! embedded bytes.
//!
//! The format is deliberately simple and bounded:
//!
//! - magic `SVLMDL01`, format version 1;
//! - a 30-byte fixed header with header length, section count, total length,
//!   and the header CRC-32;
//! - one entry per section: path length, role, flags, offset, length, SHA-256,
//!   and the UTF-8 path;
//! - the section data in sorted path order;
//! - a 32-byte SHA-256 over the header and the ordered section digests.
//!
//! Paths are relative, normalized, and bounded. Offsets and lengths are
//! validated with checked arithmetic before any slice is created.

use std::io::{Read, Write};

use crate::binary::crc32::Crc32;
use crate::binary::sha256::{Sha256, sha256};
use crate::error::{Error, Result};

/// Magic bytes at the start of a model bundle.
pub const MAGIC: [u8; 8] = *b"SVLMDL01";

/// Bundle format version.
pub const FORMAT_VERSION: u16 = 1;

/// Maximum total bundle size (8 GiB).
///
/// The frozen BF16 weights are about 1.5 GB. The cap leaves room for a future
/// model revision while still rejecting a hostile length before allocation.
pub const MAX_BUNDLE_BYTES: u64 = 1 << 33;

/// Maximum header size (1 MiB).
pub const MAX_HEADER_BYTES: usize = 1 << 20;

/// Maximum number of sections.
pub const MAX_SECTIONS: usize = 1024;

/// Maximum path length in bytes.
pub const MAX_PATH_BYTES: usize = 4096;

/// Fixed header size before the section table.
const FIXED_HEADER_BYTES: usize = 30;

/// Fixed section-entry size before the path bytes.
const SECTION_ENTRY_BYTES: usize = 52;

/// Catalog digest size at the end of the bundle.
const TRAILER_BYTES: usize = 32;

/// Offset of the header CRC-32 field.
const HEADER_CRC_OFFSET: usize = 26;

/// End offset of the header CRC-32 field.
const HEADER_CRC_END: usize = 30;

/// Role of one section in the model bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionRole {
    /// Frozen model manifest.
    Manifest,
    /// Text, vision, or audio configuration.
    Config,
    /// Tokenizer model or tokenizer configuration.
    Tokenizer,
    /// Image or audio processor configuration.
    Processor,
    /// Safetensors weights.
    Weights,
    /// Any other file needed by the engine.
    Other,
}

impl SectionRole {
    /// Returns the stable on-disk role byte.
    pub fn as_u8(self) -> u8 {
        match self {
            Self::Manifest => 0,
            Self::Config => 1,
            Self::Tokenizer => 2,
            Self::Processor => 3,
            Self::Weights => 4,
            Self::Other => 5,
        }
    }

    /// Parses a stable on-disk role byte.
    fn from_u8(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::Manifest),
            1 => Ok(Self::Config),
            2 => Ok(Self::Tokenizer),
            3 => Ok(Self::Processor),
            4 => Ok(Self::Weights),
            5 => Ok(Self::Other),
            other => Err(Error::corrupt(format!(
                "unknown bundle section role {other}"
            ))),
        }
    }
}

/// One input section for [`Bundle::build`].
#[derive(Debug, Clone, Copy)]
pub struct SectionInput<'a> {
    /// Relative path inside the bundle.
    pub path: &'a str,
    /// Section role.
    pub role: SectionRole,
    /// Section bytes.
    pub data: &'a [u8],
}

/// One parsed section of a [`Bundle`].
#[derive(Debug, Clone, Copy)]
pub struct BundleSection<'a> {
    path: &'a str,
    role: SectionRole,
    sha256: [u8; 32],
    data: &'a [u8],
}

impl<'a> BundleSection<'a> {
    /// Relative path of the section.
    pub fn path(&self) -> &'a str {
        self.path
    }

    /// Role of the section.
    pub fn role(&self) -> SectionRole {
        self.role
    }

    /// SHA-256 recorded in the header.
    pub fn sha256(&self) -> [u8; 32] {
        self.sha256
    }

    /// Section bytes.
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// Section length in bytes.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true when the section is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// A parsed, structurally validated model bundle.
#[derive(Debug, Clone)]
pub struct Bundle<'a> {
    header_len: usize,
    sections: Vec<BundleSection<'a>>,
}

impl<'a> Bundle<'a> {
    /// Parses a bundle from bytes.
    ///
    /// # Errors
    ///
    /// Returns Error::Corrupt for a bad magic, version, length, CRC-32,
    /// section table, catalog digest, or path, and Error::LimitExceeded for a
    /// documented cap. This function does not hash the section data; call
    /// [`Bundle::verify`] for that.
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        if bytes.len() < FIXED_HEADER_BYTES + TRAILER_BYTES {
            return Err(Error::corrupt("bundle is shorter than its fixed header"));
        }
        if bytes[..MAGIC.len()] != MAGIC {
            return Err(Error::corrupt("bundle magic does not match SVLMDL01"));
        }
        let version = u16::from_le_bytes(
            bytes[8..10]
                .try_into()
                .map_err(|_| Error::corrupt("bundle version field is truncated"))?,
        );
        if version != FORMAT_VERSION {
            return Err(Error::unsupported(format!(
                "bundle format version {version} is not supported"
            )));
        }
        let header_len = u32::from_le_bytes(
            bytes[10..14]
                .try_into()
                .map_err(|_| Error::corrupt("bundle header length is truncated"))?,
        ) as usize;
        let section_count = u32::from_le_bytes(
            bytes[14..18]
                .try_into()
                .map_err(|_| Error::corrupt("bundle section count is truncated"))?,
        ) as usize;
        let total_len = u64::from_le_bytes(
            bytes[18..26]
                .try_into()
                .map_err(|_| Error::corrupt("bundle total length is truncated"))?,
        );

        if header_len < FIXED_HEADER_BYTES {
            return Err(Error::corrupt(
                "bundle header length is below the fixed header",
            ));
        }
        if header_len > MAX_HEADER_BYTES {
            return Err(Error::limit_exceeded(format!(
                "bundle header is {header_len} bytes, above the {MAX_HEADER_BYTES}-byte cap"
            )));
        }
        if section_count > MAX_SECTIONS {
            return Err(Error::limit_exceeded(format!(
                "bundle has {section_count} sections, above the {MAX_SECTIONS}-section cap"
            )));
        }
        if total_len != bytes.len() as u64 {
            return Err(Error::corrupt(format!(
                "bundle declares {total_len} bytes but has {}",
                bytes.len()
            )));
        }
        if total_len > MAX_BUNDLE_BYTES {
            return Err(Error::limit_exceeded(format!(
                "bundle is {total_len} bytes, above the {MAX_BUNDLE_BYTES}-byte cap"
            )));
        }
        if header_len
            .checked_add(TRAILER_BYTES)
            .is_none_or(|minimum| minimum > bytes.len())
        {
            return Err(Error::corrupt("bundle header overlaps the catalog digest"));
        }

        let stored_crc = u32::from_le_bytes(
            bytes[HEADER_CRC_OFFSET..HEADER_CRC_END]
                .try_into()
                .map_err(|_| Error::corrupt("bundle header CRC field is truncated"))?,
        );
        if stored_crc != header_crc(&bytes[..header_len]) {
            return Err(Error::corrupt("bundle header CRC-32 mismatch"));
        }

        let data_end = total_len - TRAILER_BYTES as u64;
        let mut cursor = FIXED_HEADER_BYTES;
        let mut entries: Vec<HeaderEntry<'a>> = Vec::with_capacity(section_count);
        let mut expected_offset = header_len as u64;
        let mut previous_path: Option<&str> = None;

        for _ in 0..section_count {
            let entry_end = cursor
                .checked_add(SECTION_ENTRY_BYTES)
                .ok_or_else(|| Error::limit_exceeded("bundle section table offset overflow"))?;
            if entry_end > header_len {
                return Err(Error::corrupt("bundle section table is truncated"));
            }
            let path_len = u16::from_le_bytes(
                bytes[cursor..cursor + 2]
                    .try_into()
                    .map_err(|_| Error::corrupt("bundle path length is truncated"))?,
            ) as usize;
            let role = SectionRole::from_u8(bytes[cursor + 2])?;
            let flags = bytes[cursor + 3];
            if flags != 0 {
                return Err(Error::unsupported(format!(
                    "bundle section flags {flags} are not supported"
                )));
            }
            let offset = u64::from_le_bytes(
                bytes[cursor + 4..cursor + 12]
                    .try_into()
                    .map_err(|_| Error::corrupt("bundle section offset is truncated"))?,
            );
            let length = u64::from_le_bytes(
                bytes[cursor + 12..cursor + 20]
                    .try_into()
                    .map_err(|_| Error::corrupt("bundle section length is truncated"))?,
            );
            let sha256: [u8; 32] = bytes[cursor + 20..cursor + 52]
                .try_into()
                .map_err(|_| Error::corrupt("bundle section digest is truncated"))?;
            cursor = entry_end;

            let path_end = cursor
                .checked_add(path_len)
                .ok_or_else(|| Error::limit_exceeded("bundle path offset overflow"))?;
            if path_end > header_len {
                return Err(Error::corrupt("bundle path is truncated"));
            }
            let path = std::str::from_utf8(&bytes[cursor..path_end])
                .map_err(|_| Error::corrupt("bundle path is not valid UTF-8"))?;
            validate_path(path)?;
            cursor = path_end;

            if let Some(previous) = previous_path
                && path <= previous
            {
                return Err(Error::corrupt(
                    "bundle section paths are not strictly sorted",
                ));
            }
            previous_path = Some(path);

            if offset != expected_offset {
                return Err(Error::corrupt(format!(
                    "bundle section {path} starts at {offset}, expected {expected_offset}"
                )));
            }
            let end = offset
                .checked_add(length)
                .ok_or_else(|| Error::limit_exceeded("bundle section end overflow"))?;
            if end > data_end {
                return Err(Error::corrupt(format!(
                    "bundle section {path} ends at {end}, beyond the data area {data_end}"
                )));
            }
            expected_offset = end;
            entries.push(HeaderEntry {
                path,
                role,
                offset,
                length,
                sha256,
            });
        }

        if cursor != header_len {
            return Err(Error::corrupt(
                "bundle section table does not end at the declared header length",
            ));
        }
        if expected_offset != data_end {
            return Err(Error::corrupt(
                "bundle sections do not cover the declared data area",
            ));
        }

        let digest = catalog_digest(&bytes[..header_len], &entries);
        if bytes[bytes.len() - TRAILER_BYTES..] != digest {
            return Err(Error::corrupt("bundle catalog digest mismatch"));
        }

        let mut sections = Vec::with_capacity(entries.len());
        for entry in entries {
            let offset = usize::try_from(entry.offset)
                .map_err(|_| Error::limit_exceeded("bundle section offset does not fit usize"))?;
            let length = usize::try_from(entry.length)
                .map_err(|_| Error::limit_exceeded("bundle section length does not fit usize"))?;
            let data = &bytes[offset..offset + length];
            sections.push(BundleSection {
                path: entry.path,
                role: entry.role,
                sha256: entry.sha256,
                data,
            });
        }

        Ok(Self {
            header_len,
            sections,
        })
    }

    /// Builds a deterministic bundle from input sections.
    ///
    /// Sections are sorted by path. Duplicate paths are rejected.
    ///
    /// # Errors
    ///
    /// Returns Error::Usage for a duplicate path, Error::Corrupt for an invalid
    /// path, and Error::LimitExceeded when a documented cap is exceeded.
    pub fn build(sections: &[SectionInput<'_>]) -> Result<Vec<u8>> {
        if sections.len() > MAX_SECTIONS {
            return Err(Error::limit_exceeded(format!(
                "bundle has {} sections, above the {MAX_SECTIONS}-section cap",
                sections.len()
            )));
        }
        let mut ordered = sections.to_vec();
        ordered.sort_by(|left, right| left.path.cmp(right.path));
        for pair in ordered.windows(2) {
            if pair[0].path == pair[1].path {
                return Err(Error::usage(format!(
                    "duplicate bundle section path {}",
                    pair[0].path
                )));
            }
        }

        let mut entries = Vec::with_capacity(ordered.len());
        for section in &ordered {
            let length = u64::try_from(section.data.len())
                .map_err(|_| Error::limit_exceeded("section length does not fit u64"))?;
            entries.push(HeaderEntry {
                path: section.path,
                role: section.role,
                offset: 0,
                length,
                sha256: sha256(section.data),
            });
        }
        let header = encode_header(&mut entries)?;

        let mut data_len = 0u64;
        for section in &ordered {
            data_len = data_len
                .checked_add(section.data.len() as u64)
                .ok_or_else(|| Error::limit_exceeded("bundle data length overflow"))?;
        }
        let total_len = (header.len() as u64)
            .checked_add(data_len)
            .and_then(|value| value.checked_add(TRAILER_BYTES as u64))
            .ok_or_else(|| Error::limit_exceeded("bundle total length overflow"))?;
        if total_len > MAX_BUNDLE_BYTES {
            return Err(Error::limit_exceeded(format!(
                "bundle would be {total_len} bytes, above the {MAX_BUNDLE_BYTES}-byte cap"
            )));
        }
        let capacity = usize::try_from(total_len)
            .map_err(|_| Error::limit_exceeded("bundle length does not fit usize"))?;
        let digest = catalog_digest(&header, &entries);

        let mut output = Vec::with_capacity(capacity);
        output.extend_from_slice(&header);
        for section in &ordered {
            output.extend_from_slice(section.data);
        }
        output.extend_from_slice(&digest);
        Ok(output)
    }

    /// Returns the declared header length.
    pub fn header_len(&self) -> usize {
        self.header_len
    }

    /// Returns the sections in sorted path order.
    pub fn sections(&self) -> &[BundleSection<'a>] {
        &self.sections
    }

    /// Looks up a section by exact relative path.
    pub fn section(&self, path: &str) -> Option<&BundleSection<'a>> {
        self.sections
            .binary_search_by(|section| section.path.cmp(path))
            .ok()
            .map(|index| &self.sections[index])
    }

    /// Looks up section bytes by exact relative path.
    pub fn data(&self, path: &str) -> Option<&'a [u8]> {
        self.section(path).map(BundleSection::data)
    }

    /// Verifies every section SHA-256.
    ///
    /// # Errors
    ///
    /// Returns Error::Corrupt when a section digest does not match its bytes.
    pub fn verify(&self) -> Result<()> {
        for section in &self.sections {
            let digest = sha256(section.data);
            if digest != section.sha256 {
                return Err(Error::corrupt(format!(
                    "bundle section {} SHA-256 mismatch",
                    section.path
                )));
            }
        }
        Ok(())
    }
}

/// One streaming input for [`write_bundle`].
///
/// The caller must provide the exact size and SHA-256 of the reader's content.
/// The writer re-hashes the bytes while copying and rejects a mismatch.
pub struct BundleSource<'a> {
    /// Relative path inside the bundle.
    pub path: &'a str,
    /// Section role.
    pub role: SectionRole,
    /// Declared section size in bytes.
    pub size: u64,
    /// Declared SHA-256 of the section bytes.
    pub sha256: [u8; 32],
    /// Reader positioned at the start of the section.
    pub reader: &'a mut dyn Read,
}

/// Writes a bundle from streaming sources without buffering the whole model.
///
/// Sources must be sorted by path and unique. The writer validates each
/// source's declared size and SHA-256 while copying, so a changed file is
/// rejected instead of being silently frozen.
///
/// # Errors
///
/// Returns Error::Usage for unsorted or duplicate paths, Error::Corrupt for a
/// short or mismatching source, Error::LimitExceeded for a documented cap, and
/// Error::Io for a write failure. A failure may leave a partial output; the
/// caller must discard it.
pub fn write_bundle<W: Write>(sources: &mut [BundleSource<'_>], output: &mut W) -> Result<()> {
    if sources.len() > MAX_SECTIONS {
        return Err(Error::limit_exceeded(format!(
            "bundle has {} sections, above the {MAX_SECTIONS}-section cap",
            sources.len()
        )));
    }
    let mut previous_path: Option<&str> = None;
    let mut entries = Vec::with_capacity(sources.len());
    for source in sources.iter() {
        validate_path(source.path)?;
        if let Some(previous) = previous_path
            && source.path <= previous
        {
            return Err(Error::usage(
                "bundle sources must be sorted by path and unique",
            ));
        }
        previous_path = Some(source.path);
        entries.push(HeaderEntry {
            path: source.path,
            role: source.role,
            offset: 0,
            length: source.size,
            sha256: source.sha256,
        });
    }
    let header = encode_header(&mut entries)?;

    let mut data_len = 0u64;
    for source in sources.iter() {
        data_len = data_len
            .checked_add(source.size)
            .ok_or_else(|| Error::limit_exceeded("bundle data length overflow"))?;
    }
    let total_len = (header.len() as u64)
        .checked_add(data_len)
        .and_then(|value| value.checked_add(TRAILER_BYTES as u64))
        .ok_or_else(|| Error::limit_exceeded("bundle total length overflow"))?;
    if total_len > MAX_BUNDLE_BYTES {
        return Err(Error::limit_exceeded(format!(
            "bundle would be {total_len} bytes, above the {MAX_BUNDLE_BYTES}-byte cap"
        )));
    }
    let digest = catalog_digest(&header, &entries);

    output.write_all(&header)?;
    let mut buffer = vec![0u8; 64 * 1024];
    for source in sources.iter_mut() {
        let mut hasher = Sha256::new();
        let mut remaining = source.size;
        while remaining > 0 {
            let chunk = remaining.min(buffer.len() as u64) as usize;
            let read = source.reader.read(&mut buffer[..chunk])?;
            if read == 0 {
                return Err(Error::corrupt(format!(
                    "bundle source {} ended before its declared {}-byte size",
                    source.path, source.size
                )));
            }
            hasher.update(&buffer[..read]);
            output.write_all(&buffer[..read])?;
            remaining -= read as u64;
        }
        if hasher.finalize() != source.sha256 {
            return Err(Error::corrupt(format!(
                "bundle source {} SHA-256 mismatch",
                source.path
            )));
        }
    }
    output.write_all(&digest)?;
    output.flush()?;
    Ok(())
}

/// One parsed section-table entry.
struct HeaderEntry<'a> {
    path: &'a str,
    role: SectionRole,
    offset: u64,
    length: u64,
    sha256: [u8; 32],
}

/// Encodes the header and assigns section offsets.
fn encode_header(entries: &mut [HeaderEntry<'_>]) -> Result<Vec<u8>> {
    if entries.len() > MAX_SECTIONS {
        return Err(Error::limit_exceeded(format!(
            "bundle has {} sections, above the {MAX_SECTIONS}-section cap",
            entries.len()
        )));
    }
    let mut header_len = FIXED_HEADER_BYTES;
    let mut data_len = 0u64;
    for entry in entries.iter() {
        validate_path(entry.path)?;
        let entry_len = SECTION_ENTRY_BYTES
            .checked_add(entry.path.len())
            .ok_or_else(|| Error::limit_exceeded("bundle section entry length overflow"))?;
        header_len = header_len
            .checked_add(entry_len)
            .ok_or_else(|| Error::limit_exceeded("bundle header length overflow"))?;
        data_len = data_len
            .checked_add(entry.length)
            .ok_or_else(|| Error::limit_exceeded("bundle data length overflow"))?;
    }
    if header_len > MAX_HEADER_BYTES {
        return Err(Error::limit_exceeded(format!(
            "bundle header is {header_len} bytes, above the {MAX_HEADER_BYTES}-byte cap"
        )));
    }
    let total_len = (header_len as u64)
        .checked_add(data_len)
        .and_then(|value| value.checked_add(TRAILER_BYTES as u64))
        .ok_or_else(|| Error::limit_exceeded("bundle total length overflow"))?;
    if total_len > MAX_BUNDLE_BYTES {
        return Err(Error::limit_exceeded(format!(
            "bundle would be {total_len} bytes, above the {MAX_BUNDLE_BYTES}-byte cap"
        )));
    }
    let header_len_u32 = u32::try_from(header_len)
        .map_err(|_| Error::limit_exceeded("bundle header length does not fit u32"))?;
    let section_count_u32 = u32::try_from(entries.len())
        .map_err(|_| Error::limit_exceeded("bundle section count does not fit u32"))?;

    let mut header = Vec::with_capacity(header_len);
    header.extend_from_slice(&MAGIC);
    header.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    header.extend_from_slice(&header_len_u32.to_le_bytes());
    header.extend_from_slice(&section_count_u32.to_le_bytes());
    header.extend_from_slice(&total_len.to_le_bytes());
    header.extend_from_slice(&[0u8; 4]);

    let mut offset = header_len as u64;
    for entry in entries.iter_mut() {
        entry.offset = offset;
        let path_len = u16::try_from(entry.path.len())
            .map_err(|_| Error::limit_exceeded("bundle path does not fit u16"))?;
        header.extend_from_slice(&path_len.to_le_bytes());
        header.push(entry.role.as_u8());
        header.push(0);
        header.extend_from_slice(&entry.offset.to_le_bytes());
        header.extend_from_slice(&entry.length.to_le_bytes());
        header.extend_from_slice(&entry.sha256);
        header.extend_from_slice(entry.path.as_bytes());
        offset = offset
            .checked_add(entry.length)
            .ok_or_else(|| Error::limit_exceeded("bundle section offset overflow"))?;
    }
    if header.len() != header_len {
        return Err(Error::internal(
            "bundle header length differs from the encoded length",
        ));
    }
    let crc = header_crc(&header);
    header[HEADER_CRC_OFFSET..HEADER_CRC_END].copy_from_slice(&crc.to_le_bytes());
    Ok(header)
}

/// Computes the header CRC-32 over every header byte except the CRC field.
fn header_crc(header: &[u8]) -> u32 {
    let mut crc = Crc32::new();
    crc.update(&header[..HEADER_CRC_OFFSET]);
    crc.update(&header[HEADER_CRC_END..]);
    crc.finalize()
}

/// Computes the catalog digest over the header and ordered section digests.
fn catalog_digest(header: &[u8], entries: &[HeaderEntry<'_>]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(header);
    for entry in entries {
        hasher.update(&entry.sha256);
    }
    hasher.finalize()
}

/// Validates a relative, normalized, UTF-8 bundle path.
fn validate_path(path: &str) -> Result<()> {
    if path.is_empty() {
        return Err(Error::corrupt("bundle path is empty"));
    }
    if path.len() > MAX_PATH_BYTES {
        return Err(Error::limit_exceeded(format!(
            "bundle path is {} bytes, above the {MAX_PATH_BYTES}-byte cap",
            path.len()
        )));
    }
    if path.starts_with('/') || path.contains('\\') || path.contains('\0') {
        return Err(Error::corrupt(format!(
            "bundle path {path} is absolute or contains a forbidden character"
        )));
    }
    for component in path.split('/') {
        if component.is_empty() || component == "." || component == ".." {
            return Err(Error::corrupt(format!(
                "bundle path {path} contains an empty, current, or parent component"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_sections() -> [SectionInput<'static>; 3] {
        [
            SectionInput {
                path: "model.safetensors",
                role: SectionRole::Weights,
                data: b"weights",
            },
            SectionInput {
                path: "config.json",
                role: SectionRole::Config,
                data: b"{}",
            },
            SectionInput {
                path: "tokenizer.json",
                role: SectionRole::Tokenizer,
                data: b"tokenizer",
            },
        ]
    }

    #[test]
    fn round_trip_preserves_sections_and_sorts_paths() {
        let sections = sample_sections();
        let bytes = Bundle::build(&sections).expect("build bundle");
        let bundle = Bundle::parse(&bytes).expect("parse bundle");
        assert_eq!(bundle.sections().len(), 3);
        let paths: Vec<&str> = bundle.sections().iter().map(|s| s.path()).collect();
        assert_eq!(
            paths,
            vec!["config.json", "model.safetensors", "tokenizer.json"]
        );
        assert_eq!(bundle.data("config.json"), Some(&b"{}"[..]));
        assert_eq!(bundle.data("model.safetensors"), Some(&b"weights"[..]));
        assert_eq!(bundle.data("tokenizer.json"), Some(&b"tokenizer"[..]));
        assert!(bundle.data("missing").is_none());
        bundle.verify().expect("section digests");
    }

    #[test]
    fn empty_bundle_round_trips() {
        let bytes = Bundle::build(&[]).expect("build empty bundle");
        let bundle = Bundle::parse(&bytes).expect("parse empty bundle");
        assert!(bundle.sections().is_empty());
        assert_eq!(bundle.header_len(), FIXED_HEADER_BYTES);
        bundle.verify().expect("empty bundle has no sections");
    }

    #[test]
    fn duplicate_paths_are_rejected() {
        let sections = [
            SectionInput {
                path: "a",
                role: SectionRole::Other,
                data: b"1",
            },
            SectionInput {
                path: "a",
                role: SectionRole::Other,
                data: b"2",
            },
        ];
        let error = Bundle::build(&sections).expect_err("duplicate path must fail");
        assert!(matches!(error, Error::Usage(_)));
    }

    #[test]
    fn invalid_paths_are_rejected() {
        for path in ["", "/absolute", "a//b", "a/./b", "a/../b", "a\\b", "a\0b"] {
            let sections = [SectionInput {
                path,
                role: SectionRole::Other,
                data: b"x",
            }];
            let error = Bundle::build(&sections).expect_err("invalid path must fail");
            assert!(
                matches!(error, Error::Corrupt(_) | Error::LimitExceeded(_)),
                "path {path:?} produced {error:?}"
            );
        }
    }

    #[test]
    fn corrupt_magic_version_and_lengths_are_rejected() {
        let bytes = Bundle::build(&sample_sections()).expect("build bundle");

        let mut bad_magic = bytes.clone();
        bad_magic[0] ^= 0xFF;
        assert!(matches!(
            Bundle::parse(&bad_magic).expect_err("bad magic"),
            Error::Corrupt(_)
        ));

        let mut bad_version = bytes.clone();
        bad_version[8] = 2;
        assert!(matches!(
            Bundle::parse(&bad_version).expect_err("bad version"),
            Error::Unsupported(_)
        ));

        let mut bad_total = bytes.clone();
        bad_total[18..26].copy_from_slice(&(bytes.len() as u64 + 1).to_le_bytes());
        assert!(matches!(
            Bundle::parse(&bad_total).expect_err("bad total length"),
            Error::Corrupt(_)
        ));

        let mut truncated = bytes.clone();
        truncated.truncate(bytes.len() - 1);
        assert!(matches!(
            Bundle::parse(&truncated).expect_err("truncated bundle"),
            Error::Corrupt(_)
        ));
    }

    #[test]
    fn header_crc_and_catalog_digest_are_checked() {
        let bytes = Bundle::build(&sample_sections()).expect("build bundle");

        let mut bad_crc = bytes.clone();
        bad_crc[HEADER_CRC_OFFSET] ^= 0xFF;
        assert!(matches!(
            Bundle::parse(&bad_crc).expect_err("bad header CRC"),
            Error::Corrupt(_)
        ));

        let mut bad_digest = bytes.clone();
        let last = bad_digest.len() - 1;
        bad_digest[last] ^= 0xFF;
        assert!(matches!(
            Bundle::parse(&bad_digest).expect_err("bad catalog digest"),
            Error::Corrupt(_)
        ));
    }

    #[test]
    fn section_data_corruption_is_detected_by_verify() {
        let bytes = Bundle::build(&sample_sections()).expect("build bundle");
        let mut corrupted = bytes;
        let config_offset = {
            let bundle = Bundle::parse(&corrupted).expect("parse bundle");
            let section = bundle.section("config.json").expect("config section");
            section.data().as_ptr() as usize - corrupted.as_ptr() as usize
        };
        corrupted[config_offset] ^= 0xFF;
        let bundle = Bundle::parse(&corrupted).expect("structure still parses");
        assert!(matches!(
            bundle.verify().expect_err("section digest must fail"),
            Error::Corrupt(_)
        ));
    }

    #[test]
    fn streaming_writer_round_trips_and_rejects_mismatches() {
        let mut config_reader = std::io::Cursor::new(b"{}".to_vec());
        let mut weights_reader = std::io::Cursor::new(b"weights".to_vec());
        let mut sources = [
            BundleSource {
                path: "config.json",
                role: SectionRole::Config,
                size: 2,
                sha256: sha256(b"{}"),
                reader: &mut config_reader,
            },
            BundleSource {
                path: "model.safetensors",
                role: SectionRole::Weights,
                size: 7,
                sha256: sha256(b"weights"),
                reader: &mut weights_reader,
            },
        ];
        let mut output = Vec::new();
        write_bundle(&mut sources, &mut output).expect("write bundle");
        let bundle = Bundle::parse(&output).expect("parse streamed bundle");
        assert_eq!(bundle.data("model.safetensors"), Some(&b"weights"[..]));
        bundle.verify().expect("streamed section digests");

        let mut reader = std::io::Cursor::new(b"weights".to_vec());
        let mut sources = [BundleSource {
            path: "model.safetensors",
            role: SectionRole::Weights,
            size: 7,
            sha256: [0u8; 32],
            reader: &mut reader,
        }];
        let mut output = Vec::new();
        assert!(matches!(
            write_bundle(&mut sources, &mut output).expect_err("digest mismatch"),
            Error::Corrupt(_)
        ));

        let mut reader = std::io::Cursor::new(b"w".to_vec());
        let mut sources = [BundleSource {
            path: "model.safetensors",
            role: SectionRole::Weights,
            size: 7,
            sha256: sha256(b"weights"),
            reader: &mut reader,
        }];
        let mut output = Vec::new();
        assert!(matches!(
            write_bundle(&mut sources, &mut output).expect_err("short source"),
            Error::Corrupt(_)
        ));
    }

    #[test]
    fn unsorted_streaming_sources_are_rejected() {
        let mut first = std::io::Cursor::new(b"a".to_vec());
        let mut second = std::io::Cursor::new(b"b".to_vec());
        let mut sources = [
            BundleSource {
                path: "b",
                role: SectionRole::Other,
                size: 1,
                sha256: sha256(b"b"),
                reader: &mut second,
            },
            BundleSource {
                path: "a",
                role: SectionRole::Other,
                size: 1,
                sha256: sha256(b"a"),
                reader: &mut first,
            },
        ];
        let mut output = Vec::new();
        assert!(matches!(
            write_bundle(&mut sources, &mut output).expect_err("unsorted sources"),
            Error::Usage(_)
        ));
    }
}
