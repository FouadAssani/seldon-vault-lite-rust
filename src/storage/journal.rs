//! Append-only journal with framed, checksummed records and crash recovery.
//!
//! Record layout (all integers little-endian):
//!
//! ```text
//! header (28 bytes)
//!   magic          4  "SVLR"
//!   format version 2
//!   record type    2  (1 = transaction)
//!   sequence       8  (starts at 1, strictly increasing by one)
//!   payload length 4
//!   header CRC     4  (CRC-32 of bytes 0..20)
//!   payload CRC    4
//! payload          N
//! footer (20 bytes)
//!   magic          4  "SVRF"
//!   sequence       8
//!   payload length 4
//!   payload CRC    4
//! ```
//!
//! The footer is written last. A record is committed only when its header,
//! payload, and footer all validate. A crash can therefore leave a prefix of
//! the final record; replay truncates that uncommitted tail. Corruption inside
//! a committed record is reported as Error::Corrupt instead of being silently
//! discarded. The footer also lets replay distinguish a corrupted header from
//! a torn header: if a committed footer follows the damaged header, the file
//! is corrupt, not merely truncated.

use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::binary::crc32::crc32;
use crate::binary::reader::read_file_bounded;
use crate::error::{Error, Result};

/// Journal file name inside a data directory.
pub const JOURNAL_FILE_NAME: &str = "journal.log";

/// Magic number of a journal record header.
pub const RECORD_MAGIC: [u8; 4] = *b"SVLR";
/// Magic number of a journal record footer.
pub const FOOTER_MAGIC: [u8; 4] = *b"SVRF";
/// Journal framing version.
pub const FORMAT_VERSION: u16 = 1;
/// Record type for an atomic transaction payload.
pub const RECORD_TRANSACTION: u16 = 1;

/// Header size in bytes.
pub const HEADER_LEN: usize = 28;
/// Footer size in bytes.
pub const FOOTER_LEN: usize = 20;
/// Maximum transaction payload accepted by one record.
pub const MAX_PAYLOAD_BYTES: usize = crate::storage::model::MAX_TRANSACTION_BYTES;
/// Maximum journal file size read during replay.
pub const MAX_JOURNAL_BYTES: u64 = 256 * 1024 * 1024;

/// A committed record borrowed from the scanned journal bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Record<'a> {
    /// Monotonic transaction sequence.
    pub sequence: u64,
    /// Transaction payload bytes.
    pub payload: &'a [u8],
}

/// Result of scanning a journal file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scan<'a> {
    /// Committed records in sequence order.
    pub records: Vec<Record<'a>>,
    /// Offset of the first uncommitted byte, or the file length.
    pub valid_end: u64,
    /// Number of trailing bytes discarded as an uncommitted tail.
    pub truncated_bytes: u64,
}

/// Reads a journal file with the documented size cap.
///
/// A missing file is treated as an empty journal; the caller creates it on
/// first append.
pub fn read_file(path: &Path) -> Result<Vec<u8>> {
    if !path.try_exists()? {
        return Ok(Vec::new());
    }
    read_file_bounded(path, MAX_JOURNAL_BYTES)
}

/// Returns the sequence of the first complete-looking record header.
///
/// Returns None for an empty file or an invalid first header. The full scanner
/// remains responsible for deciding whether an invalid header is a torn tail
/// or corruption.
pub fn first_sequence(bytes: &[u8]) -> Result<Option<u64>> {
    if bytes.len() < HEADER_LEN {
        return Ok(None);
    }
    match parse_header(bytes, 0) {
        HeaderParse::Valid(header) => Ok(Some(header.sequence)),
        HeaderParse::Invalid => Ok(None),
    }
}

/// Scans journal bytes starting at sequence one.
pub fn scan(bytes: &[u8]) -> Result<Scan<'_>> {
    scan_with_start(bytes, 1)
}

/// Scans journal bytes and returns committed records plus recovery metadata.
///
/// The scanner validates sequence monotonicity from start_sequence, framing,
/// footer agreement, and payload checksums. It never allocates from an
/// untrusted length beyond the documented payload cap. A compacted journal may
/// start at a sequence above one; a journal written before compaction starts at
/// one and is filtered by the caller against a snapshot base.
pub fn scan_with_start(bytes: &[u8], start_sequence: u64) -> Result<Scan<'_>> {
    let mut records = Vec::new();
    let mut offset = 0usize;
    let mut expected_sequence = start_sequence;

    while offset < bytes.len() {
        let remaining = bytes.len() - offset;
        if remaining < HEADER_LEN {
            return Ok(Scan {
                records,
                valid_end: offset as u64,
                truncated_bytes: remaining as u64,
            });
        }

        let header = match parse_header(bytes, offset) {
            HeaderParse::Valid(header) => header,
            HeaderParse::Invalid => {
                if has_valid_record_after(bytes, offset + 1) || has_committed_footer(bytes, offset)
                {
                    return Err(Error::corrupt(format!(
                        "journal record at byte {offset} is corrupt, not a torn tail"
                    )));
                }
                return Ok(Scan {
                    records,
                    valid_end: offset as u64,
                    truncated_bytes: remaining as u64,
                });
            }
        };

        if header.format_version != FORMAT_VERSION {
            return Err(Error::unsupported(format!(
                "journal format version {} is not supported (expected {FORMAT_VERSION})",
                header.format_version
            )));
        }
        if header.record_type != RECORD_TRANSACTION {
            return Err(Error::unsupported(format!(
                "journal record type {} is not supported",
                header.record_type
            )));
        }
        if header.payload_len > MAX_PAYLOAD_BYTES {
            return Err(Error::corrupt(format!(
                "journal record at byte {offset} declares a {}-byte payload, above the {MAX_PAYLOAD_BYTES}-byte cap",
                header.payload_len
            )));
        }

        let footer_offset = match offset
            .checked_add(HEADER_LEN)
            .and_then(|value| value.checked_add(header.payload_len))
        {
            Some(value) => value,
            None => {
                return Err(Error::corrupt(format!(
                    "journal record at byte {offset} has an overflowing length"
                )));
            }
        };
        let record_end = match footer_offset.checked_add(FOOTER_LEN) {
            Some(value) => value,
            None => {
                return Err(Error::corrupt(format!(
                    "journal record at byte {offset} has an overflowing footer offset"
                )));
            }
        };
        if record_end > bytes.len() {
            return Ok(Scan {
                records,
                valid_end: offset as u64,
                truncated_bytes: remaining as u64,
            });
        }

        let footer = match parse_footer(bytes, footer_offset) {
            Some(footer) => footer,
            None => {
                if has_valid_record_after(bytes, offset + 1) {
                    return Err(Error::corrupt(format!(
                        "journal record at byte {offset} has an invalid footer and later committed records"
                    )));
                }
                return Ok(Scan {
                    records,
                    valid_end: offset as u64,
                    truncated_bytes: remaining as u64,
                });
            }
        };
        if footer.sequence != header.sequence
            || footer.payload_len != header.payload_len
            || footer.payload_crc != header.payload_crc
        {
            if has_valid_record_after(bytes, offset + 1) {
                return Err(Error::corrupt(format!(
                    "journal record at byte {offset} has a footer that disagrees with its header and later committed records"
                )));
            }
            return Ok(Scan {
                records,
                valid_end: offset as u64,
                truncated_bytes: remaining as u64,
            });
        }

        if header.sequence != expected_sequence {
            return Err(Error::corrupt(format!(
                "journal sequence {} at byte {offset} does not match expected {expected_sequence}",
                header.sequence
            )));
        }

        let payload = &bytes[offset + HEADER_LEN..footer_offset];
        if crc32(payload) != header.payload_crc {
            return Err(Error::corrupt(format!(
                "journal payload checksum mismatch at byte {offset}"
            )));
        }

        records.push(Record {
            sequence: header.sequence,
            payload,
        });
        offset = record_end;
        expected_sequence = expected_sequence
            .checked_add(1)
            .ok_or_else(|| Error::internal("journal sequence overflow"))?;
    }

    Ok(Scan {
        records,
        valid_end: offset as u64,
        truncated_bytes: 0,
    })
}

/// Append-only writer for a journal file.
#[derive(Debug)]
pub struct Journal {
    file: File,
    path: PathBuf,
    offset: u64,
    poisoned: bool,
}

impl Journal {
    /// Opens or creates a journal and positions the writer at valid_end.
    ///
    /// When truncate is true, the uncommitted tail after valid_end is removed
    /// and synchronized before any append. A newly created journal triggers a
    /// directory fsync so the file name survives a crash.
    pub fn open(path: &Path, valid_end: u64, truncate: bool) -> Result<Self> {
        let existed = path.try_exists()?;
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        if !existed {
            sync_parent_directory(path)?;
        }

        let file_len = file.metadata()?.len();
        if file_len < valid_end {
            return Err(Error::internal(format!(
                "journal {} is shorter than its scanned valid end",
                path.display()
            )));
        }
        if truncate {
            file.set_len(valid_end)?;
            file.sync_data()?;
        } else if file_len > valid_end {
            return Err(Error::internal(format!(
                "journal {} has {} bytes after valid end {valid_end} without truncation",
                path.display(),
                file_len
            )));
        }

        file.seek(SeekFrom::Start(valid_end))?;
        Ok(Self {
            file,
            path: path.to_path_buf(),
            offset: valid_end,
            poisoned: false,
        })
    }

    /// Appends one committed record and synchronizes it to stable storage.
    ///
    /// On a write or sync failure the writer rolls the file back to its
    /// previous length. If the rollback itself fails, the journal is poisoned
    /// and refuses further appends instead of risking a corrupt middle record.
    pub fn append(&mut self, sequence: u64, payload: &[u8]) -> Result<()> {
        if self.poisoned {
            return Err(Error::internal(
                "journal is poisoned after a failed rollback; reopen the store",
            ));
        }
        if payload.len() > MAX_PAYLOAD_BYTES {
            return Err(Error::limit_exceeded(format!(
                "journal payload is {} bytes, above the {MAX_PAYLOAD_BYTES}-byte cap",
                payload.len()
            )));
        }

        let record = encode_record(sequence, payload);
        let previous = self.offset;
        let result = (|| -> Result<()> {
            self.file.seek(SeekFrom::Start(previous))?;
            self.file.write_all(&record)?;
            self.file.sync_data()?;
            Ok(())
        })();

        match result {
            Ok(()) => {
                self.offset = previous
                    .checked_add(record.len() as u64)
                    .ok_or_else(|| Error::internal("journal offset overflow"))?;
                Ok(())
            }
            Err(error) => {
                let rollback = self
                    .file
                    .set_len(previous)
                    .and_then(|_| self.file.sync_data());
                if rollback.is_err() {
                    self.poisoned = true;
                }
                Err(error)
            }
        }
    }

    /// Path of the journal file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Current append offset, equal to the committed file length.
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Indicates whether a failed rollback poisoned the writer.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
}

struct Header {
    format_version: u16,
    record_type: u16,
    sequence: u64,
    payload_len: usize,
    payload_crc: u32,
}

struct Footer {
    sequence: u64,
    payload_len: usize,
    payload_crc: u32,
}

enum HeaderParse {
    Valid(Header),
    Invalid,
}

fn parse_header(bytes: &[u8], offset: usize) -> HeaderParse {
    let end = match offset.checked_add(HEADER_LEN) {
        Some(value) if value <= bytes.len() => value,
        _ => return HeaderParse::Invalid,
    };
    let header = &bytes[offset..end];
    if header[0..4] != RECORD_MAGIC {
        return HeaderParse::Invalid;
    }
    let stored_crc = read_u32_le(header, 20);
    if crc32(&header[..20]) != stored_crc {
        return HeaderParse::Invalid;
    }
    HeaderParse::Valid(Header {
        format_version: read_u16_le(header, 4),
        record_type: read_u16_le(header, 6),
        sequence: read_u64_le(header, 8),
        payload_len: read_u32_le(header, 16) as usize,
        payload_crc: read_u32_le(header, 24),
    })
}

fn parse_footer(bytes: &[u8], offset: usize) -> Option<Footer> {
    let end = offset.checked_add(FOOTER_LEN)?;
    if end > bytes.len() {
        return None;
    }
    let footer = &bytes[offset..end];
    if footer[0..4] != FOOTER_MAGIC {
        return None;
    }
    Some(Footer {
        sequence: read_u64_le(footer, 4),
        payload_len: read_u32_le(footer, 12) as usize,
        payload_crc: read_u32_le(footer, 16),
    })
}

fn has_valid_record_after(bytes: &[u8], start: usize) -> bool {
    if start >= bytes.len() {
        return false;
    }
    let mut position = start;
    while position + HEADER_LEN <= bytes.len() {
        if bytes[position..position + 4] == RECORD_MAGIC
            && let HeaderParse::Valid(header) = parse_header(bytes, position)
            && header.format_version == FORMAT_VERSION
            && header.record_type == RECORD_TRANSACTION
            && header.payload_len <= MAX_PAYLOAD_BYTES
            && let footer_offset = position + HEADER_LEN + header.payload_len
            && let Some(footer) = parse_footer(bytes, footer_offset)
            && footer.sequence == header.sequence
            && footer.payload_len == header.payload_len
            && footer.payload_crc == header.payload_crc
            && crc32(&bytes[position + HEADER_LEN..footer_offset]) == header.payload_crc
        {
            return true;
        }
        position += 1;
    }
    false
}

fn has_committed_footer(bytes: &[u8], record_offset: usize) -> bool {
    let payload_start = match record_offset.checked_add(HEADER_LEN) {
        Some(value) => value,
        None => return false,
    };
    let search_end = payload_start
        .saturating_add(MAX_PAYLOAD_BYTES)
        .min(bytes.len().saturating_sub(FOOTER_LEN));
    let mut footer_offset = payload_start;
    while footer_offset <= search_end {
        if bytes[footer_offset..footer_offset + 4] == FOOTER_MAGIC
            && let Some(footer) = parse_footer(bytes, footer_offset)
            && footer_offset - payload_start == footer.payload_len
            && crc32(&bytes[payload_start..footer_offset]) == footer.payload_crc
        {
            return true;
        }
        footer_offset += 1;
    }
    false
}

pub(crate) fn encode_record(sequence: u64, payload: &[u8]) -> Vec<u8> {
    let mut record = Vec::with_capacity(HEADER_LEN + payload.len() + FOOTER_LEN);
    let mut header = [0u8; HEADER_LEN];
    header[0..4].copy_from_slice(&RECORD_MAGIC);
    header[4..6].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    header[6..8].copy_from_slice(&RECORD_TRANSACTION.to_le_bytes());
    header[8..16].copy_from_slice(&sequence.to_le_bytes());
    let payload_len = u32::try_from(payload.len()).unwrap_or(u32::MAX);
    header[16..20].copy_from_slice(&payload_len.to_le_bytes());
    let header_crc = crc32(&header[..20]);
    header[20..24].copy_from_slice(&header_crc.to_le_bytes());
    let payload_crc = crc32(payload);
    header[24..28].copy_from_slice(&payload_crc.to_le_bytes());
    record.extend_from_slice(&header);

    record.extend_from_slice(payload);

    let mut footer = [0u8; FOOTER_LEN];
    footer[0..4].copy_from_slice(&FOOTER_MAGIC);
    footer[4..12].copy_from_slice(&sequence.to_le_bytes());
    footer[12..16].copy_from_slice(&payload_len.to_le_bytes());
    footer[16..20].copy_from_slice(&payload_crc.to_le_bytes());
    record.extend_from_slice(&footer);
    record
}

fn read_u16_le(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn read_u64_le(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
        bytes[offset + 4],
        bytes[offset + 5],
        bytes[offset + 6],
        bytes[offset + 7],
    ])
}

fn sync_parent_directory(path: &Path) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let parent = if parent.as_os_str().is_empty() {
        Path::new(".")
    } else {
        parent
    };
    let directory = File::open(parent)?;
    directory.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir(name: &str) -> PathBuf {
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "seldon-journal-{}-{}-{unique}",
            std::process::id(),
            name
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create temp dir");
        path
    }

    fn write_bytes(path: &Path, bytes: &[u8]) {
        fs::write(path, bytes).expect("write journal fixture");
    }

    fn journal_path(dir: &Path) -> PathBuf {
        dir.join(JOURNAL_FILE_NAME)
    }

    #[test]
    fn append_and_scan_round_trip() {
        let dir = temp_dir("round-trip");
        let path = journal_path(&dir);
        let mut journal = Journal::open(&path, 0, false).expect("open journal");
        journal.append(1, b"first").expect("append first");
        journal.append(2, b"second").expect("append second");
        drop(journal);

        let bytes = read_file(&path).expect("read journal");
        let scan = scan(&bytes).expect("scan journal");
        assert_eq!(scan.records.len(), 2);
        assert_eq!(scan.records[0].sequence, 1);
        assert_eq!(scan.records[0].payload, b"first");
        assert_eq!(scan.records[1].sequence, 2);
        assert_eq!(scan.records[1].payload, b"second");
        assert_eq!(scan.valid_end, bytes.len() as u64);
        assert_eq!(scan.truncated_bytes, 0);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_file_scans_as_an_empty_journal() {
        let scan = scan(b"").expect("empty journal");
        assert!(scan.records.is_empty());
        assert_eq!(scan.valid_end, 0);
        assert_eq!(scan.truncated_bytes, 0);
    }

    #[test]
    fn a_prefix_of_one_record_is_recovered_as_an_empty_journal() {
        let payload = b"payload";
        let record = encode_record(1, payload);
        for cut in 0..record.len() {
            let scan = scan(&record[..cut]).expect("torn tail must recover");
            assert!(scan.records.is_empty(), "cut {cut}");
            assert_eq!(scan.valid_end, 0, "cut {cut}");
            assert_eq!(scan.truncated_bytes, cut as u64, "cut {cut}");
        }
        let scan = scan(&record).expect("complete record");
        assert_eq!(scan.records.len(), 1);
        assert_eq!(scan.records[0].payload, payload);
    }

    #[test]
    fn a_prefix_of_a_second_record_keeps_the_first_record() {
        let first = encode_record(1, b"first");
        let second = encode_record(2, b"second");
        let mut bytes = first.clone();
        bytes.extend_from_slice(&second);

        for cut in first.len()..bytes.len() {
            let scan = scan(&bytes[..cut]).expect("torn second record must recover");
            assert_eq!(scan.records.len(), 1, "cut {cut}");
            assert_eq!(scan.records[0].payload, b"first", "cut {cut}");
            assert_eq!(scan.valid_end, first.len() as u64, "cut {cut}");
            assert_eq!(
                scan.truncated_bytes,
                (cut - first.len()) as u64,
                "cut {cut}"
            );
        }
        let scan = scan(&bytes).expect("complete records");
        assert_eq!(scan.records.len(), 2);
    }

    #[test]
    fn payload_corruption_is_reported() {
        let mut record = encode_record(1, b"payload");
        record[HEADER_LEN] ^= 0xFF;
        let error = scan(&record).expect_err("payload corruption must fail");
        assert!(matches!(error, Error::Corrupt(_)), "{error}");
        assert!(error.to_string().contains("checksum"));
    }

    #[test]
    fn header_corruption_with_an_intact_footer_is_reported() {
        let mut record = encode_record(1, b"payload");
        record[8] ^= 0xFF;
        let error = scan(&record).expect_err("header corruption must fail");
        assert!(matches!(error, Error::Corrupt(_)), "{error}");
    }

    #[test]
    fn footer_corruption_at_the_tail_is_treated_as_uncommitted() {
        let mut record = encode_record(1, b"payload");
        let footer_start = record.len() - FOOTER_LEN;
        record[footer_start] ^= 0xFF;
        let scan = scan(&record).expect("a torn footer is not a committed record");
        assert!(scan.records.is_empty());
        assert_eq!(scan.valid_end, 0);
        assert_eq!(scan.truncated_bytes, record.len() as u64);
    }

    #[test]
    fn mid_file_corruption_is_reported_even_with_later_records() {
        let first = encode_record(1, b"first");
        let second = encode_record(2, b"second");
        let mut bytes = first.clone();
        bytes.extend_from_slice(&second);
        bytes[HEADER_LEN] ^= 0xFF;
        let error = scan(&bytes).expect_err("mid-file corruption must fail");
        assert!(matches!(error, Error::Corrupt(_)), "{error}");
    }

    #[test]
    fn sequence_gaps_are_rejected() {
        let first = encode_record(1, b"first");
        let third = encode_record(3, b"third");
        let mut bytes = first;
        bytes.extend_from_slice(&third);
        let error = scan(&bytes).expect_err("sequence gap must fail");
        assert!(matches!(error, Error::Corrupt(_)), "{error}");
        assert!(error.to_string().contains("expected 2"));
    }

    #[test]
    fn unsupported_versions_and_record_types_are_rejected() {
        let mut record = encode_record(1, b"payload");
        record[4..6].copy_from_slice(&2u16.to_le_bytes());
        let header_crc = crc32(&record[..20]);
        record[20..24].copy_from_slice(&header_crc.to_le_bytes());
        let error = scan(&record).expect_err("unknown version");
        assert!(matches!(error, Error::Unsupported(_)), "{error}");

        let mut record = encode_record(1, b"payload");
        record[6..8].copy_from_slice(&9u16.to_le_bytes());
        let header_crc = crc32(&record[..20]);
        record[20..24].copy_from_slice(&header_crc.to_le_bytes());
        let error = scan(&record).expect_err("unknown type");
        assert!(matches!(error, Error::Unsupported(_)), "{error}");
    }

    #[test]
    fn append_rejects_oversized_payloads_before_writing() {
        let dir = temp_dir("oversized");
        let path = journal_path(&dir);
        let mut journal = Journal::open(&path, 0, false).expect("open journal");
        let payload = vec![0u8; MAX_PAYLOAD_BYTES + 1];
        let error = journal.append(1, &payload).expect_err("payload cap");
        assert!(matches!(error, Error::LimitExceeded(_)));
        assert_eq!(journal.offset(), 0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn reopening_truncates_the_uncommitted_tail_and_allows_appends() {
        let dir = temp_dir("recovery");
        let path = journal_path(&dir);
        let record = encode_record(1, b"first");
        let mut torn = record.clone();
        torn.extend_from_slice(&encode_record(2, b"second")[..10]);
        write_bytes(&path, &torn);

        let bytes = read_file(&path).expect("read");
        let scan_result = scan(&bytes).expect("scan");
        assert_eq!(scan_result.records.len(), 1);
        assert_eq!(scan_result.valid_end, record.len() as u64);
        assert_eq!(scan_result.truncated_bytes, 10);

        let mut journal = Journal::open(&path, scan_result.valid_end, true).expect("recover");
        assert_eq!(journal.offset(), record.len() as u64);
        journal.append(2, b"second").expect("append after recovery");
        drop(journal);

        let bytes = read_file(&path).expect("read");
        let scan_result = scan(&bytes).expect("scan");
        assert_eq!(scan_result.records.len(), 2);
        assert_eq!(scan_result.records[1].payload, b"second");
        assert_eq!(scan_result.truncated_bytes, 0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_journal_is_an_empty_file() {
        let dir = temp_dir("missing");
        let bytes = read_file(&journal_path(&dir)).expect("missing file");
        assert!(bytes.is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn creation_synchronizes_the_parent_directory() {
        let dir = temp_dir("creation");
        let path = journal_path(&dir);
        let journal = Journal::open(&path, 0, false).expect("create journal");
        assert!(path.is_file());
        assert_eq!(journal.offset(), 0);
        drop(journal);
        let _ = fs::remove_dir_all(&dir);
    }
}
