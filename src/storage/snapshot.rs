//! Versioned snapshot file for compacted store state.
//!
//! A snapshot is a fixed header followed by ordinary journal transaction
//! records. It materializes the current documents and chunks, so opening a
//! compacted store does not replay the full historical journal.
//!
//! Header layout (little-endian):
//!
//! ```text
//! magic            4  "SVS1"
//! format version   2  (1)
//! reserved         2  (0)
//! base sequence    8  last journal sequence covered by the snapshot
//! next chunk id    8  next id to assign after restore
//! embedding dim    4  0 means no chunks, otherwise 1..=8192
//! document count   8
//! chunk count      8
//! record count     8  one transaction record per document
//! header CRC       4  CRC-32 of bytes 0..52
//! ```
//! The snapshot file is written to snapshot.tmp, synchronized, and renamed to
//! snapshot.sv. A crash before the rename leaves the previous snapshot and the
//! journal untouched; a crash after the rename leaves a complete snapshot and
//! either the old or the new journal, both of which recover consistently.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;

use crate::binary::reader::read_file_bounded;
use crate::error::{Error, Result};
use crate::storage::checksum::crc32;
use crate::storage::journal::{self, Record};
use crate::storage::model::MAX_EMBEDDING_DIMENSION;

/// Snapshot file name.
pub const SNAPSHOT_FILE_NAME: &str = "snapshot.sv";
/// Temporary snapshot file name used during atomic creation.
pub const SNAPSHOT_TMP_FILE_NAME: &str = "snapshot.tmp";
/// Snapshot magic number.
pub const SNAPSHOT_MAGIC: [u8; 4] = *b"SVS1";
/// Snapshot format version.
pub const SNAPSHOT_VERSION: u16 = 1;
/// Snapshot header size in bytes.
pub const SNAPSHOT_HEADER_LEN: usize = 56;
/// Maximum snapshot file size read or written.
pub const MAX_SNAPSHOT_BYTES: u64 = 512 * 1024 * 1024;

/// Parsed snapshot header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapshotHeader {
    /// Last journal sequence covered by the snapshot.
    pub base_sequence: u64,
    /// Next chunk id to assign after restore.
    pub next_chunk_id: u64,
    /// Embedding dimension, or None when the store has no chunks.
    pub embedding_dimension: Option<usize>,
    /// Number of documents materialized.
    pub document_count: u64,
    /// Number of chunks materialized.
    pub chunk_count: u64,
    /// Number of transaction records following the header.
    pub record_count: u64,
}

/// Parsed snapshot borrowing its record payloads from the file bytes.
#[derive(Debug)]
pub struct Snapshot<'a> {
    /// Parsed header.
    pub header: SnapshotHeader,
    /// Transaction records, one per document.
    pub records: Vec<Record<'a>>,
}

/// Reads a snapshot file with the documented size cap.
///
/// A missing snapshot is not an error; it means the store has no snapshot yet.
pub fn read_file(path: &Path) -> Result<Option<Vec<u8>>> {
    if !path.try_exists()? {
        return Ok(None);
    }
    read_file_bounded(path, MAX_SNAPSHOT_BYTES).map(Some)
}

/// Parses and validates a complete snapshot file.
///
/// A snapshot must be complete: truncation or trailing bytes are corruption,
/// because the file is renamed into place only after it is fully synchronized.
pub fn parse(bytes: &[u8]) -> Result<Snapshot<'_>> {
    let header = parse_header(bytes)?;
    if bytes.len() < SNAPSHOT_HEADER_LEN {
        return Err(Error::corrupt("snapshot is shorter than its header"));
    }
    let records_bytes = &bytes[SNAPSHOT_HEADER_LEN..];
    let scan = journal::scan_with_start(records_bytes, 1)?;
    if scan.truncated_bytes != 0 || scan.valid_end != records_bytes.len() as u64 {
        return Err(Error::corrupt(format!(
            "snapshot records end at byte {}, file has {} record bytes",
            scan.valid_end,
            records_bytes.len()
        )));
    }
    if scan.records.len() as u64 != header.record_count {
        return Err(Error::corrupt(format!(
            "snapshot declares {} records, found {}",
            header.record_count,
            scan.records.len()
        )));
    }
    if header.record_count != header.document_count {
        return Err(Error::corrupt(format!(
            "snapshot declares {} documents but {} records",
            header.document_count, header.record_count
        )));
    }
    if header.chunk_count < header.document_count {
        return Err(Error::corrupt("snapshot has fewer chunks than documents"));
    }
    if header.base_sequence < header.record_count {
        return Err(Error::corrupt(format!(
            "snapshot base sequence {} is below its {} records",
            header.base_sequence, header.record_count
        )));
    }
    Ok(Snapshot {
        header,
        records: scan.records,
    })
}

/// Encodes and validates a snapshot header.
pub fn encode_header(header: &SnapshotHeader) -> Result<[u8; SNAPSHOT_HEADER_LEN]> {
    if header.record_count != header.document_count {
        return Err(Error::internal(
            "snapshot record count must equal document count",
        ));
    }
    if header.chunk_count < header.document_count {
        return Err(Error::internal(
            "snapshot chunk count must be at least the document count",
        ));
    }
    if header.base_sequence < header.record_count {
        return Err(Error::internal(
            "snapshot base sequence must cover every record",
        ));
    }
    if header.next_chunk_id == 0 {
        return Err(Error::internal("snapshot next chunk id must be positive"));
    }
    let dimension = match header.embedding_dimension {
        Some(dimension) if dimension == 0 || dimension > MAX_EMBEDDING_DIMENSION => {
            return Err(Error::internal(format!(
                "snapshot embedding dimension {dimension} is outside 1..={MAX_EMBEDDING_DIMENSION}"
            )));
        }
        Some(dimension) => u32::try_from(dimension)
            .map_err(|_| Error::internal("snapshot dimension does not fit u32"))?,
        None => 0,
    };

    let mut bytes = [0u8; SNAPSHOT_HEADER_LEN];
    bytes[0..4].copy_from_slice(&SNAPSHOT_MAGIC);
    bytes[4..6].copy_from_slice(&SNAPSHOT_VERSION.to_le_bytes());
    bytes[6..8].copy_from_slice(&0u16.to_le_bytes());
    bytes[8..16].copy_from_slice(&header.base_sequence.to_le_bytes());
    bytes[16..24].copy_from_slice(&header.next_chunk_id.to_le_bytes());
    bytes[24..28].copy_from_slice(&dimension.to_le_bytes());
    bytes[28..36].copy_from_slice(&header.document_count.to_le_bytes());
    bytes[36..44].copy_from_slice(&header.chunk_count.to_le_bytes());
    bytes[44..52].copy_from_slice(&header.record_count.to_le_bytes());
    let header_crc = crc32(&bytes[..52]);
    bytes[52..56].copy_from_slice(&header_crc.to_le_bytes());
    Ok(bytes)
}

/// Writes a snapshot atomically to the data directory.
///
/// The temporary file is fully written and synchronized before the rename, and
/// the directory is synchronized after it. On failure the temporary file is
/// removed best-effort; the previous snapshot, if any, is untouched.
pub fn write_atomic(dir: &Path, bytes: &[u8]) -> Result<()> {
    if bytes.len() as u64 > MAX_SNAPSHOT_BYTES {
        return Err(Error::limit_exceeded(format!(
            "snapshot is {} bytes, above the {MAX_SNAPSHOT_BYTES}-byte cap",
            bytes.len()
        )));
    }
    let tmp_path = dir.join(SNAPSHOT_TMP_FILE_NAME);
    let final_path = dir.join(SNAPSHOT_FILE_NAME);
    let result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp_path)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&tmp_path, &final_path)?;
        sync_directory(dir)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp_path);
    }
    result
}

fn parse_header(bytes: &[u8]) -> Result<SnapshotHeader> {
    if bytes.len() < SNAPSHOT_HEADER_LEN {
        return Err(Error::corrupt("snapshot is shorter than its header"));
    }
    if bytes[0..4] != SNAPSHOT_MAGIC {
        return Err(Error::corrupt("snapshot magic number is invalid"));
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if version != SNAPSHOT_VERSION {
        return Err(Error::unsupported(format!(
            "snapshot format version {version} is not supported (expected {SNAPSHOT_VERSION})"
        )));
    }
    let reserved = u16::from_le_bytes([bytes[6], bytes[7]]);
    if reserved != 0 {
        return Err(Error::corrupt("snapshot reserved header field is non-zero"));
    }
    let stored_crc = u32::from_le_bytes([bytes[52], bytes[53], bytes[54], bytes[55]]);
    if crc32(&bytes[..52]) != stored_crc {
        return Err(Error::corrupt("snapshot header checksum mismatch"));
    }

    let base_sequence = u64::from_le_bytes([
        bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
    ]);
    let next_chunk_id = u64::from_le_bytes([
        bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23],
    ]);
    let dimension = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]) as usize;
    let document_count = u64::from_le_bytes([
        bytes[28], bytes[29], bytes[30], bytes[31], bytes[32], bytes[33], bytes[34], bytes[35],
    ]);
    let chunk_count = u64::from_le_bytes([
        bytes[36], bytes[37], bytes[38], bytes[39], bytes[40], bytes[41], bytes[42], bytes[43],
    ]);
    let record_count = u64::from_le_bytes([
        bytes[44], bytes[45], bytes[46], bytes[47], bytes[48], bytes[49], bytes[50], bytes[51],
    ]);

    if next_chunk_id == 0 {
        return Err(Error::corrupt("snapshot next chunk id is zero"));
    }
    if dimension > MAX_EMBEDDING_DIMENSION {
        return Err(Error::limit_exceeded(format!(
            "snapshot embedding dimension {dimension} is above the {MAX_EMBEDDING_DIMENSION}-dimension cap"
        )));
    }
    Ok(SnapshotHeader {
        base_sequence,
        next_chunk_id,
        embedding_dimension: if dimension == 0 {
            None
        } else {
            Some(dimension)
        },
        document_count,
        chunk_count,
        record_count,
    })
}

fn sync_directory(path: &Path) -> Result<()> {
    let directory = fs::File::open(path)?;
    directory.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir(name: &str) -> PathBuf {
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "seldon-snapshot-{}-{}-{unique}",
            std::process::id(),
            name
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create temp dir");
        path
    }

    fn sample_header() -> SnapshotHeader {
        SnapshotHeader {
            base_sequence: 5,
            next_chunk_id: 9,
            embedding_dimension: Some(8),
            document_count: 2,
            chunk_count: 3,
            record_count: 2,
        }
    }

    #[test]
    fn header_round_trips() {
        let header = sample_header();
        let bytes = encode_header(&header).expect("encode");
        let parsed = parse_header(&bytes).expect("parse");
        assert_eq!(parsed, header);
    }

    #[test]
    fn empty_snapshot_parses() {
        let header = SnapshotHeader {
            base_sequence: 4,
            next_chunk_id: 1,
            embedding_dimension: None,
            document_count: 0,
            chunk_count: 0,
            record_count: 0,
        };
        let bytes = encode_header(&header).expect("encode");
        let parsed = parse(&bytes).expect("parse");
        assert_eq!(parsed.header, header);
        assert!(parsed.records.is_empty());
    }

    #[test]
    fn header_corruption_is_detected() {
        let mut bytes = encode_header(&sample_header()).expect("encode");
        bytes[8] ^= 0xFF;
        let error = parse_header(&bytes).expect_err("header corruption");
        assert!(matches!(error, Error::Corrupt(_)), "{error}");
    }

    #[test]
    fn truncated_snapshot_is_rejected() {
        let bytes = encode_header(&sample_header()).expect("encode");
        let error = parse(&bytes[..SNAPSHOT_HEADER_LEN - 1]).expect_err("truncated header");
        assert!(matches!(error, Error::Corrupt(_)), "{error}");
    }

    #[test]
    fn atomic_write_replaces_the_snapshot() {
        let dir = temp_dir("write");
        let bytes = encode_header(&sample_header()).expect("encode");
        write_atomic(&dir, &bytes).expect("write");
        let path = dir.join(SNAPSHOT_FILE_NAME);
        assert_eq!(fs::read(&path).expect("read"), bytes);
        assert!(!dir.join(SNAPSHOT_TMP_FILE_NAME).exists());
        write_atomic(&dir, &bytes).expect("replace");
        assert_eq!(fs::read(&path).expect("read"), bytes);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn invalid_headers_are_rejected_before_writing() {
        let mut header = sample_header();
        header.record_count = 3;
        let error = encode_header(&header).expect_err("record/document mismatch");
        assert!(matches!(error, Error::Internal(_)), "{error}");

        let mut header = sample_header();
        header.embedding_dimension = Some(MAX_EMBEDDING_DIMENSION + 1);
        let error = encode_header(&header).expect_err("dimension cap");
        assert!(matches!(error, Error::Internal(_)), "{error}");
    }
}
