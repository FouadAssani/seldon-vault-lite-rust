# ADR 0006 — Snapshots, compaction, and read-only views

- **Status**: accepted
- **Date**: 2026-11-11
- **Milestone**: M3 follow-up
- **Decision maker**: project maintainer
- **Supersedes**: none
- **Superseded by**: none

## Context

ADR 0005 introduced the append-only journal, deterministic replay, crash
recovery, and an exclusive writer lock. It deliberately deferred snapshots,
safe compaction, and concurrent read-only opening. Without them, opening a
store replays the entire historical journal and every search takes the writer
lock, which is acceptable for the first M3 increment but not for a useful local
product.

Compaction must not lose committed transactions or expose a reader to a mixed
snapshot/journal generation. A crash at any point must leave a state that the
next open can recover deterministically.

## Decisions

### D1 — Snapshot format

A snapshot is a fixed header followed by ordinary journal transaction records.
The file is named snapshot.sv. Header layout, little-endian:

    magic            4  "SVS1"
    format version   2  (1)
    reserved         2  (0)
    base sequence    8  last journal sequence covered by the snapshot
    next chunk id    8  next chunk id to assign after restore
    embedding dim    4  0 means no chunks, otherwise 1..=8192
    document count   8
    chunk count      8
    record count     8  one transaction record per document
    header CRC       4  CRC-32 of bytes 0..52

The header is 56 bytes. It records the global journal sequence covered by the
snapshot, the next chunk id, the store embedding dimension, and exact document,
chunk, and record counts. The header CRC detects accidental corruption.

The records use the journal framing from ADR 0005: magic, version, record type,
sequence, payload length, header CRC, payload CRC, payload, and footer. Snapshot
records are numbered from 1 to record_count. They are not historical journal
sequences; they are a materialized encoding of the current state. Each record
contains one PutDocument operation followed by that document's PutChunk
operations in ordinal order. Live ingestion already limits a document plus its
chunks to one 16 MiB transaction, so a snapshot record cannot exceed the
existing payload cap.

The snapshot stores chunk ids exactly as they exist in the store. Restore does
not renumber them. It also preserves next_chunk_id, including the case where
every chunk was deleted and the next id is above the largest remaining id.

### D2 — Snapshot validation

A snapshot must be complete. Truncation or trailing bytes are corruption,
because the file is renamed into place only after it is fully synchronized.

Validation checks:

- magic, format version, reserved field, and header CRC;
- next_chunk_id is positive;
- embedding dimension is zero or within 1..=8192;
- record count equals document count;
- chunk count is at least document count;
- base sequence is at least record count;
- every record is a complete transaction with a valid sequence, footer, and
  payload checksum;
- record sequences run from 1 to record count;
- the last record ends exactly at the end of the file;
- restored document and chunk counts match the header;
- every chunk has the declared embedding dimension;
- every chunk id is below next_chunk_id;
- no delete operation appears in a snapshot;
- every chunk references an existing document and no ordinal is duplicated.

### D3 — Atomic snapshot creation

The snapshot is written to snapshot.tmp with create and truncate, fully
written, and synchronized with sync_all. It is then renamed to snapshot.sv, and
the data directory is synchronized. On failure the temporary file is removed
best-effort; the previous snapshot is untouched. A crash before the rename
leaves a partial temporary file that is ignored by open.

### D4 — Compaction order and crash windows

Compaction runs under the exclusive writer lock and an exclusive compaction
lock. It:

1. builds the complete snapshot bytes from the current state;
2. writes and renames snapshot.tmp to snapshot.sv;
3. creates and synchronizes an empty journal.tmp;
4. renames journal.tmp over journal.log;
5. synchronizes the directory;
6. reopens the empty journal and advances next_sequence to base + 1.

Crash windows:

- before the snapshot rename: the old snapshot and journal remain; a partial
  snapshot.tmp is ignored;
- after the snapshot rename but before the journal rename: the new snapshot and
  the old journal remain. On open, the journal is scanned from sequence one and
  records at or below the snapshot base are skipped; records above the base are
  applied. This is why base_sequence is stored separately from record_count;
- after the journal rename but before reopening: the new snapshot and the empty
  journal remain; open restores the snapshot;
- after a failed journal rename: the new snapshot and old journal remain and
  are recoverable as above; the store can retry compaction;
- after a failed journal reopen: the writer is poisoned and refuses further
  appends until reopened.

The compaction lock is separate from writer.lock so normal writes and reads do
not serialize on it. Compaction takes it exclusively; read-only views take it
shared.

### D5 — Read-only views

ReadOnlyStore opens an existing data directory without acquiring writer.lock.
It replays the snapshot and the committed journal prefix into an immutable
in-memory state. It does not create, truncate, or rewrite the snapshot or
journal.

A concurrent append can leave a torn journal tail. The read-only view reports
the number of discarded trailing bytes in its recovery information and does not
apply them. It takes the compaction lock shared for the duration of the load,
so it cannot observe a half-replaced snapshot/journal generation.

Store::verify uses the same read-only loading path and shared lock. It reports
snapshot records, journal records applied after the snapshot, documents,
chunks, and truncated bytes without modifying the data directory.

### D6 — Sequence handling

The store sequence remains globally monotonic across compactions:

- base_sequence is the last committed journal sequence before compaction;
- snapshot records are numbered 1..record_count for validation only;
- the new journal starts at base_sequence + 1;
- if an old journal remains after a crash, open scans it from sequence one,
  skips records at or below base_sequence, and applies records above it;
- if the journal starts exactly at base_sequence + 1, open scans from that
  sequence;
- any other starting sequence is corruption.

next_chunk_id is preserved in the snapshot header, so deleted chunks are not
reused and external chunk identifiers remain stable across compaction.

### D7 — CLI compact

The CLI adds compact --data-dir <DIR>. It opens the store, compacts it, and
prints a deterministic JSON summary containing snapshot_records, snapshot_bytes,
documents, chunks, and data_dir. verify now also reports snapshot_records.

### D8 — Limits

| Limit | Value |
|---|---|
| Snapshot file read or written | 512 MiB |
| Snapshot header | 56 bytes |
| Snapshot records | one per document |
| Snapshot record payload | existing 16 MiB transaction cap |
| Snapshot embedding dimension | 1..=8192, or none when no chunks exist |

The snapshot file is read with the bounded file reader and rejected if it is
not a regular file or exceeds the cap.

### D9 — Verification

Tests cover:

- header round trip, empty snapshot, truncated header, header corruption, and
  invalid header fields;
- atomic snapshot replacement;
- compaction preserving document ids, chunk ids, next_chunk_id, embedding
  dimension, and global sequence numbers;
- compaction of an empty store and of a store whose chunks were all deleted;
- truncated snapshot.tmp at every byte being ignored while the journal
  recovers;
- truncated journal.tmp at every byte being ignored while the snapshot
  restores;
- truncated snapshot.sv at every byte being rejected;
- corrupted snapshot header rejection;
- an old journal after a snapshot being skipped, followed by new records;
- read-only views while a writer is open, including torn-tail reporting
  without modifying the journal;
- verify reporting snapshot and journal records separately.

## Consequences

### Positive

- Opening a compacted store no longer replays the full historical journal.
- Compaction is crash-safe at every write step: a crash leaves either the old
  generation, the new snapshot with the old journal, or the new snapshot with
  an empty journal, all recoverable.
- Chunk ids and next_chunk_id survive compaction.
- Searches can use ReadOnlyStore without taking the writer lock; only
  compaction serializes against readers.
- verify inspects both generations without modifying them.

### Costs and risks

- Compaction builds the whole snapshot in memory before writing; the 512 MiB
  cap bounds this cost.
- The snapshot duplicates all chunks, including vectors; disk usage temporarily
  doubles during compaction.
- A read-only view is a point-in-time copy; it does not update when the writer
  appends.
- The compaction lock adds a second lock file and a documented lock order:
  writer lock first, then compaction lock. Read-only views take only the
  compaction lock shared.
- CRC-32 still detects accidental corruption, not deliberate tampering.

### Follow-up

- HNSW remains deferred until volume and latency measurements justify it.
- HTTP API, jobs, and authentication remain milestone M6.
- Image and audio remain milestones M4 and M5.
- A future compaction policy may trigger automatically when the journal grows
  beyond a configured threshold; M3 exposes the explicit compact command.
