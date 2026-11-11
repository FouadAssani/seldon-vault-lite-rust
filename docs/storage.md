# Persistent storage and hybrid search

This document is the operational reference for milestone M3. ADR 0005 covers
the journal and hybrid search; ADR 0006 covers snapshots, compaction, and
read-only views. This file describes the on-disk layout, recovery algorithm,
limits, index behavior, and CLI usage.

## Status

Implemented on branch feat/m3-persistent-storage:

- append-only journal with framed, CRC-32-checked transactions;
- atomic commit and fsync before in-memory apply;
- deterministic replay, torn-tail truncation, and corruption detection;
- exclusive writer lock using std::fs::File::try_lock;
- versioned snapshots with a bounded header and transaction records;
- safe compaction with crash recovery at every snapshot and journal write step;
- lock-free read-only views that see committed writer state and ignore torn
  journal tails;
- BM25 lexical index, exact cosine vector search, and RRF hybrid fusion;
- CLI index, search, verify, and compact commands;
- crash-injection and corruption tests at every byte of a journal record, a
  snapshot temporary file, a snapshot file, and a compaction journal temporary
  file.

Not claimed yet:

- HNSW or any approximate vector index;
- HTTP API, jobs, and ingestion queue;
- image and audio storage.

## Data directory

A data directory contains:

    <data-dir>/
      journal.log      append-only transaction journal
      snapshot.sv      materialized state after compaction
      snapshot.tmp     temporary snapshot during atomic creation
      journal.tmp      temporary empty journal during compaction
      writer.lock      exclusive writer lock file
      compaction.lock  shared/exclusive lock serializing compaction and readers

The lock files are intentionally left in place. Lock ownership is held by the
open file description, not by file contents. The snapshot and journal are the
source of truth; the in-memory document and chunk maps are rebuilt by replay.
Temporary files are ignored by open.

The data directory is user data and is never embedded in the binary. A store
must not share a directory with unrelated files.

## Journal record layout

All integers are little-endian.

    offset  size  field
    0       4     magic "SVLR"
    4       2     format version (1)
    6       2     record type (1 = transaction)
    8       8     sequence (starts at 1, increments by one)
    16      4     payload length
    20      4     header CRC-32 over bytes 0..20
    24      4     payload CRC-32
    28      N     transaction payload
    28+N    4     footer magic "SVRF"
    32+N    8     footer sequence
    40+N    4     footer payload length
    44+N    4     footer payload CRC-32

The footer is written last. A record is committed only when the header,
payload, footer, and sequence all validate.

## Snapshot layout

A snapshot is a fixed header followed by transaction records in the same
framing as the journal. All integers are little-endian.

    offset  size  field
    0       4     magic "SVS1"
    4       2     format version (1)
    6       2     reserved (0)
    8       8     base sequence (last journal sequence covered)
    16      8     next chunk id
    24      4     embedding dimension (0 means no chunks)
    28      8     document count
    36      8     chunk count
    44      8     record count (one record per document)
    52      4     header CRC-32 over bytes 0..52
    56      N     transaction records numbered 1..record_count

Snapshot records materialize the current state: one PutDocument operation
followed by that document's PutChunk operations in ordinal order. Chunk ids and
next_chunk_id are preserved exactly; restore does not renumber them. The base
sequence is the global journal sequence before compaction and may be greater
than record_count because deleted documents and other transactions are covered
by the snapshot but not materialized.

A snapshot must be complete. Truncation or trailing bytes are corruption
because the file is renamed into place only after it is fully synchronized.

## Transaction payload

    u16  schema version (1)
    u32  operation count (1..=8192)
    operation repeated

    operation tag 1: put document
      string id
      u8 modality (0 text, 1 image, 2 audio)
      u8 title present; string title when present
      u8 source present; string source when present
      u32 metadata count
      string key, string value repeated

    operation tag 2: put chunk
      u64 chunk id
      string document id
      u32 ordinal
      string text
      u32 dimension
      f32 value repeated dimension times

    operation tag 3: delete document
      string document id

Strings are length-prefixed UTF-8 with explicit caps. Live ingestion
normalizes embeddings with f64 accumulation and stores f32 values. Replay does
not renormalize; it verifies finite values and a norm within 1e-3 of 1.0 so a
hostile journal cannot inject an arbitrary dot-product vector.

## Commit and recovery

Appending a transaction:

1. validate the document, chunks, dimension, and limits;
2. assign chunk ids and ordinals;
3. encode the transaction and check the 16 MiB payload cap;
4. build the framed record with header and footer checksums;
5. seek to the committed end, write the complete record, and call sync_data;
6. on failure, truncate back to the previous committed end; if that rollback
   fails, poison the writer and refuse further appends;
7. apply the transaction in memory and advance the sequence.

Opening a store:

1. create the directory if needed and sync it;
2. acquire the exclusive writer lock and ensure compaction.lock exists;
3. read snapshot.sv with a 512 MiB cap, validate the header and records, and
   restore the materialized documents and chunks;
4. read journal.log with a 256 MiB cap;
5. scan the journal from base + 1 when it was compacted, or from sequence one
   when an old journal remains after a crash; skip records at or below the
   snapshot base and apply the rest;
6. truncate only the uncommitted journal tail;
7. open the append writer at the new valid end.

A missing snapshot is treated as an empty base. A snapshot with an invalid
header, truncation, trailing bytes, count mismatch, invalid record, duplicate
chunk id, missing document, duplicate ordinal, or dimension mismatch is
corruption and stops the open.

Recovery classifications:

| Situation | Result |
|---|---|
| fewer than 28 header bytes remain | uncommitted tail; truncate |
| header CRC invalid and no later committed record | uncommitted tail; truncate |
| header CRC invalid but a committed footer follows | corrupt; refuse |
| payload or footer incomplete | uncommitted tail; truncate |
| footer valid, payload CRC mismatch | corrupt; refuse |
| corruption before a later committed record | corrupt; refuse |
| sequence gap or duplicate | corrupt; refuse |
| unknown format version or record type | unsupported; refuse |
| invalid transaction semantics | corrupt; refuse |

A checksum failure in the final record is treated as an uncommitted tail when
there is no later committed record and no committed footer. This is the only
ambiguous case: a torn footer and bit rot in the final footer are
indistinguishable without a second storage medium. The recovery report exposes
the number of discarded trailing bytes; verify reports it without truncating.

## Compaction

Compaction runs under the writer lock and takes compaction.lock exclusively. It
writes the complete snapshot to snapshot.tmp, synchronizes it, renames it to
snapshot.sv, synchronizes the directory, creates and synchronizes an empty
journal.tmp, renames it over journal.log, synchronizes the directory, reopens
the empty journal, and sets next_sequence to base + 1.

Crash windows:

| Crash point | Recovered state |
|---|---|
| before snapshot rename | old snapshot and journal; partial snapshot.tmp ignored |
| after snapshot rename, before journal rename | new snapshot and old journal; records at or below the base are skipped |
| after journal rename, before reopen | new snapshot and empty journal |
| journal rename failure | new snapshot and old journal; compaction can be retried |
| journal reopen failure | writer is poisoned until reopened |

The snapshot is built in memory and capped at 512 MiB. Disk usage temporarily
doubles because the snapshot duplicates chunks and vectors.

## Read-only views

ReadOnlyStore opens an existing data directory without taking writer.lock. It
takes compaction.lock shared for the duration of the load, replays the snapshot
and the committed journal prefix into an immutable in-memory state, and does not
modify snapshot.sv or journal.log. A concurrent append can leave a torn journal
tail; the view reports the discarded bytes and does not apply them.

Store::verify uses the same read-only loading path and reports snapshot records,
journal records applied after the snapshot, documents, chunks, and truncated
bytes without modifying the data directory.

## Store API

The public store surface is:

- Store::open(data_dir): lock, replay snapshot and journal, recover, and open
  for append;
- Store::put_document(document, chunks): one atomic transaction;
- Store::delete_document(id): cascade chunks, return false when absent;
- Store::compact(): materialize a snapshot and replace the journal;
- Store::open_read_only(data_dir): lock-free ReadOnlyStore view;
- Store::documents and Store::chunks: deterministic iterators;
- Store::document_chunks(id): chunk ids in ordinal order;
- Store::recovery: snapshot records, journal records, and truncated bytes;
- Store::embedding_dimension: the single dimension shared by all chunks;
- Store::verify(data_dir): read-only integrity report.

ReadOnlyStore exposes the same read accessors without mutation methods. It is a
point-in-time view: later writer appends are not visible until it is reopened.

A store rejects duplicate document ids, non-text modalities, empty chunk
lists, dimension changes, oversized payloads, and exhausted count caps before
writing. A failed validation leaves the journal and in-memory state unchanged.

## Limits

| Limit | Value |
|---|---|
| Document id | 256 bytes |
| Title or source | 4,096 bytes |
| Metadata entries | 64 |
| Metadata key | 128 bytes |
| Metadata value | 4,096 bytes |
| Chunk text | 1 MiB |
| Chunks per document | 4,096 |
| Embedding dimension | 8,192 |
| Transaction operations | 8,192 |
| Transaction payload | 16 MiB |
| Journal file read | 256 MiB |
| Snapshot file read or written | 512 MiB |
| Snapshot header | 56 bytes |
| Documents in a store | 1,000,000 |
| Chunks in a store | 10,000,000 |

## Lexical index

The index is rebuilt from the committed store on each open. Tokenization:

- split on non-alphanumeric Unicode characters;
- lowercase each scalar with char::to_lowercase;
- cap text at 1 MiB, tokens per chunk at 8,192, and token length at 64 bytes.

BM25 uses k1 = 1.2, b = 0.75, and the standard idf formula. Postings are stored
in a BTreeMap keyed by term, with one entry per chunk and its term frequency.
Query terms are deduplicated. Filters are evaluated before scoring. Ranking is
by descending score then ascending chunk id.

## Exact vector search

Stored vectors are normalized at ingestion. Query vectors are validated as
finite, non-empty, and non-zero. The score is cosine similarity computed with
f64 accumulation; a query dimension that differs from the store dimension is a
usage error. Ranking is by descending score then ascending chunk id.

This exact path is the correctness oracle. HNSW is deferred until volume and
latency measurements justify it.

## Hybrid RRF

Hybrid search computes the complete filtered lexical and vector rankings and
fuses them with:

    score = sum(1 / (60 + rank))

Ranks are one-based. The fused ranking is sorted by descending score then
ascending chunk id. Pagination slices that deterministic ranking.

## CLI examples

Index with the frozen model and official tokenizer:

    cargo run --release -- index --data-dir ./data \
      --model-dir .model-cache/f9972ac22e696a105c482202646979e33e4b5a1e \
      documents.jsonl

Index with diagnostic ids against the tiny official model:

    cargo run -- index --data-dir ./data --model-dir fixtures/model/tiny_text \
      --dim 8 --ids documents.jsonl

A JSONL line:

    {"id":"doc-1","text":"rust language systems","title":"Rust","metadata":{"lang":"en"},"ids":[1,5,7,9,2]}

The ids field is only accepted with --ids; text is still stored for BM25.

Search:

    cargo run --release -- search --data-dir ./data --mode hybrid \
      --limit 10 --filter lang=en "What is ownership in Rust?"

Lexical search does not load the model:

    cargo run -- search --data-dir ./data --mode lexical "ownership"

Verify without modifying the snapshot or journal:

    cargo run -- verify --data-dir ./data

Compact the store into a snapshot and an empty journal:

    cargo run -- compact --data-dir ./data

Output is compact deterministic JSON on stdout. Errors go to stderr with exit
code 1 for execution or corruption failures and 2 for usage errors.

## Verification

The storage tests include:

- CRC-32 standard vectors and bit flips;
- codec round trips and cap enforcement;
- journal round trip and torn tails at every byte of one and two records;
- payload corruption, header corruption with an intact footer, mid-file
  corruption, sequence gaps, unknown versions, and unknown record types;
- store reopen, delete cascade, writer lock contention, duplicate documents,
  dimension mismatch, payload caps, and crash injection at every write byte;
- snapshot header validation, atomic replacement, and truncation rejection;
- compaction preserving document ids, chunk ids, next_chunk_id, embedding
  dimension, and global sequence numbers;
- truncated snapshot.tmp and journal.tmp at every byte being ignored;
- old journal records below the snapshot base being skipped, followed by new
  records above the base;
- read-only views while a writer is open, including torn-tail reporting without
  modifying the journal;
- verify reporting snapshot and journal records separately;
- index ranking, filters, pagination, determinism, and token caps.

Run the full gate with scripts/verify.sh or the commands in README.md.
