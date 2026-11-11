# Persistent storage and hybrid search

This document is the operational reference for milestone M3. ADR 0005 is the
decision record; this file describes the on-disk layout, recovery algorithm,
limits, index behavior, and CLI usage.

## Status

Implemented on branch feat/m3-persistent-storage:

- append-only journal with framed, CRC-32-checked transactions;
- atomic commit and fsync before in-memory apply;
- deterministic replay, torn-tail truncation, and corruption detection;
- exclusive writer lock using std::fs::File::try_lock;
- BM25 lexical index, exact cosine vector search, and RRF hybrid fusion;
- CLI index, search, and verify commands;
- crash-injection and corruption tests at every byte of a journal record.

Not claimed yet:

- snapshots and compaction;
- concurrent read-only store opening;
- HNSW or any approximate vector index;
- HTTP API, jobs, and ingestion queue;
- image and audio storage.

## Data directory

A data directory contains:

    <data-dir>/
      journal.log      append-only transaction journal
      writer.lock      exclusive writer lock file

The lock file is intentionally left in place. Lock ownership is held by the
open file description, not by file contents. The journal is the source of
truth; the in-memory document and chunk maps are rebuilt by replay.

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
2. acquire the exclusive writer lock;
3. read journal.log with a 256 MiB cap;
4. scan records in sequence order;
5. decode and apply each committed transaction to a temporary state;
6. truncate only the uncommitted tail;
7. open the append writer at the new valid end.

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

## Store API

The public store surface is:

- Store::open(data_dir): lock, replay, recover, and open for append;
- Store::put_document(document, chunks): one atomic transaction;
- Store::delete_document(id): cascade chunks, return false when absent;
- Store::documents and Store::chunks: deterministic iterators;
- Store::document_chunks(id): chunk ids in ordinal order;
- Store::recovery: replayed record count and truncated bytes;
- Store::embedding_dimension: the single dimension shared by all chunks;
- Store::verify(data_dir): read-only integrity report.

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

Verify without modifying the journal:

    cargo run -- verify --data-dir ./data

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
- read-only verify with a torn tail and unchanged journal bytes;
- index ranking, filters, pagination, determinism, and token caps.

Run the full gate with scripts/verify.sh or the commands in README.md.
