# ADR 0005 — Persistent storage and hybrid search

- **Status**: accepted
- **Date**: 2026-11-11
- **Milestone**: M3
- **Decision maker**: project maintainer
- **Supersedes**: none
- **Superseded by**: none

## Context

Milestone M3 turns the text engine into a useful local product. M2 can compute
compliant text embeddings, but nothing survives a restart and there is no way
to retrieve a document by meaning or by term. M3 must add persistent storage,
crash recovery, a lexical index, exact vector search, and hybrid ranking while
respecting the project contract: no third-party crate, no external database,
no external inference engine, no network at runtime, and no unsafe code in
storage or business logic.

The storage format is written by the product and read back by the product, but
it is still treated as hostile. A journal can be truncated by a crash, altered
by a filesystem, or replaced by an attacker with write access to the data
directory. Replay must therefore validate every length, checksum, sequence,
and referential invariant before changing in-memory state.

## Decisions

### D1 — Data model and schema version

The persistent model is intentionally small:

- a document id is a non-empty UTF-8 string of at most 256 bytes, without
  control characters;
- a document carries an optional title, an optional source, one modality, and
  at most 64 metadata key/value pairs;
- a chunk belongs to exactly one document, has a zero-based ordinal, bounded
  text, and a normalized FP32 embedding;
- a chunk id is a positive u64 assigned by the store in increasing order;
- a transaction is a non-empty list of operations applied atomically.

The transaction schema version is 1 and is written in every payload. An
unknown version is rejected as unsupported, never guessed. Documents, chunks,
and metadata are encoded with explicit caps and little-endian integers.

Milestone M3 stores only text documents. The modality enum reserves image and
audio tags for M4 and M5, but ingestion rejects them with Error::Unsupported;
no fake embedding is produced.

### D2 — Journal framing

The journal is a sequence of records in one file, journal.log. Each record is:

    header (28 bytes)
      magic          4  "SVLR"
      format version 2  (1)
      record type    2  (1 = transaction)
      sequence       8  (starts at 1, increments by one)
      payload length 4
      header CRC     4  CRC-32 of header bytes 0..20
      payload CRC    4  CRC-32 of the payload
    payload          N
    footer (20 bytes)
      magic          4  "SVRF"
      sequence       8
      payload length 4
      payload CRC    4

The footer duplicates the sequence, length, and payload checksum and is
written last. The checksum is CRC-32/ISO-HDLC, implemented in
src/storage/checksum.rs; it detects torn writes and accidental corruption but
is not a cryptographic authentication mechanism.

The footer is what makes recovery unambiguous:

- a partial header, payload, or footer is an uncommitted tail and is removed;
- a complete record with a valid footer but a payload checksum mismatch is
  corruption and stops recovery;
- a damaged header followed by a committed footer is corruption, not a torn
  tail, because the footer proves the record was fully written;
- corruption before a later valid record is corruption, even when the damaged
  record is not the last one.

### D3 — Atomic commit, fsync, and recovery

Writes append one transaction record and call File::sync_data before the
in-memory state changes. The journal file is created with a directory fsync so
the file name survives a crash. On a write or sync failure, the writer rolls
the file back to its previous length; if the rollback also fails, the writer is
poisoned and refuses further appends rather than risking a corrupt middle
record.

Opening a store:

1. creates the data directory if needed and synchronizes it;
2. acquires an exclusive writer lock;
3. reads the journal with the documented 256 MiB cap;
4. scans and validates the committed prefix;
5. decodes and applies every transaction to a temporary state;
6. truncates only the uncommitted tail;
7. opens the append writer at the new valid end.

A semantic error in a committed transaction, such as a duplicate document id,
a missing referenced document, a chunk-id gap, or a non-normalized embedding,
is reported as Error::Corrupt. The journal is not truncated in that case.

Crash injection covers every byte prefix of a record: a prefix shorter than
the complete record recovers the previous committed prefix; the complete
record recovers the new transaction. Tests also cover two transactions,
payload corruption, header corruption with an intact footer, mid-file
corruption, sequence gaps, unknown versions, unknown record types, and
oversized payloads.

### D4 — Writer locking and concurrent reads

The store holds an exclusive lock on writer.lock for its lifetime. A second
writer receives Error::Io with ErrorKind::WouldBlock. The lock uses
std::fs::File::try_lock, stabilized in Rust 1.89, so the crate minimum is
raised to 1.89. No libc FFI or lock-file PID heuristic is used.

Milestone M3 is writer-serialized. The verify command reads the journal
without acquiring the lock and without modifying it, so an operator can
inspect a directory while a writer is running. Concurrent API readers and
snapshot-based read-only opening are deferred to M6.

### D5 — Explicit limits

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
| Index tokens per chunk | 8,192 |
| Query tokens | 1,024 |

Every allocation derived from external input is checked against these caps
before it is used. Embeddings must be finite, non-empty, and have a positive
norm; live ingestion normalizes them with f64 accumulation and stores f32
values. Replay verifies that stored vectors are normalized within 1e-3 and
does not renormalize them, so replay reproduces the recorded bytes exactly.

### D6 — BM25 lexical index

The lexical index is rebuilt deterministically from the store on open. Text is
split into Unicode alphanumeric runs, lowercased with char::to_lowercase, and
capped per chunk. Postings are held in a BTreeMap keyed by term; each posting
stores the chunk index and term frequency. Document length is the token count.

The score is standard BM25:

    idf = ln(1 + (N - df + 0.5) / (df + 0.5))
    score += idf * tf * (k1 + 1) /
             (tf + k1 * (1 - b + b * dl / avgdl))

with k1 = 1.2 and b = 0.75. Query terms are deduplicated. Filters are applied
before a posting contributes. Results are sorted by descending score then
ascending chunk id, so ties are independent of map iteration order.

No stemming, stop-word list, or positional index is used at M3. Positions are
not needed for the current ranking contract and would add storage cost without
a validated benefit.

### D7 — Exact vector search

Stored embeddings are unit-normalized, so exact cosine search is a dot product
with a query norm correction. The query is validated as finite, non-empty, and
non-zero; its dimension must match the store dimension. The score is computed
with f64 accumulation and returned as a f64. Results are sorted by descending
cosine then ascending chunk id.

This exact path is the correctness oracle. HNSW is not introduced at M3:
the plan requires volume and latency measurements before an approximate index
is accepted, and no such measurements exist yet.

### D8 — Hybrid Reciprocal Rank Fusion

Hybrid search fuses the complete lexical and vector rankings with RRF:

    score(chunk) = sum over lists containing the chunk of 1 / (60 + rank)

Ranks are one-based. The constant 60 is the standard RRF value. Filters are
applied before both rankings, so filtered documents do not consume ranks.
Fused results are sorted by descending RRF score then ascending chunk id.
Pagination slices the fully deterministic fused ranking.

### D9 — CLI

M3 adds:

- index: ingests a strict JSONL file into the store. Each line is an object
  with id, text, optional title, optional source, and optional metadata.
  Duplicate ids, unknown fields, malformed JSON, oversized lines, and more
  than 10,000 documents are rejected before any write. Embeddings use the
  official Document prompt and the frozen tokenizer.
- search: runs hybrid, lexical, or vector search with repeatable metadata
  filters, limit, offset, and a JSON output containing ids, ranks, scores,
  title, source, modality, text, and metadata. Lexical mode does not load the
  model.
- verify: decodes and validates the committed journal prefix without acquiring
  the writer lock or modifying the journal.

A diagnostic --ids option bypasses the tokenizer for reproducible tests
against the tiny official model. It does not fake an embedding: it calls the
same TextModel::embed_ids path as the existing embed command.

### D10 — Deferred work

The following M3-plan items are deliberately not claimed yet:

- snapshots and safe compaction: the journal is currently replayed in full;
- concurrent read-only store opening and API readers;
- HNSW: exact search remains the only vector index until measurements justify
  an approximate one;
- HTTP API, jobs, and ingestion queue: milestone M6;
- image and audio: milestones M4 and M5.

Snapshots and compaction are the next storage increment. Their crash-injection
tests must cover every snapshot and compaction write step before the format is
declared stable.

### D11 — Verification

The M3 quality gate is the existing gate plus the new tests:

    cargo fmt --check
    cargo clippy --all-targets -- -D warnings
    cargo test
    cargo build --release --target aarch64-apple-darwin
    cargo tree --edges normal,build,dev
    otool -L target/aarch64-apple-darwin/release/seldon-vault-lite
    vtool -show-build target/aarch64-apple-darwin/release/seldon-vault-lite

M3 tests cover:

- CRC-32 standard vectors and bit-flip detection;
- bounded codec round trips, truncation, caps, and invalid UTF-8;
- document, chunk, metadata, transaction, and embedding validation;
- journal append, replay, torn tails at every byte, corruption, sequence
  gaps, unsupported versions and types, and recovery truncation;
- store put, delete, reopen, writer lock, duplicate rejection, dimension
  consistency, payload caps, crash injection at every byte of one and two
  transactions, and read-only verification;
- BM25 ranking, ties, filters, and unknown terms;
- exact cosine ranking, filters, dimension mismatch, and invalid queries;
- RRF scores and tie-breaking;
- hybrid ranking, pagination, determinism, token caps, and empty stores;
- CLI parsing, malformed JSONL rejection, and an end-to-end tiny-model
  index/search/verify flow using the diagnostic ids path.

## Consequences

### Positive

- User data survives restarts with a validated, checksummed journal.
- A crash truncates only an uncommitted tail; committed corruption is reported
  instead of silently discarded.
- BM25 and exact vector search share deterministic ordering and filters, and
  RRF combines them without score calibration.
- The CLI provides a usable local text product without an external database,
  server, or crate.
- The exact vector path remains the oracle for any future approximate index.

### Costs and risks

- The journal grows without compaction until the snapshot increment lands;
  opening a store replays the whole committed prefix.
- Writer locking is exclusive at M3; concurrent readers are deferred.
- CRC-32 detects accidental corruption, not deliberate tampering. A local
  attacker with write access to the data directory is outside the threat model
  until the API adds authentication and access control.
- The index is rebuilt in memory on every search; this is acceptable for the
  M3 corpus sizes but will need measurement before large collections.
- The Rust 1.89 minimum is a deliberate increase for std file locking.

### Follow-up

- Snapshots, compaction, and crash injection at every snapshot write step.
- Read-only store opening and concurrent readers before the HTTP API.
- Volume and latency measurements before any HNSW decision.
- ADR 0006 for the HTTP contract, LAN authentication, and limits at M6.
