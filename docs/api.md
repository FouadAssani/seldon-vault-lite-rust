# HTTP API — milestone M6, current at M8

**Status:** implemented on branch `feat/m6-http-api` and unchanged at M8. The
service is a native `std::net::TcpListener` HTTP/1.1 server with a strict
parser, bounded connections, a bounded ingestion queue, and no third-party
crate. It is a local service, not an Internet-facing gateway. The M8 release
harness exercises the loopback health, ready, models, embeddings, ingestion,
search, deletion, and shutdown lifecycle and verifies that documents survive a
restart.

## Bind, authentication, and transport

The default bind address is `127.0.0.1:8080`. Binding to a non-loopback address
requires an access token of at least 16 bytes:

    seldon-vault-lite serve --data-dir ./data --bind 0.0.0.0:8080       --token "$SELDON_API_TOKEN"

When a token is configured, every request must carry one of:

    Authorization: Bearer <token>
    X-API-Key: <token>

The comparison is constant-time. The token is never written to logs or
responses. Without a token, only loopback clients are accepted because the
default bind is loopback.

There is **no TLS implementation**. Do not expose the service directly to the
Internet. For a trusted LAN, use a token and a firewall or an external TLS
terminating proxy; the product itself does not provide transport encryption.

The server handles one request per connection and sends
`Connection: close`. HTTP/1.1 keep-alive, pipelining, chunked transfer encoding,
and `Expect: 100-continue` are deliberately not supported.

## Limits

| Limit | Default | Hard cap |
|---|---|---|
| Concurrent connections | 32 | 1,024 |
| Request body | 32 MiB | 64 MiB |
| Text or chunk text | 1 MiB | 1 MiB |
| Decoded media | 16 MiB | 16 MiB |
| Token ids per request | 8,192 | 8,192 |
| Search limit | 10 | 1,000 |
| Search offset | 0 | 1,000,000 |
| Ingestion queue (queued + running) | 64 | 4,096 |
| Job workers | 1 | 64 |
| Read/write timeout | 10 s | 300 s |

The JSON parser also caps nesting depth, string size, container entries, and
total values. Every allocation derived from a request is checked before it is
made.

## Errors

All errors are JSON:

    {"error":{"code":"invalid_request","message":"..."}}

| Status | Typical code | Meaning |
|---|---|---|
| 400 | `invalid_request`, `invalid_input` | malformed JSON, unknown field, bad value |
| 401 | `unauthorized` | missing or invalid bearer token |
| 403 | `forbidden` | shutdown from a non-loopback connection |
| 404 | `not_found`, `job_not_found` | unknown route, document, or job |
| 405 | `method_not_allowed` | wrong method; the `Allow` header lists valid methods |
| 408 | `request_timeout` | the request was not completed before the timeout |
| 413 | `payload_too_large` | body, text, media, or token list above its cap |
| 415 | `unsupported_media_type` | POST without `application/json` |
| 417 | `expectation_failed` | `Expect` header is unsupported |
| 429 | `queue_full` | ingestion queue is at capacity |
| 500 | `internal_error` | an internal invariant or I/O failure |
| 501 | `unsupported` | model has no such modality, or no tokenizer |
| 503 | `connection_limit`, `unavailable` | server saturated or not ready |

POST routes require `Content-Type: application/json` and an optional
`charset=utf-8` parameter. Other media types are rejected with 415. Unknown
fields are rejected rather than ignored.

## Health and metadata

### GET /health

Process liveness. It does not load a model or touch storage.

    {"status":"ok","version":"0.1.0","target":"aarch64-apple-darwin","uptime_seconds":12}

### GET /ready

Readiness. The text model and storage are loaded at startup. Vision and audio
towers are loaded lazily on their first request, so `loaded` reports the current
state while `models` reports what the frozen configuration declares.

    {
      "ready": true,
      "storage": true,
      "documents": 3,
      "chunks": 3,
      "models": {"text": true, "image": true, "audio": true},
      "loaded": {"text": true, "image": false, "audio": false},
      "text_input": true
    }

A poisoned store or a failed readiness probe returns 503 with the same schema.

### GET /v1/models

    {
      "model": "google/embeddinggemma-2",
      "revision": "f9972ac22e696a105c482202646979e33e4b5a1e",
      "version": "0.1.0",
      "program": "seldon-vault-lite",
      "target": "aarch64-apple-darwin",
      "embedding_dimension": 768,
      "matryoshka_dimensions": [768, 512, 256, 128],
      "modalities": {"text": true, "image": true, "audio": true},
      "text_input": true,
      "token_input": true,
      "limits": {
        "max_body_bytes": 33554432,
        "max_connections": 32,
        "queue_capacity": 64,
        "job_workers": 1
      }
    }

`text_input` is false when the model source has no `tokenizer.json`; the
explicit token-id input remains available. The tiny diagnostic fixtures use
that path. `matryoshka_dimensions` lists the official dimensions that do not
exceed the model embedding dimension.

### GET /v1/stats

Local observability counters:

    {
      "uptime_seconds": 12,
      "requests_total": 8,
      "status_2xx": 7,
      "status_4xx": 1,
      "status_5xx": 0,
      "bytes_in": 512,
      "bytes_out": 4096,
      "active_connections": 0,
      "rejected_connections": 0,
      "spawn_failures": 0,
      "jobs": {"queued":0,"running":0,"succeeded":1,"failed":0,"cancelled":0},
      "documents": 1,
      "chunks": 1,
      "recovery": {"replayed_records":0,"snapshot_records":0,"truncated_bytes":0}
    }

## Embeddings

### POST /v1/embeddings

The `input` object selects one real input path. `dimension` is optional and
defaults to the server's configured dimension. Matryoshka truncation is
followed by renormalization.

Text (requires a tokenizer):

    {
      "input": {"type": "text", "text": "What causes the northern lights?", "prompt": "SearchQuery"},
      "dimension": 768
    }

`prompt` is one of the official names and defaults to `SearchQuery`.

Explicit token ids (diagnostic; bypasses the tokenizer and prompt prefix):

    {"input":{"type":"tokens","ids":[1,5,7,9,2]},"dimension":768}

Image (PNG or JPEG, base64 in the `data` field):

    {"input":{"type":"image","encoding":"base64","data":"..."},"dimension":768}

Audio (RIFF/WAVE, base64 in the `data` field):

    {"input":{"type":"audio","encoding":"base64","data":"..."},"dimension":768}

`encoding` is optional and must be `base64` when present. The server detects
PNG versus JPEG from the decoded bytes. The native decoders, preprocessing, and
towers apply their own caps. No path, file name, caption, or transcription is
used.

Response:

    {
      "model": "google/embeddinggemma-2",
      "revision": "f9972ac22e696a105c482202646979e33e4b5a1e",
      "dimension": 768,
      "modality": "text",
      "embedding": [0.0123, -0.0456, ...]
    }

A model without `vision_config` returns 501 for image input; a model without
`audio_config` returns 501 for audio input; a model without `tokenizer.json`
returns 501 for text input.

## Documents and jobs

### POST /v1/documents

Ingestion is asynchronous. The request is validated and enqueued, then a worker
embeds the input and writes one atomic transaction.

    {
      "id": "doc-1",
      "input": {"type": "text", "text": "Ownership and borrowing in Rust."},
      "title": "Rust ownership",
      "source": "notes/rust.md",
      "metadata": {"lang": "en", "topic": "rust"},
      "dimension": 768
    }

The input forms are the same as for embeddings, except that a document always
uses the `Document` prompt:

- `{"type":"text","text":"..."}`
- `{"type":"tokens","ids":[...],"text":"optional BM25 text"}`
- `{"type":"image","encoding":"base64","data":"..."}`
- `{"type":"audio","encoding":"base64","data":"..."}`

Image and audio chunks have empty text; they participate in vector search and
modality filters but not BM25. A token-id document may carry `text` so it also
participates in lexical and hybrid search.

Accepted response (202):

    {"job":{"id":1,"kind":"ingest","status":"queued","created_ms":...,"started_ms":null,"finished_ms":null,"documents":0,"chunks":0,"message":null}}

A full queue returns 429 `queue_full`. Duplicate ids are not rejected at
enqueue time; the job fails with a message when the store reports that the
document already exists.

### GET /v1/jobs/{id}

    {"job":{"id":1,"kind":"ingest","status":"succeeded","created_ms":...,"started_ms":...,"finished_ms":...,"documents":1,"chunks":1,"message":null}}

Statuses are `queued`, `running`, `succeeded`, `failed`, and `cancelled`. A
failed job carries the error message in `message`. Unknown ids return 404.

### DELETE /v1/jobs/{id}

Requests cooperative cancellation:

- 202 `{"id":1,"status":"cancellation_requested"}` for a queued or running job;
- 200 `{"id":1,"status":"already_finished"}` for a terminal job;
- 404 for an unknown job.

A queued job is marked cancelled before it starts. A running task checks the
flag before and after the expensive embedding phase and before the storage
write. Cancellation cannot interrupt a kernel already inside a model forward
pass; a write that has already committed is reported as succeeded.

### DELETE /v1/documents/{id}

The path segment is percent-decoded as UTF-8 and validated as a document id.

    {"id":"doc-1","deleted":true}

Deletion is idempotent at the API level: deleting a missing document returns
200 with `"deleted":false`.

## Search

### POST /v1/search

    {
      "query": {"type": "text", "text": "ownership in Rust", "prompt": "SearchQuery"},
      "mode": "hybrid",
      "limit": 10,
      "offset": 0,
      "filters": {"modality": "text", "metadata": {"lang": "en"}},
      "dimension": 768
    }

`mode` is `hybrid` (default), `lexical`, or `vector`. Lexical search does not
load the model and requires text. Hybrid search requires text for BM25 and an
embedding; a token query may carry `text` for the lexical side. Vector search
accepts text, token ids, an image, or audio:

    {"query":{"type":"tokens","ids":[1,5,7,9,2]},"mode":"vector"}
    {"query":{"type":"image","encoding":"base64","data":"..."},"mode":"vector"}
    {"query":{"type":"audio","encoding":"base64","data":"..."},"mode":"vector"}

Image and audio queries require vector mode. Filters are applied before
ranking. `modality` is `text`, `image`, or `audio`; `metadata` entries are
exact string equality matches.

Response:

    {
      "mode": "hybrid",
      "limit": 10,
      "offset": 0,
      "hits": [
        {
          "document_id": "doc-1",
          "chunk_id": 1,
          "ordinal": 0,
          "score": 0.0325,
          "lexical_rank": 1,
          "vector_rank": 2,
          "text": "Ownership and borrowing in Rust.",
          "title": "Rust ownership",
          "source": "notes/rust.md",
          "modality": "text",
          "metadata": {"lang": "en", "topic": "rust"}
        }
      ]
    }

Ranking is deterministic. Exact cosine search is the vector correctness oracle;
hybrid fusion uses Reciprocal Rank Fusion with k = 60. Pagination slices the
full ranking after filters.

## Shutdown

### POST /v1/shutdown

Only a loopback peer may request shutdown. When a token is configured, the
request must still authenticate. The request must declare
`Content-Type: application/json` (the body may be empty); this prevents a
cross-origin simple POST from shutting down a local server. The response is 202:

    {"status":"shutting_down"}

The accept loop stops, waits briefly for active connections, cancels queued
jobs, and joins the job workers. There is no POSIX signal handler: the product
does not use FFI or `unsafe`, so the supported graceful-shutdown mechanism is
this authenticated loopback route.

## Examples

    curl -s http://127.0.0.1:8080/health
    curl -s http://127.0.0.1:8080/v1/models

    curl -s -X POST http://127.0.0.1:8080/v1/embeddings \
      -H 'Content-Type: application/json' \
      --data-binary '{"input":{"type":"text","text":"hello","prompt":"SearchQuery"},"dimension":768}'

    curl -s -X POST http://127.0.0.1:8080/v1/documents \
      -H 'Content-Type: application/json' \
      --data-binary '{"id":"doc-1","input":{"type":"text","text":"Ownership and borrowing in Rust."},"metadata":{"lang":"en"}}'

    curl -s http://127.0.0.1:8080/v1/jobs/1

    curl -s -X POST http://127.0.0.1:8080/v1/search \
      -H 'Content-Type: application/json' \
      --data-binary '{"query":{"type":"text","text":"ownership"},"mode":"hybrid"}'

    curl -s -X DELETE http://127.0.0.1:8080/v1/documents/doc-1

## Security notes

- The service is loopback-only by default. A non-loopback bind refuses to start
  without a token of at least 16 bytes.
- There is no TLS, no proxy trust, no CORS policy, and no cookie session. It is
  not safe to expose directly to an untrusted network.
- Media is accepted only as bounded base64 in the JSON body. The API does not
  read arbitrary local paths supplied by a client.
- Request parsing rejects duplicate `Host`, `Content-Length`,
  `Transfer-Encoding`, `Authorization`, and `Expect` headers, obsolete line
  folding, bare LF line endings, absolute request targets, fragments, control
  characters, and unknown methods.
- The body cap is checked before allocation. The JSON parser and the base64
  decoder apply their own caps before media decoding.
- Connection count and thread creation are bounded. A full connection limit
  returns 503 after draining up to 64 KiB of the request. Slow clients are
  disconnected after the configured timeout.
- The ingestion queue is bounded; a full queue returns 429 rather than growing
  memory without limit.
- No fake embedding is returned. Missing tokenizer, missing modality weights,
  invalid media, and model failures are typed errors.
