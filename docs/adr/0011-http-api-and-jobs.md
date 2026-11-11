# ADR 0011 — Native HTTP API, bounded jobs, and M6 hardening

- **Status:** accepted
- **Date:** 2026-11-11
- **Milestone:** M6
- **Supersedes:** none

## Context

Milestones M0–M5 produced a tested engine, persistent store, hybrid index, and
native text/image/audio embeddings. M6 turns the library into a complete local
service: a native HTTP/1.1 API, asynchronous document ingestion, bounded
resource use, authentication for non-loopback binds, observability, and
graceful shutdown. The project forbids third-party crates, external HTTP or
JSON libraries, FFI, and unsafe code in business logic. The only allowed
primitives are `std` and `core::arch` intrinsics; M6 does not need SIMD.

## Decision

### 1. A strict `std::net::TcpListener` server, one request per connection

The server uses `std::net::TcpListener` in non-blocking mode so the accept loop
can observe a shutdown flag without a signal handler. Each accepted stream is
switched back to blocking mode and handled on a bounded thread. One request is
read, one response is written, and the connection is closed with
`Connection: close`.

This deliberately excludes keep-alive, pipelining, chunked transfer encoding,
HTTP/2, and `Expect: 100-continue`. Those features enlarge the state machine and
the attack surface without serving the local-service use case. A client that
needs many requests opens many connections; the connection cap bounds that.

The request parser is strict:

- only HTTP/1.1, GET/POST/DELETE;
- a request target that starts with `/`, with no fragment, control character,
  or absolute URI;
- a mandatory non-empty `Host` header;
- decimal `Content-Length` only, checked before allocation;
- no `Transfer-Encoding`, no `Expect`, no obsolete line folding, no bare LF;
- bounded request line, header line, total headers, header count, and body;
- duplicate `Host`, `Content-Length`, `Transfer-Encoding`, `Authorization`,
  and `Expect` headers are rejected.

The body cap is checked before the body is allocated. The parser returns a
typed `HttpError` with a status and stable code; it never panics on hostile
input.

### 2. JSON is the internal parser and serializer

The API reuses `crate::json` with API-specific limits. Unknown fields are
rejected at every object level, so a typo cannot silently change behavior.
Errors use one schema:

    {"error":{"code":"...","message":"..."}}

The serializer emits deterministic BTreeMap key order. Non-finite numbers are
rejected. JSON responses are small; embedding vectors are serialized as arrays
of f64 values converted from f32.

### 3. Bounded connections and bounded jobs

The accept loop increments an atomic active-connection counter and refuses a
connection above `max_connections` with 503. The rejected request is drained
first so closing the socket does not turn the response into a TCP reset.

Ingestion is asynchronous. A bounded queue (`JobQueue`) has a fixed number of
worker threads, a bounded pending count (queued + running), deterministic
positive job ids, cooperative cancellation, and a bounded retained history.
`POST /v1/documents` validates and enqueues a task and returns 202 with the job
id; `GET /v1/jobs/{id}` reports status; `DELETE /v1/jobs/{id}` requests
cancellation.

The queue never grows without bound. A full queue returns 429. A task checks the
cancel flag before and after the embedding phase and before the storage write.
A kernel already running inside a model forward pass cannot be interrupted by
this mechanism; the design documents that limitation rather than claiming
preemption.

Searches open a `ReadOnlyStore` view and rebuild the in-memory index. This
avoids holding the writer lock during a search. Writes serialize through a
`Mutex<Store>` and commit one atomic transaction. The index is rebuilt per
search at M6; caching and HNSW are deferred until measurements justify them.

### 4. Model loading: text at startup, media lazily

The text model is loaded at startup. This gives `/ready` an honest storage and
text-model signal and exposes the embedding dimension before the first request.
The tokenizer is loaded when `tokenizer.json` exists; a model without it still
serves explicit token-id embeddings (the reproducible tiny-fixture path) and
returns 501 for text input.

Vision and audio towers are large. They are loaded lazily on the first request
for their modality, behind a mutex, and reused. The configuration is validated at
startup so `/v1/models` does not claim a modality the frozen configuration does
not declare. `/ready` reports declared modalities separately from currently
loaded towers. This avoids allocating media weights in a text-only deployment
while remaining truthful about readiness.

The tiny audio fixture has no tokenizer because a 64-entry vocabulary cannot
contain the 256 byte-fallback tokens required by the frozen tokenizer pipeline.
`AudioEmbedder` therefore accepts an absent tokenizer for audio-only use and
uses the special ids from the configuration; text embedding through that
embedder returns 501. The frozen production model always has a tokenizer, and
the existing validation still runs when it is present.

### 5. Authentication and network posture

The default bind is `127.0.0.1:8080`. A non-loopback bind refuses to start
without an access token of at least 16 bytes. When a token is configured, every
request must present `Authorization: Bearer <token>` or `X-API-Key: <token>`.
The comparison is constant-time. The token is never logged or included in a
response.

There is no TLS implementation. The documentation forbids exposing the service
directly to the Internet. A trusted LAN deployment needs a token and an external
firewall or TLS-terminating proxy. The server does not trust forwarded headers,
does not implement CORS, and does not accept cookies.

`POST /v1/shutdown` is restricted to a loopback peer and still requires the
token when one is configured. This is the supported graceful-shutdown
mechanism. `std` does not expose POSIX signal handling, and the project forbids
FFI and unsafe code, so catching SIGINT would require a forbidden mechanism.

### 6. Media input is bounded base64, never a client path

`POST /v1/embeddings`, `POST /v1/documents`, and `POST /v1/search` accept media
as base64 in the JSON body. The decoder is strict RFC 4648 standard base64 with
mandatory padding, canonical trailing bits, no whitespace, and a 16 MiB decoded
cap. PNG versus JPEG is detected from the leading bytes. Audio must be RIFF/WAVE.
The native decoders then apply their own dimension, pixel, sample, MCU, block,
and decompressed-memory caps.

The API does not read arbitrary local paths supplied by a client. This avoids
turning the service into a file oracle for a LAN client. The CLI retains its
path-based interface for local development.

### 7. Error mapping

The central `Error` taxonomy maps to HTTP statuses:

- Usage → 400 invalid_request;
- Corrupt → 400 invalid_input;
- LimitExceeded → 413 payload_too_large;
- Unsupported → 501 unsupported;
- Io → 500 internal_error (or 503 for WouldBlock);
- Internal → 500 internal_error.

Malformed HTTP has its own statuses: 400, 408, 413, 415, 417, 501. A full
connection limit is 503; a full job queue is 429. The mapping is documented in
`docs/api.md` so a client can distinguish a bad request, a saturated service,
and a missing feature.

### 8. Observability

`GET /v1/stats` exposes monotonic counters for requests, 2xx/4xx/5xx responses,
bytes in/out, active connections, rejected connections, thread-spawn failures,
job states, document and chunk counts, and recovery counters. `GET /health` is
process liveness; `GET /ready` is storage/model readiness. No metric contains a
token, document text, or embedding value.

## Alternatives considered

- **Third-party HTTP server or JSON crate:** forbidden by the zero-crate
  contract and would add a large unaudited dependency surface.
- **Keep-alive and pipelining:** more state, more timeouts, and no benefit for
  the local-service target. One request per connection is simpler to bound.
- **Synchronous document ingestion:** simpler, but a large media embedding would
  block the request thread and make cancellation and queue limits meaningless.
- **Unbounded ingestion queue:** violates the explicit memory-bound invariant.
- **Eager loading of vision and audio towers:** simpler readiness semantics, but
  multiplies memory for deployments that only use text.
- **Path-based media input:** convenient locally, but exposes arbitrary file
  reads to LAN clients.
- **Catching SIGINT with `libc`:** forbidden FFI. The loopback shutdown route is
  the explicit alternative.

## Consequences

- The server is intentionally narrow: HTTP/1.1, one request per connection,
  JSON only, no TLS.
- A client must base64-encode media and must poll a job for asynchronous
  ingestion.
- Media towers pay a one-time lazy-load cost on the first request.
- The index is rebuilt per search; this is acceptable for M6 and measured later.
- Cancellation is cooperative and cannot interrupt an in-flight model kernel.
- The tiny audio fixture path is supported without a tokenizer; text embedding
  through that fixture is explicitly unavailable rather than faked.
- No third-party crate, FFI, or unsafe block is introduced.

## Verification

- `cargo fmt --check`
- `cargo clippy --all-targets -- -D warnings`
- `cargo test`
- `cargo build --release --target aarch64-apple-darwin`
- `cargo tree --edges normal,build,dev` shows only the root crate.

The `tests/http_api.rs` integration suite starts the real server on an
ephemeral loopback port and speaks raw HTTP/1.1. It covers health, readiness,
models, stats, token embeddings, audio embeddings and audio document ingestion on
the tiny fixture, lexical/vector/hybrid search, metadata filters, Unicode JSON,
percent-encoded deletion, duplicate-id job failure, authentication, malformed
requests, oversized bodies, transfer encoding, `Expect`, connection limits,
slow-request timeouts, concurrent health requests, and graceful shutdown. The
full-model text and image HTTP embedding tests are ignored because they load the
frozen model; the existing text/image parity suites cover the numerical path.

## References

- `docs/api.md` — API contract and examples.
- ADR 0004 — text embeddings and Matryoshka dimensions.
- ADR 0005/0006 — persistent store, journal, snapshots, and read-only views.
- ADR 0009 — image preprocessing and vision tower.
- ADR 0010 — WAV decoding and audio embeddings.
- `seldon-vault-lite-plan.md` section 11 — HTTP API and CLI requirements.
