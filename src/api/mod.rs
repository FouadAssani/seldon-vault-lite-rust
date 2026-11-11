//! Native HTTP/1.1 API, model dispatch, and hardening for milestone M6.
//!
//! The module is split into a strict HTTP parser/writer, a model engine, and a
//! server with bounded connections and jobs. No third-party HTTP or JSON crate
//! is used; the listener is `std::net::TcpListener` and the JSON is produced by
//! `crate::json`.
//!
//! The public surface is `ServerConfig`, `Server`, and `RunningServer`. The
//! integration tests start a real server on an ephemeral loopback port and
//! speak raw HTTP/1.1 over `std::net::TcpStream`, so the tests exercise the same
//! parser and response writer as a client.

mod base64;
mod engine;
mod http;
mod server;

pub use server::{
    DEFAULT_BIND, DEFAULT_JOB_WORKERS, DEFAULT_MAX_BODY_BYTES, DEFAULT_MAX_CONNECTIONS,
    DEFAULT_QUEUE_CAPACITY, DEFAULT_REQUEST_TIMEOUT_MS, MAX_BODY_BYTES, MAX_CONNECTIONS,
    MAX_REJECT_DRAIN_BYTES, MAX_SEARCH_LIMIT, MAX_SEARCH_OFFSET, MAX_TEXT_INPUT_BYTES,
    MAX_TOKEN_BYTES, MAX_TOKEN_IDS, MIN_BODY_BYTES, MIN_LAN_TOKEN_BYTES, MIN_REQUEST_TIMEOUT,
    RunningServer, Server, ServerConfig,
};
