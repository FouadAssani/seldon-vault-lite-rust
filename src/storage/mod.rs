//! Native persistent storage for documents, chunks, and embeddings.
//!
//! Milestone M3 introduces an append-only journal with framed records,
//! CRC-32 checksums, atomic transactions, deterministic replay, crash recovery,
//! and an exclusive writer lock. The journal is the source of truth; the store
//! rebuilds its deterministic in-memory maps on open. Snapshots and compaction
//! are the next storage increment and are not claimed yet.
//!
//! No external database or serialization crate is used.

pub mod checksum;
pub mod codec;
pub mod journal;
pub mod model;
pub mod store;

pub use store::{Recovery, Store, Verification};
