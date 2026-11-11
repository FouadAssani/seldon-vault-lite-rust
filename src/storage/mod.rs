//! Native persistent storage for documents, chunks, and embeddings.
//!
//! Milestone M3 provides an append-only journal with framed CRC-32 records,
//! atomic transactions, deterministic replay, crash recovery, snapshots, safe
//! compaction, an exclusive writer lock, and a lock-free read-only view. The
//! journal and snapshot are the source of truth; the store rebuilds its
//! deterministic in-memory maps on open.
//!
//! No external database or serialization crate is used.

pub mod checksum;
pub mod codec;
pub mod journal;
pub mod model;
pub mod snapshot;
pub mod store;

pub use store::{Compaction, ReadOnlyStore, Recovery, Store, Verification};
