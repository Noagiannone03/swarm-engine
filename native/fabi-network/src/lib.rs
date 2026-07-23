//! Fabi's authenticated, relay-capable QUIC transport primitives.
//!
//! The transport intentionally owns only identity, connectivity and bounded
//! byte streams. Scheduler membership, routing and model semantics stay in the
//! Python control plane.

pub mod catalog;
pub mod catalog_dht;
pub mod endpoint;
pub mod identity;
pub mod protocol;
pub mod telemetry;

#[cfg(feature = "python")]
mod python;

pub const ALPN: &[u8] = b"fabi/network/1";
