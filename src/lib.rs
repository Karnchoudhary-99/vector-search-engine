//! Vector Search Engine — library root.
//!
//! Building blocks for similarity search over dense embeddings:
//! [`Vector`](vector::Vector), distance metrics ([`distance`]), exact search
//! ([`index`]), approximate search with soft-delete ([`hnsw`]), binary
//! persistence ([`persistence`]), demo-grade text embeddings ([`text`]), and
//! the axum REST API ([`server`]).

pub mod distance;
pub mod hnsw;
pub mod index;
pub mod persistence;
pub mod server;
pub mod text;
pub mod vector;

pub use vector::Vector;
