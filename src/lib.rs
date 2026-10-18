//! Vector Search Engine — library root.
//!
//! Building blocks for similarity search over dense embeddings:
//! [`Vector`](vector::Vector), distance metrics ([`distance`]), exact search
//! ([`index`]), approximate search ([`hnsw`]), binary persistence
//! ([`persistence`]), and demo-grade text embeddings ([`text`]).

pub mod distance;
pub mod hnsw;
pub mod index;
pub mod persistence;
pub mod text;
pub mod vector;

pub use vector::Vector;
