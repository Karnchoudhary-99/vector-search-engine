//! Vector Search Engine — library root.
//!
//! Building blocks for similarity search over dense embeddings:
//! [`Vector`](vector::Vector), distance metrics ([`distance`]),
//! and index structures ([`index`]).

pub mod distance;
pub mod index;
pub mod vector;

pub use vector::Vector;
