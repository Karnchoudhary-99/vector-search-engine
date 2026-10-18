//! End-to-end semantic search demo: embed a few documents about databases,
//! build an HNSW index, and run a query.
//!
//! Run with: `cargo run --example semantic_search`

use vector_search_engine::{hnsw::HnswIndex, text};

fn main() {
    let docs = [
        (
            "db-1",
            "A vector database stores embeddings for similarity search",
        ),
        (
            "db-2",
            "PostgreSQL is a relational database with SQL support",
        ),
        (
            "db-3",
            "HNSW graphs enable fast approximate nearest neighbor search",
        ),
        (
            "db-4",
            "Redis can be used as an in-memory cache and message broker",
        ),
        (
            "db-5",
            "Cosine similarity measures the angle between embedding vectors",
        ),
        ("db-6", "A recipe for chocolate cake with cocoa and flour"),
    ];
    let dim = 128;

    let mut idx = HnswIndex::new(dim, 16, 200);
    for (id, doc) in docs {
        idx.add(id, text::embed(doc, dim));
    }

    let query = "fast vector database for similarity search";
    println!("query: {query}\n");
    for (id, score) in idx.search(&text::embed(query, dim), 3, 50) {
        let doc = docs
            .iter()
            .find(|(d, _)| *d == id)
            .map(|(_, t)| *t)
            .unwrap();
        println!("{score:.4}  {id}  — {doc}");
    }
}
