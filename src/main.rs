//! Demo CLI: index a few toy documents and run a similarity query.

use vector_search_engine::{index::BruteForceIndex, Vector};

fn main() {
    let mut idx = BruteForceIndex::new(3);
    idx.add("doc-1", Vector::new(vec![1.0, 0.0, 0.0]));
    idx.add("doc-2", Vector::new(vec![0.0, 1.0, 0.0]));
    idx.add("doc-3", Vector::new(vec![0.9, 0.1, 0.0]));

    let query = Vector::new(vec![0.8, 0.2, 0.0]);
    println!("query: {:?}", query.dims());
    for (id, score) in idx.search(&query, 2) {
        println!("{id}: {score:.4}");
    }
}
