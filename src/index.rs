//! Exact (brute-force) similarity index.
//!
//! Scores every stored vector against the query with cosine similarity.
//! O(n) per query — the correctness baseline that the upcoming HNSW
//! approximate index will be benchmarked against.

use crate::{distance::cosine_similarity, vector::Vector};

/// Exact top-K search over an in-memory set of vectors.
pub struct BruteForceIndex {
    dim: usize,
    ids: Vec<String>,
    vectors: Vec<Vector>,
}

impl BruteForceIndex {
    /// Create an empty index for `dim`-dimensional vectors.
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "dimension must be positive");
        Self {
            dim,
            ids: Vec::new(),
            vectors: Vec::new(),
        }
    }

    /// Insert a vector under `id`. Panics on dimension mismatch.
    pub fn add(&mut self, id: impl Into<String>, v: Vector) {
        assert_eq!(v.len(), self.dim, "vector dimension mismatch");
        self.ids.push(id.into());
        self.vectors.push(v);
    }

    /// Return up to `top_k` `(id, cosine_similarity)` pairs, best first.
    pub fn search(&self, query: &Vector, top_k: usize) -> Vec<(String, f32)> {
        assert_eq!(query.len(), self.dim, "query dimension mismatch");
        let mut scored: Vec<(String, f32)> = self
            .ids
            .iter()
            .zip(self.vectors.iter())
            .map(|(id, v)| (id.clone(), cosine_similarity(query, v)))
            .collect();
        // Sort descending by score; NaN is impossible here (guarded in cosine_similarity).
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(top_k);
        scored
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// True when the index holds nothing.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_returns_best_match_first() {
        let mut idx = BruteForceIndex::new(2);
        idx.add("a", Vector::new(vec![1.0, 0.0]));
        idx.add("b", Vector::new(vec![0.0, 1.0]));
        let hits = idx.search(&Vector::new(vec![0.9, 0.1]), 2);
        assert_eq!(hits[0].0, "a");
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn top_k_is_respected() {
        let mut idx = BruteForceIndex::new(2);
        for i in 0..5 {
            idx.add(format!("d{i}"), Vector::new(vec![i as f32, 0.0]));
        }
        assert_eq!(idx.search(&Vector::new(vec![1.0, 0.0]), 3).len(), 3);
    }
}
