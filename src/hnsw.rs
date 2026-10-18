//! HNSW approximate nearest-neighbor index.
//!
//! A from-scratch implementation of "Efficient and robust approximate nearest
//! neighbor search using Hierarchical Navigable Small World graphs"
//! (Malkov & Yashunin, 2018).
//!
//! Design notes:
//! - Vectors are L2-normalized on insert, so cosine similarity reduces to a
//!   dot product and cosine distance to `1 - dot`.
//! - Zero external dependencies: level assignment uses an internal splitmix64
//!   stream, so builds are reproducible for a fixed seed.
//! - Neighbor selection uses the HNSW diversity heuristic, not just "closest
//!   M", which is what gives the graph its navigability.

use crate::vector::Vector;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Cosine distance between two (already normalized) vectors, in [0, 2].
#[inline]
fn dist(a: &Vector, b: &Vector) -> f32 {
    1.0 - a.dot(b)
}

/// Distance wrapper with a total order (no NaNs can occur: stored vectors are
/// normalized and finite, so every dot product is finite).
#[derive(Debug, Clone, Copy, PartialEq)]
struct Dist(f32);

impl Eq for Dist {}

impl PartialOrd for Dist {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Dist {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// A candidate node scored by distance to the query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Scored {
    dist: Dist,
    node: usize,
}

impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Scored {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist.cmp(&other.dist).then(self.node.cmp(&other.node))
    }
}

/// One stored vector: its external id, normalized components, top layer,
/// and soft-delete flag (tombstones keep the graph navigable).
#[derive(Debug, Clone)]
pub(crate) struct Node {
    pub(crate) id: String,
    pub(crate) vector: Vector,
    pub(crate) level: usize,
    pub(crate) deleted: bool,
}

/// Hierarchical Navigable Small World index for approximate top-K search.
///
/// Build with [`HnswIndex::new`] (or [`HnswIndex::with_seed`] for reproducible
/// builds), insert with [`HnswIndex::add`], query with
/// [`HnswIndex::search`].
///
/// ### Tuning knobs
/// - `m`: max connections per node per layer (paper suggests 5–48; 16 is a
///   good default). Higher = better recall, more memory.
/// - `ef_construction`: beam width during inserts (higher = better graph
///   quality, slower builds; 200 is a good default).
/// - `ef` (per query): beam width at search time. Raise it for more recall at
///   the cost of latency.
pub struct HnswIndex {
    dim: usize,
    m: usize,
    ef_construction: usize,
    ml: f64,
    max_level: usize,
    entry_point: Option<usize>,
    nodes: Vec<Node>,
    /// `layers[lc][node]` = neighbor node indices on layer `lc`.
    layers: Vec<HashMap<usize, Vec<usize>>>,
    /// External id -> node slot (includes tombstoned slots).
    id_lookup: HashMap<String, usize>,
    /// Vectors that are not tombstoned.
    live: usize,
    rng: u64,
}

impl HnswIndex {
    /// Create an empty index for `dim`-dimensional vectors.
    ///
    /// Panics if `dim == 0` or `m == 0`.
    pub fn new(dim: usize, m: usize, ef_construction: usize) -> Self {
        Self::with_seed(dim, m, ef_construction, 0x1234_5678_9ABC_DEF0)
    }

    /// Create an empty index with an explicit RNG seed (reproducible builds).
    pub fn with_seed(dim: usize, m: usize, ef_construction: usize, seed: u64) -> Self {
        assert!(dim > 0, "dimension must be positive");
        assert!(m > 0, "m must be positive");
        Self {
            dim,
            m,
            ef_construction: ef_construction.max(1),
            ml: 1.0 / (m as f64).ln(),
            max_level: 0,
            entry_point: None,
            nodes: Vec::new(),
            layers: Vec::new(),
            id_lookup: HashMap::new(),
            live: 0,
            rng: seed,
        }
    }

    /// Insert a vector under `id`. The vector is normalized on the way in.
    ///
    /// If `id` already exists and is live, the old entry is soft-deleted
    /// first (upsert semantics) — ids stay unique among live vectors.
    /// Panics on dimension mismatch.
    pub fn add(&mut self, id: impl Into<String>, v: Vector) {
        assert_eq!(v.len(), self.dim, "vector dimension mismatch");
        let id = id.into();
        if self.contains(&id) {
            self.remove(&id);
        }
        let vector = v.normalized();
        let level = self.random_level();
        let node = self.nodes.len();
        self.nodes.push(Node {
            id: id.clone(),
            vector,
            level,
            deleted: false,
        });
        self.id_lookup.insert(id, node);
        self.live += 1;
        while self.layers.len() <= level {
            self.layers.push(HashMap::new());
        }

        let Some(mut cur) = self.entry_point else {
            self.entry_point = Some(node);
            self.max_level = level;
            return;
        };

        let query = self.nodes[node].vector.clone();

        // Phase 1: greedy descent from the top layer to `level + 1`.
        for lc in (level + 1..=self.max_level).rev() {
            cur = self.greedy_search(&query, cur, lc);
        }

        // Phase 2: beam-search and connect on every layer down to 0.
        for lc in (0..=level.min(self.max_level)).rev() {
            let candidates = self.search_layer(&query, cur, self.ef_construction, lc);
            let neighbors: Vec<usize> = Self::select_neighbors(&self.nodes, candidates, self.m)
                .into_iter()
                .map(|s| s.node)
                .collect();
            self.connect(node, &neighbors, lc);
            cur = neighbors.first().copied().unwrap_or(cur);
        }

        if level > self.max_level {
            self.entry_point = Some(node);
            self.max_level = level;
        }
    }

    /// Approximate top-K search. Returns up to `top_k` `(id, cosine_similarity)`
    /// pairs, best first. `ef` controls the recall/latency trade-off at query
    /// time (values below `top_k` are raised to `top_k`).
    ///
    /// Soft-deleted (tombstoned) vectors are skipped: the search over-fetches
    /// to compensate, so results still contain up to `top_k` *live* vectors.
    pub fn search(&self, query: &Vector, top_k: usize, ef: usize) -> Vec<(String, f32)> {
        assert_eq!(query.len(), self.dim, "query dimension mismatch");
        if top_k == 0 {
            return Vec::new();
        }
        let Some(mut cur) = self.entry_point else {
            return Vec::new();
        };
        let q = query.normalized();

        for lc in (1..=self.max_level).rev() {
            cur = self.greedy_search(&q, cur, lc);
        }

        let tombstones = self.nodes.len() - self.live;
        let fetch = top_k + tombstones;
        let ef = ef.max(fetch).max(1);
        let hits = self.search_layer(&q, cur, ef, 0);
        let mut out = Vec::with_capacity(top_k.min(self.live));
        for s in hits {
            if !self.nodes[s.node].deleted {
                out.push((self.nodes[s.node].id.clone(), 1.0 - s.dist.0));
                if out.len() == top_k {
                    break;
                }
            }
        }
        out
    }

    /// Soft-delete the vector stored under `id` (tombstone). The node stays in
    /// the graph so routing remains navigable; it is simply skipped by search
    /// and `len()`. Returns `true` when a live entry was deleted.
    pub fn remove(&mut self, id: &str) -> bool {
        match self.id_lookup.get(id) {
            Some(&node) if !self.nodes[node].deleted => {
                self.nodes[node].deleted = true;
                self.live -= 1;
                true
            }
            _ => false,
        }
    }

    /// True when a live (non-deleted) vector exists under `id`.
    pub fn contains(&self, id: &str) -> bool {
        self.id_lookup
            .get(id)
            .is_some_and(|&n| !self.nodes[n].deleted)
    }

    /// Number of live (non-deleted) vectors.
    pub fn len(&self) -> usize {
        self.live
    }

    /// Number of soft-deleted (tombstoned) vectors.
    pub fn tombstones(&self) -> usize {
        self.nodes.len() - self.live
    }

    /// True when the index holds no live vectors.
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Vector dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Max connections per node per layer.
    pub fn m(&self) -> usize {
        self.m
    }

    /// Beam width used during construction.
    pub fn ef_construction(&self) -> usize {
        self.ef_construction
    }

    // ---- internals -------------------------------------------------------

    /// Sample a layer for a new node: `floor(-ln(U) * ml)`.
    fn random_level(&mut self) -> usize {
        // splitmix64
        let mut z = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        self.rng = z;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let u = ((z >> 11) as f64) / 9007199254740992.0; // 2^53
        let u = u.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
        (-u.ln() * self.ml).floor() as usize
    }

    /// Greedy walk on one layer: repeatedly move to the closest neighbor
    /// until no neighbor is closer. Returns the closest node found.
    fn greedy_search(&self, query: &Vector, mut cur: usize, lc: usize) -> usize {
        let mut best = dist(query, &self.nodes[cur].vector);
        loop {
            let mut improved = false;
            if let Some(neigh) = self.layers.get(lc).and_then(|l| l.get(&cur)) {
                for &n in neigh {
                    let d = dist(query, &self.nodes[n].vector);
                    if d < best {
                        best = d;
                        cur = n;
                        improved = true;
                    }
                }
            }
            if !improved {
                return cur;
            }
        }
    }

    /// Beam search on one layer: explores up to `ef` candidates, returns them
    /// sorted by ascending distance.
    fn search_layer(&self, query: &Vector, entry: usize, ef: usize, lc: usize) -> Vec<Scored> {
        let ef = ef.max(1);
        let mut visited = HashSet::new();
        visited.insert(entry);

        let d0 = Dist(dist(query, &self.nodes[entry].vector));
        // `candidates`: min-heap (closest first). `found`: max-heap (farthest first).
        let mut candidates: BinaryHeap<Reverse<Scored>> = BinaryHeap::new();
        candidates.push(Reverse(Scored {
            dist: d0,
            node: entry,
        }));
        let mut found: BinaryHeap<Scored> = BinaryHeap::new();
        found.push(Scored {
            dist: d0,
            node: entry,
        });

        while let Some(Reverse(cur)) = candidates.pop() {
            let worst = found.peek().expect("found is never empty").dist;
            if cur.dist > worst {
                break; // remaining candidates can't beat the worst result
            }
            if let Some(neigh) = self.layers.get(lc).and_then(|l| l.get(&cur.node)) {
                for &n in neigh {
                    if !visited.insert(n) {
                        continue;
                    }
                    let d = Dist(dist(query, &self.nodes[n].vector));
                    if found.len() < ef || d < worst {
                        candidates.push(Reverse(Scored { dist: d, node: n }));
                        found.push(Scored { dist: d, node: n });
                        if found.len() > ef {
                            found.pop();
                        }
                    }
                }
            }
        }

        let mut out: Vec<Scored> = found.into_vec();
        out.sort();
        out
    }

    /// HNSW neighbor-selection heuristic: prefer candidates that are closer to
    /// the query than to any already-selected neighbor (diversity pruning),
    /// then fill up with the closest remaining candidates.
    fn select_neighbors(nodes: &[Node], mut candidates: Vec<Scored>, m: usize) -> Vec<Scored> {
        let mut selected: Vec<Scored> = Vec::with_capacity(m);
        for c in candidates.drain(..) {
            if selected.len() >= m {
                break;
            }
            let vc = &nodes[c.node].vector;
            let dc = c.dist.0;
            let diverse = selected
                .iter()
                .all(|s| dc < dist(vc, &nodes[s.node].vector));
            if diverse {
                selected.push(c);
            }
        }
        if selected.len() < m {
            for c in candidates {
                if selected.len() >= m {
                    break;
                }
                if !selected.iter().any(|s| s.node == c.node) {
                    selected.push(c);
                }
            }
        }
        selected
    }

    /// Wire `node` bidirectionally to `neighbors` on layer `lc`, pruning any
    /// adjacency list that grew past its cap back down to `m`.
    fn connect(&mut self, node: usize, neighbors: &[usize], lc: usize) {
        {
            let layer = &mut self.layers[lc];
            for &n in neighbors {
                layer.entry(node).or_default().push(n);
                layer.entry(n).or_default().push(node);
            }
        }
        let m = self.m;
        for &n in neighbors {
            let max_conn = if lc == 0 { m * 2 } else { m };
            let overfull = self.layers[lc]
                .get(&n)
                .map(|v| v.len() > max_conn)
                .unwrap_or(false);
            if overfull {
                let qv = self.nodes[n].vector.clone();
                let mut cands: Vec<Scored> = self.layers[lc][&n]
                    .iter()
                    .map(|&x| Scored {
                        dist: Dist(dist(&qv, &self.nodes[x].vector)),
                        node: x,
                    })
                    .collect();
                cands.sort();
                let pruned = Self::select_neighbors(&self.nodes, cands, m);
                self.layers[lc].insert(n, pruned.into_iter().map(|s| s.node).collect());
            }
        }
    }

    // ---- persistence accessors (same crate) -------------------------------

    pub(crate) fn max_level(&self) -> usize {
        self.max_level
    }

    pub(crate) fn entry_point(&self) -> Option<usize> {
        self.entry_point
    }

    pub(crate) fn raw_nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub(crate) fn raw_layers(&self) -> &[HashMap<usize, Vec<usize>>] {
        &self.layers
    }

    pub(crate) fn rng_state(&self) -> u64 {
        self.rng
    }

    /// Rebuild an index from persisted parts. `layers` must be consistent with
    /// `nodes` (as written by the persistence layer).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        dim: usize,
        m: usize,
        ef_construction: usize,
        max_level: usize,
        entry_point: Option<usize>,
        nodes: Vec<Node>,
        layers: Vec<HashMap<usize, Vec<usize>>>,
        rng: u64,
    ) -> Self {
        let mut id_lookup = HashMap::with_capacity(nodes.len());
        let mut live = 0;
        for (slot, node) in nodes.iter().enumerate() {
            // Later slots win for duplicate ids, matching `add`'s upsert
            // semantics on re-insert.
            id_lookup.insert(node.id.clone(), slot);
            if !node.deleted {
                live += 1;
            }
        }
        Self {
            dim,
            m,
            ef_construction: ef_construction.max(1),
            ml: 1.0 / (m as f64).ln(),
            max_level,
            entry_point,
            nodes,
            layers,
            id_lookup,
            live,
            rng,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(dims: &[f32]) -> Vector {
        Vector::new(dims.to_vec())
    }

    #[test]
    fn empty_index_returns_no_hits() {
        let idx = HnswIndex::new(2, 8, 50);
        assert!(idx.is_empty());
        assert!(idx.search(&v(&[1.0, 0.0]), 5, 10).is_empty());
    }

    #[test]
    fn single_vector_is_found() {
        let mut idx = HnswIndex::new(2, 8, 50);
        idx.add("only", v(&[1.0, 0.0]));
        let hits = idx.search(&v(&[0.9, 0.1]), 5, 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, "only");
        assert!((hits[0].1 - 1.0).abs() < 0.01);
    }

    #[test]
    fn nearest_neighbor_matches_brute_force() {
        let mut hnsw = HnswIndex::with_seed(4, 16, 100, 42);
        let data: Vec<Vector> = (0..200)
            .map(|i| {
                let x = i as f32;
                v(&[(x % 7.0) - 3.0, (x % 11.0) - 5.0, (x % 5.0) - 2.0, 1.0])
            })
            .collect();
        for (i, vec) in data.iter().enumerate() {
            hnsw.add(format!("d{i}"), vec.clone());
        }
        assert_eq!(hnsw.len(), 200);

        let query = v(&[0.5, -1.5, 0.5, 1.0]);
        let approx = hnsw.search(&query, 1, 100);
        // brute-force ground truth
        let qn = query.normalized();
        let mut best = (String::new(), f32::NEG_INFINITY);
        for (i, vec) in data.iter().enumerate() {
            let s = qn.dot(&vec.normalized());
            if s > best.1 {
                best = (format!("d{i}"), s);
            }
        }
        assert_eq!(approx[0].0, best.0, "top-1 must match brute force");
    }

    #[test]
    fn recall_at_10_on_clustered_data() {
        // 5 clusters in 8-D; HNSW should recover ~all of brute force's top-10.
        let mut hnsw = HnswIndex::with_seed(8, 16, 200, 7);
        let mut data: Vec<Vector> = Vec::new();
        for c in 0..5 {
            let center: Vec<f32> = (0..8).map(|d| (c * 8 + d) as f32).collect();
            for i in 0..60 {
                let jitter = |k: usize| ((i * 31 + k * 17 + c * 13) % 100) as f32 / 100.0 - 0.5;
                let comps: Vec<f32> = center
                    .iter()
                    .enumerate()
                    .map(|(d, x)| x + jitter(d))
                    .collect();
                data.push(v(&comps));
            }
        }
        for (i, vec) in data.iter().enumerate() {
            hnsw.add(format!("p{i}"), vec.clone());
        }

        let query = v(&[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]); // near cluster 0
        let approx: Vec<String> = hnsw
            .search(&query, 10, 100)
            .into_iter()
            .map(|(id, _)| id)
            .collect();

        let qn = query.normalized();
        let mut scored: Vec<(f32, String)> = data
            .iter()
            .enumerate()
            .map(|(i, vec)| (qn.dot(&vec.normalized()), format!("p{i}")))
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        let truth: Vec<String> = scored.into_iter().take(10).map(|(_, id)| id).collect();

        let hits = approx.iter().filter(|id| truth.contains(id)).count();
        let recall = hits as f32 / 10.0;
        assert!(
            recall >= 0.9,
            "recall@10 = {recall}, expected >= 0.9 (hits: {hits}/10)"
        );
    }

    #[test]
    fn search_results_are_sorted_best_first() {
        let mut idx = HnswIndex::with_seed(3, 8, 50, 1);
        for i in 0..50 {
            idx.add(format!("n{i}"), v(&[i as f32, 1.0, 0.5]));
        }
        let hits = idx.search(&v(&[25.0, 1.0, 0.5]), 10, 50);
        assert_eq!(hits.len(), 10);
        for w in hits.windows(2) {
            assert!(w[0].1 >= w[1].1, "results must be best-first");
        }
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn rejects_wrong_dimension_on_add() {
        let mut idx = HnswIndex::new(2, 8, 50);
        idx.add("bad", v(&[1.0, 2.0, 3.0]));
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn rejects_wrong_dimension_on_search() {
        let idx = HnswIndex::new(2, 8, 50);
        idx.search(&v(&[1.0]), 5, 10);
    }

    #[test]
    fn remove_hides_vector_from_search() {
        let mut idx = HnswIndex::with_seed(2, 8, 50, 3);
        idx.add("a", v(&[1.0, 0.0]));
        idx.add("b", v(&[0.0, 1.0]));
        idx.add("c", v(&[-1.0, 0.0]));
        assert!(idx.remove("b"));
        assert!(!idx.remove("b")); // already gone
        assert!(!idx.remove("nope")); // never existed
        assert_eq!(idx.len(), 2);
        assert_eq!(idx.tombstones(), 1);
        assert!(!idx.contains("b"));
        assert!(idx.contains("a"));

        let hits = idx.search(&v(&[0.0, 1.0]), 5, 20);
        assert_eq!(hits.len(), 2);
        assert!(hits.iter().all(|(id, _)| id != "b"));
    }

    #[test]
    fn add_with_duplicate_id_replaces_old_vector() {
        let mut idx = HnswIndex::with_seed(2, 8, 50, 5);
        idx.add("dup", v(&[1.0, 0.0]));
        idx.add("dup", v(&[0.0, 1.0])); // upsert: old entry tombstoned
        assert_eq!(idx.len(), 1);
        assert_eq!(idx.tombstones(), 1);
        let hits = idx.search(&v(&[0.0, 1.0]), 1, 20);
        assert_eq!(hits[0].0, "dup");
        assert!(hits[0].1 > 0.99);
    }

    #[test]
    fn search_still_returns_top_k_after_many_deletes() {
        let mut idx = HnswIndex::with_seed(4, 16, 100, 11);
        for i in 0..100 {
            idx.add(
                format!("d{i}"),
                v(&[(i % 7) as f32, (i % 5) as f32, 1.0, 0.5]),
            );
        }
        for i in (0..100).step_by(2) {
            idx.remove(&format!("d{i}"));
        }
        assert_eq!(idx.len(), 50);
        let hits = idx.search(&v(&[1.0, 1.0, 1.0, 0.5]), 10, 100);
        assert_eq!(hits.len(), 10);
        assert!(hits
            .iter()
            .all(|(id, _)| { id.strip_prefix('d').unwrap().parse::<usize>().unwrap() % 2 == 1 }));
    }
}
