# Vector Search Engine

Blazing-fast **vector similarity search** in Rust — the core primitive behind
semantic search and RAG (Retrieval-Augmented Generation) pipelines. Zero
dependencies, from-scratch HNSW.

## Why this exists

Modern AI apps (chatbots, recommenders, code search) all need the same primitive:
*"given this embedding, find the K most similar vectors — fast."*
This project builds that primitive in Rust: correct first, then fast, with the
exact brute-force index as the ground-truth baseline.

## Features

- [x] Core `Vector` type (dot product, norm, normalization)
- [x] Distance metrics: euclidean, cosine, manhattan
- [x] Brute-force top-K similarity index (correctness baseline)
- [x] **HNSW approximate-nearest-neighbor index** (from scratch, no deps)
- [x] Versioned binary persistence (`VSE1` format; graph saved verbatim)
- [x] CLI: `index` / `search` / `bench` subcommands
- [x] Demo-grade text embedder (hashed n-grams — zero model downloads)
- [ ] HTTP server (`/index`, `/search`)

## Architecture

```
                    ┌─────────────┐
 query ────────────▶│  HNSW index │──▶ top-K (id, cosine similarity)
                    └──────┬──────┘
                           │ layers[2]  sparse "highway" layer
                    ┌──────┴──────┐
                    │ layers[1]   │
                    ┌──────┴──────┐
                    │ layers[0]   │── dense base layer, full recall lives here
                    └─────────────┘
```

- **Layer assignment:** each vector draws a level from `⌊-ln(U)·m_L⌋`
  (splitmix64 stream — reproducible for a fixed seed).
- **Insert:** greedy descent from the top layer, then beam search
  (`ef_construction`) + HNSW diversity-heuristic neighbor selection on every
  layer down to 0, with bidirectional links and overfull-list pruning.
- **Query:** greedy descent to layer 0, then beam search with per-query `ef`.
  Raise `ef` for more recall, lower it for less latency.
- Vectors are L2-normalized on insert, so cosine similarity is just a dot
  product.

## Quick start

```bash
# Build an index from a text file (one "id<TAB>text" per line)
cargo run --release -- index --texts data/sample_docs.txt --out docs.vse

# Search it
cargo run --release -- search --index docs.vse --query "fast vector database"

# Benchmark brute force vs HNSW (build time, latency, recall@k)
cargo run --release -- bench --n 5000 --dim 64 --k 10

# Or run the in-code demo
cargo run --example semantic_search
```

Sample search output:

```
0.8123	db-1  — A vector database stores embeddings for similarity search
0.7741	db-3  — HNSW graphs enable fast approximate nearest neighbor search
0.7519	db-5  — Cosine similarity measures the angle between embedding vectors
```

## Library usage

```rust
use vector_search_engine::{Vector, hnsw::HnswIndex};

let mut idx = HnswIndex::new(128, 16, 200);
idx.add("doc-1", Vector::new(vec![0.1; 128]));

// ef = 50: higher = better recall, slower queries
let hits: Vec<(String, f32)> = idx.search(&Vector::new(vec![0.1; 128]), 5, 50);

// Persist and reload — the graph is saved verbatim
idx.save("docs.vse".as_ref()).unwrap();
let idx2 = HnswIndex::load("docs.vse".as_ref()).unwrap();
```

## Benchmarks

`vse bench` on synthetic clustered data (8 Gaussian clusters):

| vectors | dim | build (brute) | build (HNSW) | query (brute) | query (HNSW) | recall@10 |
|--------:|----:|--------------:|-------------:|--------------:|-------------:|----------:|
| 5,000   | 64  | 1 ms          | 1.03 s       | 1217 µs       | 191 µs       | 1.0000    |

*(measured with `vse bench --n 5000 --dim 64 --k 10`; dev-machine run —
run it yourself)*

HNSW trades a slower one-time build for ~6× faster queries at perfect
recall on this workload. Raise per-query `ef` for harder datasets.

## On-disk format

`VSE1`: magic bytes + kind tag + dimensionality, then entries (brute force)
or nodes + full layer graphs (HNSW), all little-endian. See
`src/persistence.rs` for the exact layout.

## Project layout

```
src/
  lib.rs          crate root
  vector.rs       dense Vector type + linear algebra
  distance.rs     euclidean / cosine / manhattan metrics
  index.rs        exact brute-force top-K index (ground truth)
  hnsw.rs         approximate HNSW index  ← the interesting part
  persistence.rs  VSE1 binary save/load
  text.rs         deterministic demo text embeddings
  main.rs         `vse` CLI (index / search / bench)
examples/
  semantic_search.rs   end-to-end demo
data/
  sample_docs.txt      tiny corpus for the CLI demo
```

## License

MIT — see [LICENSE](LICENSE).
