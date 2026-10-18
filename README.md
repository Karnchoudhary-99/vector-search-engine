# Vector Search Engine

Blazing-fast **vector similarity search** in Rust — the core building block behind
semantic search and RAG (Retrieval-Augmented Generation) pipelines.

## Why this exists

Modern AI apps (chatbots, recommenders, code search) all need the same primitive:
*"given this embedding, find the K most similar vectors — fast."*
This project builds that primitive from scratch in Rust: correct first, then fast.

## Features (roadmap)

- [x] Core `Vector` type (dot product, norm, normalization)
- [x] Distance metrics: euclidean, cosine, manhattan
- [x] Brute-force top-K similarity index
- [ ] HNSW approximate-nearest-neighbor index
- [ ] On-disk persistence (mmap)
- [ ] HTTP server (`/index`, `/search`)
- [ ] Benchmarks vs brute force (recall/latency)

## Quick start

```bash
cargo run --release
```

Sample output:

```
doc-3: 0.9939
doc-1: 0.8660
```

## Library usage

```rust
use vector_search_engine::{Vector, index::BruteForceIndex};

let mut idx = BruteForceIndex::new(3);
idx.add("doc-1", Vector::new(vec![1.0, 0.0, 0.0]));
let hits = idx.search(&Vector::new(vec![0.9, 0.1, 0.0]), 5);
```

## License

MIT — see [LICENSE](LICENSE).
