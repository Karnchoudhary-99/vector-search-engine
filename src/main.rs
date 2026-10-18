//! `vse` — command-line interface for the vector search engine.
//!
//! ```text
//! vse index  --texts <path> --out <path> [--dim 128] [--m 16] [--ef 200]
//!     Build an HNSW index from a text file (one `id<TAB>text` per line,
//!     or plain lines auto-numbered) and save it.
//!
//! vse search --index <path> --query <text> [--k 5] [--ef 50]
//!     Load an index and print the top-k matches for a text query.
//!
//! vse bench  [--n 5000] [--dim 64] [--k 10]
//!     Benchmark brute-force vs HNSW on synthetic clustered data:
//!     build time, mean query latency, and recall@k.
//! ```

use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use vector_search_engine::{hnsw::HnswIndex, index::BruteForceIndex, persistence, text, Vector};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("help");
    let rest = &args[2..];
    let code = match cmd {
        "index" => cmd_index(rest),
        "search" => cmd_search(rest),
        "bench" => cmd_bench(rest),
        _ => {
            eprintln!("{}", usage());
            2
        }
    };
    std::process::exit(code);
}

fn usage() -> &'static str {
    "usage:\n  \
     vse index  --texts <path> --out <path> [--dim 128] [--m 16] [--ef 200]\n  \
     vse search --index <path> --query <text> [--k 5] [--ef 50]\n  \
     vse bench  [--n 5000] [--dim 64] [--k 10]"
}

/// Value of `--name <value>`; `None` when absent.
fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn flag_or(args: &[String], name: &str, default: &str) -> String {
    flag(args, name).unwrap_or_else(|| default.to_string())
}

fn parse_usize(args: &[String], name: &str, default: usize) -> Result<usize, String> {
    flag_or(args, name, &default.to_string())
        .parse()
        .map_err(|_| format!("--{name} must be a positive integer"))
}

fn cmd_index(args: &[String]) -> i32 {
    let texts = match flag(args, "--texts") {
        Some(p) => PathBuf::from(p),
        None => return err("--texts <path> is required"),
    };
    let out = match flag(args, "--out") {
        Some(p) => PathBuf::from(p),
        None => return err("--out <path> is required"),
    };
    let dim = match parse_usize(args, "--dim", 128) {
        Ok(d) if d > 0 => d,
        _ => return err("--dim must be a positive integer"),
    };
    let m = match parse_usize(args, "--m", 16) {
        Ok(v) if v > 0 => v,
        _ => return err("--m must be a positive integer"),
    };
    let ef = match parse_usize(args, "--ef", 200) {
        Ok(v) => v,
        Err(e) => return err(&e),
    };

    let content = match fs::read_to_string(&texts) {
        Ok(c) => c,
        Err(e) => return err(&format!("cannot read {}: {e}", texts.display())),
    };
    let lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.is_empty() {
        return err("no documents found in input file");
    }

    let t0 = Instant::now();
    let mut idx = HnswIndex::new(dim, m, ef);
    for (i, line) in lines.iter().enumerate() {
        let (id, doc) = match line.split_once('\t') {
            Some((id, doc)) => (id.trim().to_string(), doc),
            None => (format!("doc-{i}"), *line),
        };
        idx.add(id, text::embed(doc, dim));
    }
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;

    if let Err(e) = idx.save(&out) {
        return err(&format!("cannot write {}: {e}", out.display()));
    }
    println!(
        "indexed {} documents (dim={dim}, m={m}) in {build_ms:.0} ms -> {}",
        idx.len(),
        out.display()
    );
    0
}

fn cmd_search(args: &[String]) -> i32 {
    let index = match flag(args, "--index") {
        Some(p) => PathBuf::from(p),
        None => return err("--index <path> is required"),
    };
    let query = match flag(args, "--query") {
        Some(q) => q,
        None => return err("--query <text> is required"),
    };
    let k = match parse_usize(args, "--k", 5) {
        Ok(v) if v > 0 => v,
        _ => return err("--k must be a positive integer"),
    };
    let ef = match parse_usize(args, "--ef", 50) {
        Ok(v) => v,
        Err(e) => return err(&e),
    };

    let loaded = match persistence::load(&index) {
        Ok(l) => l,
        Err(e) => return err(&format!("cannot load {}: {e}", index.display())),
    };
    let qv: Vector;
    let hits: Vec<(String, f32)> = match &loaded {
        persistence::LoadedIndex::Brute(b) => {
            qv = text::embed(&query, b.dim());
            b.search(&qv, k)
        }
        persistence::LoadedIndex::Hnsw(h) => {
            qv = text::embed(&query, h.dim());
            h.search(&qv, k, ef)
        }
    };
    if hits.is_empty() {
        println!("no results");
    }
    for (id, score) in hits {
        println!("{score:.4}\t{id}");
    }
    0
}

// ---- benchmark ------------------------------------------------------------

/// Tiny deterministic RNG (splitmix64) so benchmarks are reproducible.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / 9007199254740992.0
    }
    /// Approximately standard-normal via Box-Muller.
    fn gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn cmd_bench(args: &[String]) -> i32 {
    let n = match parse_usize(args, "--n", 5000) {
        Ok(v) if v > 0 => v,
        _ => return err("--n must be a positive integer"),
    };
    let dim = match parse_usize(args, "--dim", 64) {
        Ok(d) if d > 0 => d,
        _ => return err("--dim must be a positive integer"),
    };
    let k = match parse_usize(args, "--k", 10) {
        Ok(v) if v > 0 => v,
        _ => return err("--k must be a positive integer"),
    };

    // Synthetic clustered data: 8 gaussian clusters in `dim` dimensions.
    let mut rng = Rng(0xC0FF_EE42);
    let centers: Vec<Vec<f64>> = (0..8)
        .map(|_| (0..dim).map(|_| rng.gaussian() * 3.0).collect())
        .collect();
    let data: Vec<Vector> = (0..n)
        .map(|i| {
            let c = &centers[i % 8];
            Vector::new(
                c.iter()
                    .map(|x| (x + rng.gaussian() * 0.5) as f32)
                    .collect(),
            )
        })
        .collect();
    let queries: Vec<Vector> = (0..100)
        .map(|_| {
            let c = &centers[(rng.next_u64() % 8) as usize];
            Vector::new(
                c.iter()
                    .map(|x| (x + rng.gaussian() * 0.5) as f32)
                    .collect(),
            )
        })
        .collect();

    let t0 = Instant::now();
    let mut brute = BruteForceIndex::new(dim);
    for (i, v) in data.iter().enumerate() {
        brute.add(format!("d{i}"), v.clone());
    }
    let brute_build = t0.elapsed();

    let t0 = Instant::now();
    let mut hnsw = HnswIndex::with_seed(dim, 16, 200, 1234);
    for (i, v) in data.iter().enumerate() {
        hnsw.add(format!("d{i}"), v.clone());
    }
    let hnsw_build = t0.elapsed();

    // Ground truth from brute force.
    let truth: Vec<Vec<String>> = queries
        .iter()
        .map(|q| brute.search(q, k).into_iter().map(|(id, _)| id).collect())
        .collect();
    let t0 = Instant::now();
    for q in &queries {
        let _ = brute.search(q, k);
    }
    let brute_q = t0.elapsed() / queries.len() as u32;

    let mut hits = 0;
    let t0 = Instant::now();
    for (qi, q) in queries.iter().enumerate() {
        let approx: Vec<String> = hnsw
            .search(q, k, 100)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        hits += approx.iter().filter(|id| truth[qi].contains(id)).count();
    }
    let hnsw_q = t0.elapsed() / queries.len() as u32;
    let recall = hits as f64 / (queries.len() * k) as f64;

    println!(
        "vectors: {n}, dim: {dim}, k: {k}, queries: {}",
        queries.len()
    );
    println!(
        "build:    brute-force {:>8.0} ms | HNSW {:>8.0} ms",
        brute_build.as_secs_f64() * 1000.0,
        hnsw_build.as_secs_f64() * 1000.0
    );
    println!(
        "query:    brute-force {:>8.1} µs | HNSW {:>8.1} µs  ({:.0}x faster)",
        brute_q.as_secs_f64() * 1e6,
        hnsw_q.as_secs_f64() * 1e6,
        brute_q.as_secs_f64() / hnsw_q.as_secs_f64().max(1e-9)
    );
    println!("recall@{k}: {recall:.4}");
    0
}

fn err(msg: &str) -> i32 {
    eprintln!("error: {msg}");
    1
}
