//! Binary persistence for indexes.
//!
//! Versioned little-endian format (`VSE1` magic, format revisions in the
//! 4th magic byte):
//!
//! ```text
//! magic:      b'V' b'S' b'E' <version: 0x01 or 0x02>
//! kind:       u8   (0 = brute force, 1 = HNSW)
//! dim:        u32
//! -- brute force -----------------------------------------------
//! count:      u32
//! per entry:  id_len u32, id bytes, dim × f32
//! -- HNSW ------------------------------------------------------
//! m, ef_construction, max_level: u32 × 3
//! entry_point: i32  (-1 = none)
//! rng_state:  u64   (splitmix64 stream, so post-load inserts continue it)
//! count:      u32
//! per node:   id_len u32, id bytes, level u32, [deleted u8, VSE2+], dim × f32
//! layers:     for lc in 0..=max_level:
//!                 edge_count u32
//!                 per edge: node u32, degree u32, degree × u32 neighbor ids
//! ```
//!
//! Version 0x02 adds the per-node tombstone flag (`deleted`); the loader
//! accepts both revisions (0x01 files load with all flags clear).
//!
//! The HNSW graph is persisted verbatim (not rebuilt), so a loaded index
//! answers queries identically to the one that was saved.

use crate::hnsw::{HnswIndex, Node};
use crate::index::BruteForceIndex;
use crate::vector::Vector;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: [u8; 3] = *b"VSE";
const VERSION: u8 = 0x02;
const KIND_BRUTE: u8 = 0;
const KIND_HNSW: u8 = 1;

/// An index read back from disk.
pub enum LoadedIndex {
    Brute(BruteForceIndex),
    Hnsw(HnswIndex),
}

struct Writer<W: Write>(W);

impl<W: Write> Writer<W> {
    fn u8(&mut self, v: u8) -> io::Result<()> {
        self.0.write_all(&[v])
    }
    fn u64(&mut self, v: u64) -> io::Result<()> {
        self.0.write_all(&v.to_le_bytes())
    }
    fn u32(&mut self, v: u32) -> io::Result<()> {
        self.0.write_all(&v.to_le_bytes())
    }
    fn i32(&mut self, v: i32) -> io::Result<()> {
        self.0.write_all(&v.to_le_bytes())
    }
    fn f32(&mut self, v: f32) -> io::Result<()> {
        self.0.write_all(&v.to_le_bytes())
    }
    fn bytes(&mut self, b: &[u8]) -> io::Result<()> {
        self.u32(b.len() as u32)?;
        self.0.write_all(b)
    }
    fn vector(&mut self, v: &Vector) -> io::Result<()> {
        for x in v.dims() {
            self.f32(*x)?;
        }
        Ok(())
    }
}

struct Reader<R: Read>(R);

impl<R: Read> Reader<R> {
    fn read_exact<const N: usize>(&mut self) -> io::Result<[u8; N]> {
        let mut buf = [0u8; N];
        self.0.read_exact(&mut buf)?;
        Ok(buf)
    }
    fn u8(&mut self) -> io::Result<u8> {
        Ok(self.read_exact::<1>()?[0])
    }
    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.read_exact::<4>()?))
    }
    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.read_exact::<8>()?))
    }
    fn i32(&mut self) -> io::Result<i32> {
        Ok(i32::from_le_bytes(self.read_exact::<4>()?))
    }
    fn f32(&mut self) -> io::Result<f32> {
        Ok(f32::from_le_bytes(self.read_exact::<4>()?))
    }
    fn bytes(&mut self) -> io::Result<Vec<u8>> {
        let len = self.u32()? as usize;
        let mut buf = vec![0u8; len];
        self.0.read_exact(&mut buf)?;
        Ok(buf)
    }
    fn vector(&mut self, dim: usize) -> io::Result<Vector> {
        let mut dims = Vec::with_capacity(dim);
        for _ in 0..dim {
            dims.push(self.f32()?);
        }
        Ok(Vector::new(dims))
    }
    fn id(&mut self) -> io::Result<String> {
        let b = self.bytes()?;
        String::from_utf8(b).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}

fn bad(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

impl BruteForceIndex {
    /// Save the index to `path` in the `VSE1` binary format.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut w = Writer(BufWriter::new(File::create(path)?));
        w.0.write_all(&MAGIC)?;
        w.u8(VERSION)?;
        w.u8(KIND_BRUTE)?;
        w.u32(self.dim() as u32)?;
        let entries: Vec<(&str, &Vector)> = self.entries().collect();
        w.u32(entries.len() as u32)?;
        for (id, v) in entries {
            w.bytes(id.as_bytes())?;
            w.vector(v)?;
        }
        Ok(())
    }

    /// Load a brute-force index previously written by [`BruteForceIndex::save`].
    pub fn load(path: &Path) -> io::Result<Self> {
        match load(path)? {
            LoadedIndex::Brute(idx) => Ok(idx),
            LoadedIndex::Hnsw(_) => Err(bad("file holds an HNSW index, not brute force")),
        }
    }
}

impl HnswIndex {
    /// Save the index (vectors *and* graph) to `path`.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut w = Writer(BufWriter::new(File::create(path)?));
        w.0.write_all(&MAGIC)?;
        w.u8(VERSION)?;
        w.u8(KIND_HNSW)?;
        w.u32(self.dim() as u32)?;
        w.u32(self.m() as u32)?;
        w.u32(self.ef_construction() as u32)?;
        w.u32(self.max_level() as u32)?;
        w.i32(self.entry_point().map(|e| e as i32).unwrap_or(-1))?;
        w.u64(self.rng_state())?;

        let nodes = self.raw_nodes();
        w.u32(nodes.len() as u32)?;
        for n in nodes {
            w.bytes(n.id.as_bytes())?;
            w.u32(n.level as u32)?;
            w.u8(n.deleted as u8)?;
            w.vector(&n.vector)?;
        }
        for layer in self.raw_layers() {
            w.u32(layer.len() as u32)?;
            for (node, neigh) in layer {
                w.u32(*node as u32)?;
                w.u32(neigh.len() as u32)?;
                for nb in neigh {
                    w.u32(*nb as u32)?;
                }
            }
        }
        Ok(())
    }

    /// Load an HNSW index previously written by [`HnswIndex::save`].
    pub fn load(path: &Path) -> io::Result<Self> {
        match load(path)? {
            LoadedIndex::Hnsw(idx) => Ok(idx),
            LoadedIndex::Brute(_) => Err(bad("file holds a brute-force index, not HNSW")),
        }
    }
}

/// Load whatever index kind `path` holds.
pub fn load(path: &Path) -> io::Result<LoadedIndex> {
    let mut r = Reader(BufReader::new(File::open(path)?));
    let magic = r.read_exact::<3>()?;
    let version = r.u8()?;
    if magic != MAGIC || (version != 0x01 && version != VERSION) {
        return Err(bad("not a VSE1 index file (bad magic or version)"));
    }
    let kind = r.u8()?;
    let dim = r.u32()? as usize;
    if dim == 0 {
        return Err(bad("zero dimension"));
    }

    match kind {
        KIND_BRUTE => {
            let count = r.u32()? as usize;
            let mut ids = Vec::with_capacity(count);
            let mut vectors = Vec::with_capacity(count);
            for _ in 0..count {
                ids.push(r.id()?);
                vectors.push(r.vector(dim)?);
            }
            Ok(LoadedIndex::Brute(BruteForceIndex::from_parts(
                dim, ids, vectors,
            )))
        }
        KIND_HNSW => {
            let m = r.u32()? as usize;
            let ef_construction = r.u32()? as usize;
            let max_level = r.u32()? as usize;
            let entry_raw = r.i32()?;
            let entry_point = if entry_raw < 0 {
                None
            } else {
                Some(entry_raw as usize)
            };
            let rng = r.u64()?;
            let count = r.u32()? as usize;
            let mut nodes = Vec::with_capacity(count);
            for _ in 0..count {
                let id = r.id()?;
                let level = r.u32()? as usize;
                let deleted = if version == VERSION {
                    r.u8()? != 0
                } else {
                    false
                };
                let vector = r.vector(dim)?;
                nodes.push(Node {
                    id,
                    vector,
                    level,
                    deleted,
                });
            }
            let mut layers: Vec<HashMap<usize, Vec<usize>>> = Vec::with_capacity(max_level + 1);
            for _ in 0..=max_level {
                let edge_count = r.u32()? as usize;
                let mut layer = HashMap::with_capacity(edge_count);
                for _ in 0..edge_count {
                    let node = r.u32()? as usize;
                    let degree = r.u32()? as usize;
                    let mut neigh = Vec::with_capacity(degree);
                    for _ in 0..degree {
                        neigh.push(r.u32()? as usize);
                    }
                    if node >= count || neigh.iter().any(|&nb| nb >= count) {
                        return Err(bad("node id out of range"));
                    }
                    layer.insert(node, neigh);
                }
                layers.push(layer);
            }
            // The RNG stream continues from the saved state, so vectors added
            // after loading draw levels exactly as if the index had never
            // been serialized.
            Ok(LoadedIndex::Hnsw(HnswIndex::from_parts(
                dim,
                m,
                ef_construction,
                max_level,
                entry_point,
                nodes,
                layers,
                rng,
            )))
        }
        _ => Err(bad("unknown index kind")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn tmp(name: &str) -> std::path::PathBuf {
        env::temp_dir().join(format!("vse-test-{name}"))
    }

    fn sample_brute() -> BruteForceIndex {
        let mut idx = BruteForceIndex::new(3);
        idx.add("a", Vector::new(vec![1.0, 0.0, 0.0]));
        idx.add("b", Vector::new(vec![0.0, 1.0, 0.0]));
        idx.add("c", Vector::new(vec![0.0, 0.0, 1.0]));
        idx
    }

    fn sample_hnsw() -> HnswIndex {
        let mut idx = HnswIndex::with_seed(3, 8, 50, 99);
        idx.add("a", Vector::new(vec![1.0, 0.0, 0.0]));
        idx.add("b", Vector::new(vec![0.0, 1.0, 0.0]));
        idx.add("c", Vector::new(vec![0.0, 0.0, 1.0]));
        idx.add("d", Vector::new(vec![0.7, 0.7, 0.0]));
        idx
    }

    #[test]
    fn brute_force_roundtrip() {
        let idx = sample_brute();
        let p = tmp("brute.vse");
        idx.save(&p).unwrap();
        let back = BruteForceIndex::load(&p).unwrap();
        let q = Vector::new(vec![0.9, 0.1, 0.0]);
        assert_eq!(idx.search(&q, 2), back.search(&q, 2));
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn hnsw_roundtrip_preserves_search_results() {
        let idx = sample_hnsw();
        let p = tmp("hnsw.vse");
        idx.save(&p).unwrap();
        let back = HnswIndex::load(&p).unwrap();
        assert_eq!(back.len(), idx.len());
        let q = Vector::new(vec![0.9, 0.1, 0.0]);
        assert_eq!(idx.search(&q, 3, 20), back.search(&q, 3, 20));
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn hnsw_stays_queryable_after_adding_post_load() {
        let idx = sample_hnsw();
        let p = tmp("hnsw-grow.vse");
        idx.save(&p).unwrap();
        let mut back = HnswIndex::load(&p).unwrap();
        back.add("e", Vector::new(vec![0.1, 0.9, 0.0]));
        let hits = back.search(&Vector::new(vec![0.0, 1.0, 0.0]), 2, 20);
        assert_eq!(hits[0].0, "b");
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn loading_wrong_kind_fails() {
        let p = tmp("wrong-kind.vse");
        sample_brute().save(&p).unwrap();
        assert!(HnswIndex::load(&p).is_err());
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn loading_garbage_fails() {
        let p = tmp("garbage.vse");
        std::fs::write(&p, b"definitely not an index").unwrap();
        assert!(load(&p).is_err());
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn hnsw_tombstones_survive_roundtrip() {
        let mut idx = sample_hnsw();
        assert!(idx.remove("b"));
        let p = tmp("hnsw-tomb.vse");
        idx.save(&p).unwrap();
        let back = HnswIndex::load(&p).unwrap();
        assert_eq!(back.len(), 3);
        assert_eq!(back.tombstones(), 1);
        assert!(!back.contains("b"));
        let hits = back.search(&Vector::new(vec![0.0, 1.0, 0.0]), 5, 20);
        assert!(hits.iter().all(|(id, _)| id != "b"));
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn legacy_v1_file_loads_with_clear_tombstones() {
        // Hand-write a 0x01-format HNSW file (no per-node deleted flag).
        let p = tmp("legacy-v1.vse");
        {
            let mut buf: Vec<u8> = Vec::new();
            buf.extend_from_slice(b"VSE");
            buf.push(0x01);
            buf.push(1u8); // KIND_HNSW
            buf.extend_from_slice(&3u32.to_le_bytes()); // dim
            buf.extend_from_slice(&8u32.to_le_bytes()); // m
            buf.extend_from_slice(&50u32.to_le_bytes()); // ef_construction
            buf.extend_from_slice(&0u32.to_le_bytes()); // max_level
            buf.extend_from_slice(&(-1i32).to_le_bytes()); // no entry
            buf.extend_from_slice(&0u64.to_le_bytes()); // rng
            buf.extend_from_slice(&1u32.to_le_bytes()); // 1 node
            buf.extend_from_slice(&1u32.to_le_bytes()); // id len
            buf.push(b'x'); // id
            buf.extend_from_slice(&0u32.to_le_bytes()); // level
            for x in [1.0f32, 0.0, 0.0] {
                buf.extend_from_slice(&x.to_le_bytes());
            }
            buf.extend_from_slice(&0u32.to_le_bytes()); // layer 0: no edges
            std::fs::write(&p, buf).unwrap();
        }
        let back = HnswIndex::load(&p).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back.tombstones(), 0);
        assert!(back.contains("x"));
        std::fs::remove_file(p).ok();
    }
}
