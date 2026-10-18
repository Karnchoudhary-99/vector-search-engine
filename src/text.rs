//! Deterministic demo-grade text embeddings.
//!
//! [`embed`] maps text to a unit vector via hashed character n-grams (FNV-1a
//! with a sign bit). This is **not** a neural embedding — it exists so the
//! CLI, examples, and benchmarks can run end-to-end semantic-ish search with
//! zero model dependencies and zero downloads.
//!
//! For production use, swap [`embed`] for real embeddings (e.g. from an
//! ONNX or API model) — the index types only care about [`Vector`]s.

use crate::vector::Vector;

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// Embed `text` into a `dim`-dimensional unit vector.
///
/// Lowercased character uni/bi/trigrams are hashed into `dim` buckets; the
/// hash's low bit decides the sign, so unrelated texts scatter while texts
/// sharing n-grams land closer together. Deterministic: the same text always
/// yields the same vector.
///
/// Panics if `dim == 0`.
pub fn embed(text: &str, dim: usize) -> Vector {
    assert!(dim > 0, "dimension must be positive");
    let mut acc = vec![0f32; dim];
    let chars: Vec<char> = text.to_lowercase().chars().collect();
    let mut buf = [0u8; 12]; // up to 3 chars × 4 bytes
    for start in 0..chars.len() {
        for n in 1..=3usize {
            if start + n > chars.len() {
                break;
            }
            let mut len = 0;
            for c in &chars[start..start + n] {
                let s = c.encode_utf8(&mut buf[len..]);
                len += s.len();
            }
            let h = fnv1a(&buf[..len]);
            let idx = (h as usize) % dim;
            let sign = if h & 1 == 1 { 1.0 } else { -1.0 };
            acc[idx] += sign;
        }
    }
    Vector::new(acc).normalized()
}

/// Cosine similarity between the embeddings of two texts.
pub fn similarity(a: &str, b: &str, dim: usize) -> f32 {
    embed(a, dim).dot(&embed(b, dim))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_is_deterministic() {
        assert_eq!(embed("hello world", 64), embed("hello world", 64));
    }

    #[test]
    fn embedding_is_unit_norm() {
        let e = embed("the quick brown fox", 64);
        assert!((e.norm() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn similar_texts_score_higher_than_unrelated() {
        let dim = 128;
        let sim_close = similarity("fast vector database", "quick vector search engine", dim);
        let sim_far = similarity("fast vector database", "chocolate cake recipe", dim);
        assert!(
            sim_close > sim_far,
            "close={sim_close:.3} should beat far={sim_far:.3}"
        );
    }

    #[test]
    fn identical_texts_have_similarity_one() {
        let s = similarity("repeat after me", "repeat after me", 64);
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn empty_text_embeds_to_zero_vector() {
        let e = embed("", 32);
        assert!(e.dims().iter().all(|&x| x == 0.0));
    }
}
