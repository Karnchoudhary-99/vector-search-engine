//! Distance / similarity metrics between [`Vector`](crate::Vector)s.

use crate::vector::Vector;

fn check_dims(a: &Vector, b: &Vector) {
    assert_eq!(a.len(), b.len(), "vectors must share dimensions");
}

/// Euclidean (L2) distance. 0.0 = identical.
pub fn euclidean(a: &Vector, b: &Vector) -> f32 {
    check_dims(a, b);
    a.dims()
        .iter()
        .zip(b.dims().iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Cosine similarity in [-1, 1]. 1.0 = same direction.
/// Zero vectors yield 0.0 instead of NaN.
pub fn cosine_similarity(a: &Vector, b: &Vector) -> f32 {
    check_dims(a, b);
    let denom = a.norm() * b.norm();
    if denom == 0.0 {
        return 0.0;
    }
    a.dot(b) / denom
}

/// Cosine distance in [0, 2]. 0.0 = same direction.
pub fn cosine_distance(a: &Vector, b: &Vector) -> f32 {
    1.0 - cosine_similarity(a, b)
}

/// Manhattan (L1) distance. 0.0 = identical.
pub fn manhattan(a: &Vector, b: &Vector) -> f32 {
    check_dims(a, b);
    a.dims()
        .iter()
        .zip(b.dims().iter())
        .map(|(x, y)| (x - y).abs())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_have_zero_distance() {
        let a = Vector::new(vec![1.0, 2.0]);
        assert_eq!(euclidean(&a, &a), 0.0);
        assert_eq!(manhattan(&a, &a), 0.0);
        assert_eq!(cosine_distance(&a, &a), 0.0);
    }

    #[test]
    fn euclidean_3_4_5_triangle() {
        let a = Vector::new(vec![0.0, 0.0]);
        let b = Vector::new(vec![3.0, 4.0]);
        assert!((euclidean(&a, &b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_of_orthogonal_is_zero() {
        let a = Vector::new(vec![1.0, 0.0]);
        let b = Vector::new(vec![0.0, 1.0]);
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }
}
