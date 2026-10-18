//! Core dense-vector type with the linear-algebra ops search needs.

/// A dense embedding vector (f32 components).
#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    dims: Vec<f32>,
}

impl Vector {
    /// Build a vector from its components. Empty vectors are rejected.
    pub fn new(dims: Vec<f32>) -> Self {
        assert!(!dims.is_empty(), "vector must have at least one dimension");
        Self { dims }
    }

    /// Borrow the raw components.
    pub fn dims(&self) -> &[f32] {
        &self.dims
    }

    /// Number of dimensions.
    pub fn len(&self) -> usize {
        self.dims.len()
    }

    /// True when the vector has no dimensions (never for [`Vector::new`]).
    pub fn is_empty(&self) -> bool {
        self.dims.is_empty()
    }

    /// Dot product. Panics if dimensions differ.
    pub fn dot(&self, other: &Vector) -> f32 {
        assert_eq!(
            self.len(),
            other.len(),
            "dot product needs equal dimensions"
        );
        self.dims
            .iter()
            .zip(other.dims.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Euclidean (L2) norm.
    pub fn norm(&self) -> f32 {
        self.dot(self).sqrt()
    }

    /// Unit vector in the same direction. Zero vectors map to themselves.
    pub fn normalized(&self) -> Vector {
        let n = self.norm();
        if n == 0.0 {
            return self.clone();
        }
        Vector::new(self.dims.iter().map(|x| x / n).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_product_is_correct() {
        let a = Vector::new(vec![1.0, 2.0, 3.0]);
        let b = Vector::new(vec![4.0, 5.0, 6.0]);
        assert_eq!(a.dot(&b), 32.0);
    }

    #[test]
    fn norm_of_unit_axes_is_one() {
        let a = Vector::new(vec![1.0, 0.0, 0.0]);
        assert!((a.norm() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalized_has_unit_norm() {
        let a = Vector::new(vec![3.0, 4.0]);
        assert!((a.normalized().norm() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn zero_vector_normalizes_to_itself() {
        let z = Vector::new(vec![0.0, 0.0]);
        assert_eq!(z.normalized(), z);
    }
}
