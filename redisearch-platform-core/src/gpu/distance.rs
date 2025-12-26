//! Distance metrics for GPU operations

use serde::{Deserialize, Serialize};

/// Distance/similarity metric
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// L2 (Euclidean) distance - smaller is more similar
    L2,

    /// Cosine similarity - larger is more similar
    Cosine,

    /// Inner product (dot product) - larger is more similar
    InnerProduct,
}

impl Default for DistanceMetric {
    fn default() -> Self {
        Self::Cosine
    }
}

impl DistanceMetric {
    /// Whether larger scores mean more similar
    pub fn is_similarity(&self) -> bool {
        matches!(self, Self::Cosine | Self::InnerProduct)
    }

    /// Whether smaller scores mean more similar
    pub fn is_distance(&self) -> bool {
        matches!(self, Self::L2)
    }

    /// Compare two scores according to this metric's semantics
    /// Returns true if score_a is "better" than score_b
    pub fn is_better(&self, score_a: f32, score_b: f32) -> bool {
        if self.is_similarity() {
            score_a > score_b
        } else {
            score_a < score_b
        }
    }

    /// Get the worst possible score for this metric
    pub fn worst_score(&self) -> f32 {
        if self.is_similarity() {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        }
    }

    /// CPU scalar implementation for reference/fallback
    pub fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Self::L2 => Self::l2_distance(a, b),
            Self::Cosine => Self::cosine_similarity(a, b),
            Self::InnerProduct => Self::inner_product(a, b),
        }
    }

    /// L2 (Euclidean) distance
    #[inline]
    pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// L2 squared distance (avoids sqrt, good for comparison)
    #[inline]
    pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum()
    }

    /// Cosine similarity
    #[inline]
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Inner product (dot product)
    #[inline]
    pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }

    /// Normalize vector to unit length
    pub fn normalize(v: &mut [f32]) {
        let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
    }

    /// Check if vector is normalized
    pub fn is_normalized(v: &[f32], tolerance: f32) -> bool {
        let norm_sq: f32 = v.iter().map(|&x| x * x).sum();
        (norm_sq - 1.0).abs() < tolerance
    }
}

/// WGSL shader code for distance computations
pub mod shaders {
    /// L2 squared distance shader
    pub const L2_SQUARED_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> database: array<f32>;
@group(0) @binding(2) var<storage, read_write> distances: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    dim: u32,
    n_vectors: u32,
    query_offset: u32,
    _pad: u32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vec_idx = global_id.x;
    if (vec_idx >= params.n_vectors) {
        return;
    }
    
    let db_offset = vec_idx * params.dim;
    var sum_sq: f32 = 0.0;
    
    for (var d: u32 = 0u; d < params.dim; d = d + 1u) {
        let diff = query[params.query_offset + d] - database[db_offset + d];
        sum_sq = sum_sq + diff * diff;
    }
    
    distances[vec_idx] = sum_sq;
}
"#;

    /// Cosine similarity shader
    pub const COSINE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> database: array<f32>;
@group(0) @binding(2) var<storage, read_write> similarities: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    dim: u32,
    n_vectors: u32,
    query_offset: u32,
    _pad: u32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vec_idx = global_id.x;
    if (vec_idx >= params.n_vectors) {
        return;
    }
    
    let db_offset = vec_idx * params.dim;
    var dot: f32 = 0.0;
    var norm_a_sq: f32 = 0.0;
    var norm_b_sq: f32 = 0.0;
    
    for (var d: u32 = 0u; d < params.dim; d = d + 1u) {
        let a = query[params.query_offset + d];
        let b = database[db_offset + d];
        dot = dot + a * b;
        norm_a_sq = norm_a_sq + a * a;
        norm_b_sq = norm_b_sq + b * b;
    }
    
    let norm_a = sqrt(norm_a_sq);
    let norm_b = sqrt(norm_b_sq);
    
    if (norm_a > 0.0 && norm_b > 0.0) {
        similarities[vec_idx] = dot / (norm_a * norm_b);
    } else {
        similarities[vec_idx] = 0.0;
    }
}
"#;

    /// Inner product shader
    pub const INNER_PRODUCT_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> database: array<f32>;
@group(0) @binding(2) var<storage, read_write> products: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    dim: u32,
    n_vectors: u32,
    query_offset: u32,
    _pad: u32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vec_idx = global_id.x;
    if (vec_idx >= params.n_vectors) {
        return;
    }
    
    let db_offset = vec_idx * params.dim;
    var dot: f32 = 0.0;
    
    for (var d: u32 = 0u; d < params.dim; d = d + 1u) {
        dot = dot + query[params.query_offset + d] * database[db_offset + d];
    }
    
    products[vec_idx] = dot;
}
"#;

    /// Top-K selection shader (partial sort)
    pub const TOPK_WGSL: &str = r#"
// Bitonic sort for top-k selection
// This is a simplified version; production would use more efficient algorithms

@group(0) @binding(0) var<storage, read_write> scores: array<f32>;
@group(0) @binding(1) var<storage, read_write> indices: array<u32>;
@group(0) @binding(2) var<uniform> params: TopKParams;

struct TopKParams {
    n: u32,
    k: u32,
    stage: u32,
    is_similarity: u32,  // 1 if larger is better
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.n / 2u) {
        return;
    }
    
    // Bitonic compare-swap
    let block_size = 1u << params.stage;
    let half_block = block_size >> 1u;
    
    let block_idx = idx / half_block;
    let local_idx = idx % half_block;
    
    let i = block_idx * block_size + local_idx;
    let j = i + half_block;
    
    if (j < params.n) {
        var should_swap: bool;
        if (params.is_similarity == 1u) {
            should_swap = scores[i] < scores[j];
        } else {
            should_swap = scores[i] > scores[j];
        }
        
        if (should_swap) {
            let temp_score = scores[i];
            scores[i] = scores[j];
            scores[j] = temp_score;
            
            let temp_idx = indices[i];
            indices[i] = indices[j];
            indices[j] = temp_idx;
        }
    }
}
"#;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 1.0, 1.0];
        let dist = DistanceMetric::l2_distance(&a, &b);
        assert!((dist - 3.0f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        let sim = DistanceMetric::cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0];
        let sim_orthogonal = DistanceMetric::cosine_similarity(&a, &c);
        assert!(sim_orthogonal.abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let mut v = vec![3.0, 4.0];
        DistanceMetric::normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
        assert!(DistanceMetric::is_normalized(&v, 1e-5));
    }
}

