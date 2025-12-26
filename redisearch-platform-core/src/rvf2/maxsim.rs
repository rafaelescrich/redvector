//! MaxSim: Late-interaction scoring for multi-vector retrieval
//!
//! Computes: MaxSim(Q, D) = Σ max(q · d) for each query token q, over all doc patches d
//!
//! ## SIMD Optimization
//!
//! This module provides SIMD-accelerated implementations for:
//! - x86_64: AVX2 + FMA
//! - aarch64: NEON (Apple Silicon)
//!
//! ## Usage
//!
//! ```rust,ignore
//! let scorer = SimdMaxSimScorer::new(128);
//! let score = scorer.compute(&query_tokens, &doc_patches);
//! ```

use crate::rvf2::Result;

/// MaxSim scorer for multi-vector similarity (scalar fallback)
pub struct MaxSimScorer {
    /// Dimension of vectors
    pub dims: usize,
}

impl MaxSimScorer {
    /// Create new scorer
    pub fn new(dims: usize) -> Self {
        Self { dims }
    }

    /// Compute MaxSim between query tokens and document patches
    ///
    /// - `query_tokens`: [T × D] flattened array of query token vectors
    /// - `doc_patches`: [P × D] flattened array of document patch vectors
    ///
    /// Returns: sum of max similarities for each query token
    pub fn compute(&self, query_tokens: &[f32], doc_patches: &[f32]) -> f32 {
        let n_tokens = query_tokens.len() / self.dims;
        let n_patches = doc_patches.len() / self.dims;

        let mut total_score = 0.0f32;

        for t in 0..n_tokens {
            let q_start = t * self.dims;
            let q = &query_tokens[q_start..q_start + self.dims];

            let mut max_sim = f32::NEG_INFINITY;

            for p in 0..n_patches {
                let p_start = p * self.dims;
                let patch = &doc_patches[p_start..p_start + self.dims];

                let sim = Self::dot_product_scalar(q, patch);
                max_sim = max_sim.max(sim);
            }

            total_score += max_sim;
        }

        total_score
    }

    /// Compute MaxSim with SQ8-encoded patches (faster, avoids full decode)
    pub fn compute_sq8(
        &self,
        query_tokens: &[f32],
        patch_codes: &[i8],
        scales: &[half::f16],
        block_size: usize,
    ) -> f32 {
        let n_tokens = query_tokens.len() / self.dims;
        let n_patches = patch_codes.len() / self.dims;

        let mut total_score = 0.0f32;

        for t in 0..n_tokens {
            let q_start = t * self.dims;
            let q = &query_tokens[q_start..q_start + self.dims];

            let mut max_sim = f32::NEG_INFINITY;

            for p in 0..n_patches {
                let block_idx = p / block_size;
                let scale = scales.get(block_idx).map(|s| s.to_f32()).unwrap_or(1.0) / 127.0;

                let p_start = p * self.dims;
                let patch = &patch_codes[p_start..p_start + self.dims];

                let sim = Self::dot_product_f32_i8_scalar(q, patch) * scale;
                max_sim = max_sim.max(sim);
            }

            total_score += max_sim;
        }

        total_score
    }

    /// Batch MaxSim for multiple candidates
    pub fn compute_batch(
        &self,
        query_tokens: &[f32],
        candidates: &[&[f32]],
    ) -> Vec<f32> {
        candidates
            .iter()
            .map(|patches| self.compute(query_tokens, patches))
            .collect()
    }

    /// Scalar dot product
    #[inline]
    fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Dot product: f32 × i8 -> f32 (scalar)
    #[inline]
    fn dot_product_f32_i8_scalar(a: &[f32], b: &[i8]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| x * y as f32)
            .sum()
    }
}

// ============================================================================
// SIMD Implementations
// ============================================================================

/// SIMD-optimized MaxSim scorer
///
/// Automatically selects the best implementation for the current platform:
/// - AVX2 + FMA on x86_64
/// - NEON on aarch64 (Apple Silicon)
/// - Scalar fallback otherwise
pub struct SimdMaxSimScorer {
    dims: usize,
}

impl SimdMaxSimScorer {
    /// Create new SIMD scorer
    pub fn new(dims: usize) -> Self {
        Self { dims }
    }

    /// Get dimension
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Compute MaxSim using best available SIMD
    pub fn compute(&self, query_tokens: &[f32], doc_patches: &[f32]) -> f32 {
        let n_tokens = query_tokens.len() / self.dims;
        let n_patches = doc_patches.len() / self.dims;

        let mut total_score = 0.0f32;

        for t in 0..n_tokens {
            let q_start = t * self.dims;
            let q = &query_tokens[q_start..q_start + self.dims];

            let mut max_sim = f32::NEG_INFINITY;

            for p in 0..n_patches {
                let p_start = p * self.dims;
                let patch = &doc_patches[p_start..p_start + self.dims];

                let sim = self.dot_product(q, patch);
                max_sim = max_sim.max(sim);
            }

            total_score += max_sim;
        }

        total_score
    }

    /// Compute MaxSim with SQ8-encoded patches
    pub fn compute_sq8(
        &self,
        query_tokens: &[f32],
        patch_codes: &[i8],
        scales: &[half::f16],
        block_size: usize,
    ) -> f32 {
        let n_tokens = query_tokens.len() / self.dims;
        let n_patches = patch_codes.len() / self.dims;

        let mut total_score = 0.0f32;

        for t in 0..n_tokens {
            let q_start = t * self.dims;
            let q = &query_tokens[q_start..q_start + self.dims];

            let mut max_sim = f32::NEG_INFINITY;

            for p in 0..n_patches {
                let block_idx = p / block_size;
                let scale = scales.get(block_idx).map(|s| s.to_f32()).unwrap_or(1.0) / 127.0;

                let p_start = p * self.dims;
                let patch = &patch_codes[p_start..p_start + self.dims];

                let sim = self.dot_product_i8(q, patch) * scale;
                max_sim = max_sim.max(sim);
            }

            total_score += max_sim;
        }

        total_score
    }

    /// Batch MaxSim for multiple candidates (parallel-friendly)
    pub fn compute_batch(&self, query_tokens: &[f32], candidates: &[&[f32]]) -> Vec<f32> {
        candidates
            .iter()
            .map(|patches| self.compute(query_tokens, patches))
            .collect()
    }

    /// SIMD-accelerated dot product
    #[inline]
    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { dot_f32_avx2_fma(a, b) };
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            return unsafe { dot_f32_neon(a, b) };
        }

        // Scalar fallback
        #[allow(unreachable_code)]
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// SIMD-accelerated dot product for i8 codes
    #[inline]
    fn dot_product_i8(&self, a: &[f32], b: &[i8]) -> f32 {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { dot_f32_i8_avx2(a, b) };
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            return unsafe { dot_f32_i8_neon(a, b) };
        }

        // Scalar fallback
        #[allow(unreachable_code)]
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y as f32).sum()
    }
}

// ============================================================================
// AVX2 + FMA Implementation (x86_64)
// ============================================================================

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX2 + FMA dot product for f32 vectors
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_f32_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;

    let mut sum = _mm256_setzero_ps();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    // Horizontal sum of 256-bit register
    let sum128 = _mm_add_ps(
        _mm256_extractf128_ps(sum, 0),
        _mm256_extractf128_ps(sum, 1),
    );
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32);

    // Handle remainder
    for i in (chunks * 8)..len {
        result += a[i] * b[i];
    }

    result
}

/// AVX2 dot product for f32 × i8
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_f32_i8_avx2(a: &[f32], b: &[i8]) -> f32 {
    let len = a.len();
    let chunks = len / 8;

    let mut sum = _mm256_setzero_ps();

    for i in 0..chunks {
        let offset = i * 8;

        // Load 8 f32 values
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));

        // Load 8 i8 values and convert to f32
        // First load as i64, then unpack
        let vb_i8_ptr = b.as_ptr().add(offset) as *const i64;
        let vb_i8 = _mm_set_epi64x(0, *vb_i8_ptr);

        // Sign-extend i8 to i16
        let vb_i16 = _mm256_cvtepi8_epi16(vb_i8);

        // Convert lower 8 i16 to i32, then to f32
        let vb_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vb_i16));
        let vb_f32 = _mm256_cvtepi32_ps(vb_i32);

        // Multiply and accumulate
        sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb_f32));
    }

    // Horizontal sum
    let sum128 = _mm_add_ps(
        _mm256_extractf128_ps(sum, 0),
        _mm256_extractf128_ps(sum, 1),
    );
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32);

    // Handle remainder
    for i in (chunks * 8)..len {
        result += a[i] * b[i] as f32;
    }

    result
}

// ============================================================================
// NEON Implementation (aarch64 / Apple Silicon)
// ============================================================================

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON dot product for f32 vectors
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;

    let mut sum = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a.as_ptr().add(offset));
        let vb = vld1q_f32(b.as_ptr().add(offset));
        sum = vfmaq_f32(sum, va, vb);
    }

    let mut result = vaddvq_f32(sum);

    // Handle remainder
    for i in (chunks * 4)..len {
        result += a[i] * b[i];
    }

    result
}

/// NEON dot product for f32 × i8
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_f32_i8_neon(a: &[f32], b: &[i8]) -> f32 {
    let len = a.len();
    let chunks = len / 4;

    let mut sum = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let offset = i * 4;

        let va = vld1q_f32(a.as_ptr().add(offset));

        // Load 4 i8, widen to i16, then to i32, then to f32
        let vb_i8 = vld1_s8(b.as_ptr().add(offset));
        let vb_i16 = vmovl_s8(vb_i8);
        let vb_i32 = vmovl_s16(vget_low_s16(vb_i16));
        let vb_f32 = vcvtq_f32_s32(vb_i32);

        sum = vfmaq_f32(sum, va, vb_f32);
    }

    let mut result = vaddvq_f32(sum);

    // Handle remainder
    for i in (chunks * 4)..len {
        result += a[i] * b[i] as f32;
    }

    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxsim_basic() {
        let scorer = MaxSimScorer::new(4);

        // 2 query tokens × 4 dims
        let query = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];

        // 3 patches × 4 dims
        let patches = vec![
            0.5, 0.0, 0.0, 0.0, // patch 0: sim with q0 = 0.5
            0.0, 0.8, 0.0, 0.0, // patch 1: sim with q1 = 0.8
            0.9, 0.1, 0.0, 0.0, // patch 2: sim with q0 = 0.9, q1 = 0.1
        ];

        let score = scorer.compute(&query, &patches);
        // q0 max = 0.9 (patch 2), q1 max = 0.8 (patch 1)
        assert!((score - 1.7).abs() < 0.001);
    }

    #[test]
    fn test_simd_maxsim() {
        let scorer = SimdMaxSimScorer::new(128);

        // Generate random-ish test data
        let query: Vec<f32> = (0..256).map(|i| (i as f32 / 100.0).sin()).collect(); // 2 tokens
        let patches: Vec<f32> = (0..1280).map(|i| (i as f32 / 100.0).cos()).collect(); // 10 patches

        let score = scorer.compute(&query, &patches);
        assert!(score.is_finite());
    }

    #[test]
    fn test_simd_vs_scalar() {
        let dims = 128;
        let scalar_scorer = MaxSimScorer::new(dims);
        let simd_scorer = SimdMaxSimScorer::new(dims);

        let query: Vec<f32> = (0..dims * 2).map(|i| (i as f32 / 50.0).sin()).collect();
        let patches: Vec<f32> = (0..dims * 10).map(|i| (i as f32 / 50.0).cos()).collect();

        let scalar_score = scalar_scorer.compute(&query, &patches);
        let simd_score = simd_scorer.compute(&query, &patches);

        assert!(
            (scalar_score - simd_score).abs() < 0.01,
            "Scalar {} vs SIMD {}",
            scalar_score,
            simd_score
        );
    }

    #[test]
    fn test_batch_maxsim() {
        let scorer = MaxSimScorer::new(4);
        let query = vec![1.0, 0.0, 0.0, 0.0];

        let doc1 = vec![0.5, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0];
        let doc2 = vec![0.9, 0.0, 0.0, 0.0];

        let scores = scorer.compute_batch(&query, &[&doc1, &doc2]);
        assert!((scores[0] - 0.5).abs() < 0.001);
        assert!((scores[1] - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_sq8_maxsim() {
        let scorer = MaxSimScorer::new(4);

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let codes: Vec<i8> = vec![127, 0, 0, 0, 64, 0, 0, 0]; // 2 patches
        let scales = vec![half::f16::from_f32(1.0)]; // 1 block

        let score = scorer.compute_sq8(&query, &codes, &scales, 32);
        assert!(score > 0.0);
    }
}
