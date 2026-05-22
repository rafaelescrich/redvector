//! SIMD-accelerated distance metrics
//!
//! Provides SIMD-optimized implementations of distance calculations
//! with automatic CPU feature detection and fallback to scalar.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// SIMD-accelerated cosine similarity
/// 
/// Uses AVX-512 when available, falls back to AVX2, SSE4.1, then scalar.
/// On ARM, uses NEON.
pub fn cosine_similarity_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { cosine_similarity_avx512(a, b) }
        } else if is_x86_feature_detected!("avx2") {
            unsafe { cosine_similarity_avx2(a, b) }
        } else if is_x86_feature_detected!("sse4.1") {
            unsafe { cosine_similarity_sse(a, b) }
        } else {
            cosine_similarity_scalar_fallback(a, b)
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { cosine_similarity_neon(a, b) }
    }
    
    #[cfg(all(not(target_arch = "x86_64"), not(target_arch = "aarch64")))]
    {
        cosine_similarity_scalar_fallback(a, b)
    }
}

// AVX-512 implementations (16 f32s at a time)

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn cosine_similarity_avx512(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 16;
    
    let mut dot = _mm512_setzero_ps();
    let mut norm_a_sq = _mm512_setzero_ps();
    let mut norm_b_sq = _mm512_setzero_ps();
    
    for i in 0..chunks {
        let idx = i * 16;
        let va = _mm512_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm512_loadu_ps(b.as_ptr().add(idx));
        
        dot = _mm512_fmadd_ps(va, vb, dot);
        norm_a_sq = _mm512_fmadd_ps(va, va, norm_a_sq);
        norm_b_sq = _mm512_fmadd_ps(vb, vb, norm_b_sq);
    }
    
    // Horizontal sum
    let dot_sum = _mm512_reduce_add_ps(dot);
    let norm_a_sq_sum = _mm512_reduce_add_ps(norm_a_sq);
    let norm_b_sq_sum = _mm512_reduce_add_ps(norm_b_sq);
    
    // Remainder
    let remainder_start = chunks * 16;
    let dot_rem: f32 = a[remainder_start..]
        .iter()
        .zip(&b[remainder_start..])
        .map(|(&x, &y)| x * y)
        .sum();
    
    let norm_a_rem_sq: f32 = a[remainder_start..].iter().map(|&x| x * x).sum();
    let norm_b_rem_sq: f32 = b[remainder_start..].iter().map(|&x| x * x).sum();
    
    let dot_total = dot_sum + dot_rem;
    let norm_a = (norm_a_sq_sum + norm_a_rem_sq).sqrt();
    let norm_b = (norm_b_sq_sum + norm_b_rem_sq).sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_total / (norm_a * norm_b)
    }
}

// NEON implementations (4 f32s at a time)

#[cfg(target_arch = "aarch64")]
unsafe fn cosine_similarity_neon(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    
    let mut dot = vdupq_n_f32(0.0);
    let mut norm_a_sq = vdupq_n_f32(0.0);
    let mut norm_b_sq = vdupq_n_f32(0.0);
    
    for i in 0..chunks {
        let idx = i * 4;
        let va = vld1q_f32(a.as_ptr().add(idx));
        let vb = vld1q_f32(b.as_ptr().add(idx));
        
        dot = vfmaq_f32(dot, va, vb);
        norm_a_sq = vfmaq_f32(norm_a_sq, va, va);
        norm_b_sq = vfmaq_f32(norm_b_sq, vb, vb);
    }
    
    // Horizontal sum
    let dot_sum = vaddvq_f32(dot);
    let norm_a_sq_sum = vaddvq_f32(norm_a_sq);
    let norm_b_sq_sum = vaddvq_f32(norm_b_sq);
    
    // Remainder
    let remainder_start = chunks * 4;
    let dot_rem: f32 = a[remainder_start..]
        .iter()
        .zip(&b[remainder_start..])
        .map(|(&x, &y)| x * y)
        .sum();
    
    let norm_a_rem_sq: f32 = a[remainder_start..].iter().map(|&x| x * x).sum();
    let norm_b_rem_sq: f32 = b[remainder_start..].iter().map(|&x| x * x).sum();
    
    let dot_total = dot_sum + dot_rem;
    let norm_a = (norm_a_sq_sum + norm_a_rem_sq).sqrt();
    let norm_b = (norm_b_sq_sum + norm_b_rem_sq).sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_total / (norm_a * norm_b)
    }
}

/// SIMD-accelerated Euclidean distance
pub fn euclidean_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { euclidean_distance_avx512(a, b) }
        } else if is_x86_feature_detected!("avx2") {
            unsafe { euclidean_distance_avx2(a, b) }
        } else if is_x86_feature_detected!("sse4.1") {
            unsafe { euclidean_distance_sse(a, b) }
        } else {
            euclidean_distance_scalar_fallback(a, b)
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { euclidean_distance_neon(a, b) }
    }
    
    #[cfg(all(not(target_arch = "x86_64"), not(target_arch = "aarch64")))]
    {
        euclidean_distance_scalar_fallback(a, b)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn euclidean_distance_avx512(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 16;
    
    let mut sum_sq = _mm512_setzero_ps();
    
    for i in 0..chunks {
        let idx = i * 16;
        let va = _mm512_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm512_loadu_ps(b.as_ptr().add(idx));
        let diff = _mm512_sub_ps(va, vb);
        sum_sq = _mm512_fmadd_ps(diff, diff, sum_sq);
    }
    
    let sum_sq_total = _mm512_reduce_add_ps(sum_sq);
    
    // Remainder
    let remainder_start = chunks * 16;
    let remainder_sum_sq: f32 = a[remainder_start..]
        .iter()
        .zip(&b[remainder_start..])
        .map(|(&x, &y)| (x - y).powi(2))
        .sum();
    
    (sum_sq_total + remainder_sum_sq).sqrt()
}

#[cfg(target_arch = "aarch64")]
unsafe fn euclidean_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    
    let mut sum_sq = vdupq_n_f32(0.0);
    
    for i in 0..chunks {
        let idx = i * 4;
        let va = vld1q_f32(a.as_ptr().add(idx));
        let vb = vld1q_f32(b.as_ptr().add(idx));
        let diff = vsubq_f32(va, vb);
        sum_sq = vfmaq_f32(sum_sq, diff, diff);
    }
    
    let sum_sq_total = vaddvq_f32(sum_sq);
    
    // Remainder
    let remainder_start = chunks * 4;
    let remainder_sum_sq: f32 = a[remainder_start..]
        .iter()
        .zip(&b[remainder_start..])
        .map(|(&x, &y)| (x - y).powi(2))
        .sum();
    
    (sum_sq_total + remainder_sum_sq).sqrt()
}

/// SIMD-accelerated inner product
pub fn inner_product_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { inner_product_avx512(a, b) }
        } else if is_x86_feature_detected!("avx2") {
            unsafe { inner_product_avx2(a, b) }
        } else if is_x86_feature_detected!("sse4.1") {
            unsafe { inner_product_sse(a, b) }
        } else {
            inner_product_scalar_fallback(a, b)
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { inner_product_neon(a, b) }
    }
    
    #[cfg(all(not(target_arch = "x86_64"), not(target_arch = "aarch64")))]
    {
        inner_product_scalar_fallback(a, b)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn inner_product_avx512(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 16;
    
    let mut dot = _mm512_setzero_ps();
    
    for i in 0..chunks {
        let idx = i * 16;
        let va = _mm512_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm512_loadu_ps(b.as_ptr().add(idx));
        dot = _mm512_fmadd_ps(va, vb, dot);
    }
    
    let dot_sum = _mm512_reduce_add_ps(dot);
    
    // Remainder
    let remainder_start = chunks * 16;
    let dot_rem: f32 = a[remainder_start..]
        .iter()
        .zip(&b[remainder_start..])
        .map(|(&x, &y)| x * y)
        .sum();
    
    dot_sum + dot_rem
}

#[cfg(target_arch = "aarch64")]
unsafe fn inner_product_neon(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    
    let mut dot = vdupq_n_f32(0.0);
    
    for i in 0..chunks {
        let idx = i * 4;
        let va = vld1q_f32(a.as_ptr().add(idx));
        let vb = vld1q_f32(b.as_ptr().add(idx));
        dot = vfmaq_f32(dot, va, vb);
    }
    
    let dot_sum = vaddvq_f32(dot);
    
    // Remainder
    let remainder_start = chunks * 4;
    let dot_rem: f32 = a[remainder_start..]
        .iter()
        .zip(&b[remainder_start..])
        .map(|(&x, &y)| x * y)
        .sum();
    
    dot_sum + dot_rem
}

// AVX2 implementations (8 f32s at a time)

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn cosine_similarity_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    
    let mut dot = _mm256_setzero_ps();
    let mut norm_a_sq = _mm256_setzero_ps();
    let mut norm_b_sq = _mm256_setzero_ps();
    
    for i in 0..chunks {
        let idx = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm256_loadu_ps(b.as_ptr().add(idx));
        
        dot = _mm256_fmadd_ps(va, vb, dot);
        norm_a_sq = _mm256_fmadd_ps(va, va, norm_a_sq);
        norm_b_sq = _mm256_fmadd_ps(vb, vb, norm_b_sq);
    }
    
    // Horizontal sum
    let dot_sum = horizontal_sum_avx(dot);
    let norm_a_sq_sum = horizontal_sum_avx(norm_a_sq);
    let norm_b_sq_sum = horizontal_sum_avx(norm_b_sq);
    
    // Remainder
    let remainder_start = chunks * 8;
    let dot_rem: f32 = a[remainder_start..]
        .iter()
        .zip(&b[remainder_start..])
        .map(|(&x, &y)| x * y)
        .sum();
    
    let norm_a_rem_sq: f32 = a[remainder_start..].iter().map(|&x| x * x).sum();
    let norm_b_rem_sq: f32 = b[remainder_start..].iter().map(|&x| x * x).sum();
    
    let dot_total = dot_sum + dot_rem;
    let norm_a = (norm_a_sq_sum + norm_a_rem_sq).sqrt();
    let norm_b = (norm_b_sq_sum + norm_b_rem_sq).sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_total / (norm_a * norm_b)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn euclidean_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    
    let mut sum_sq = _mm256_setzero_ps();
    
    for i in 0..chunks {
        let idx = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm256_loadu_ps(b.as_ptr().add(idx));
        let diff = _mm256_sub_ps(va, vb);
        sum_sq = _mm256_fmadd_ps(diff, diff, sum_sq);
    }
    
    let sum_sq_total = horizontal_sum_avx(sum_sq);
    
    // Remainder
    let remainder_start = chunks * 8;
    let remainder_sum_sq: f32 = a[remainder_start..]
        .iter()
        .zip(&b[remainder_start..])
        .map(|(&x, &y)| (x - y).powi(2))
        .sum();
    
    (sum_sq_total + remainder_sum_sq).sqrt()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn inner_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    
    let mut dot = _mm256_setzero_ps();
    
    for i in 0..chunks {
        let idx = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm256_loadu_ps(b.as_ptr().add(idx));
        dot = _mm256_fmadd_ps(va, vb, dot);
    }
    
    let dot_sum = horizontal_sum_avx(dot);
    
    // Remainder
    let remainder_start = chunks * 8;
    let dot_rem: f32 = a[remainder_start..]
        .iter()
        .zip(&b[remainder_start..])
        .map(|(&x, &y)| x * y)
        .sum();
    
    dot_sum + dot_rem
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_avx(v: __m256) -> f32 {
    // Extract high and low 128-bit lanes
    let high = _mm256_extractf128_ps(v, 1);
    let low = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(high, low);
    
    // Horizontal sum of 128-bit register
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf = _mm_movehl_ps(sums, sums);
    let sums = _mm_add_ss(sums, shuf);
    _mm_cvtss_f32(sums)
}

// SSE4.1 implementations (4 f32s at a time)

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn cosine_similarity_sse(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    
    let mut dot = _mm_setzero_ps();
    let mut norm_a_sq = _mm_setzero_ps();
    let mut norm_b_sq = _mm_setzero_ps();
    
    for i in 0..chunks {
        let idx = i * 4;
        let va = _mm_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm_loadu_ps(b.as_ptr().add(idx));
        
        dot = _mm_add_ps(_mm_mul_ps(va, vb), dot);
        norm_a_sq = _mm_add_ps(_mm_mul_ps(va, va), norm_a_sq);
        norm_b_sq = _mm_add_ps(_mm_mul_ps(vb, vb), norm_b_sq);
    }
    
    // Horizontal sum
    let dot_sum = horizontal_sum_sse(dot);
    let norm_a_sq_sum = horizontal_sum_sse(norm_a_sq);
    let norm_b_sq_sum = horizontal_sum_sse(norm_b_sq);
    
    // Remainder
    let remainder_start = chunks * 4;
    let dot_rem: f32 = a[remainder_start..]
        .iter()
        .zip(&b[remainder_start..])
        .map(|(&x, &y)| x * y)
        .sum();
    
    let norm_a_rem_sq: f32 = a[remainder_start..].iter().map(|&x| x * x).sum();
    let norm_b_rem_sq: f32 = b[remainder_start..].iter().map(|&x| x * x).sum();
    
    let dot_total = dot_sum + dot_rem;
    let norm_a = (norm_a_sq_sum + norm_a_rem_sq).sqrt();
    let norm_b = (norm_b_sq_sum + norm_b_rem_sq).sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_total / (norm_a * norm_b)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn euclidean_distance_sse(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    
    let mut sum_sq = _mm_setzero_ps();
    
    for i in 0..chunks {
        let idx = i * 4;
        let va = _mm_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm_loadu_ps(b.as_ptr().add(idx));
        let diff = _mm_sub_ps(va, vb);
        sum_sq = _mm_add_ps(_mm_mul_ps(diff, diff), sum_sq);
    }
    
    let sum_sq_total = horizontal_sum_sse(sum_sq);
    
    // Remainder
    let remainder_start = chunks * 4;
    let remainder_sum_sq: f32 = a[remainder_start..]
        .iter()
        .zip(&b[remainder_start..])
        .map(|(&x, &y)| (x - y).powi(2))
        .sum();
    
    (sum_sq_total + remainder_sum_sq).sqrt()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn inner_product_sse(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    
    let mut dot = _mm_setzero_ps();
    
    for i in 0..chunks {
        let idx = i * 4;
        let va = _mm_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm_loadu_ps(b.as_ptr().add(idx));
        dot = _mm_add_ps(_mm_mul_ps(va, vb), dot);
    }
    
    let dot_sum = horizontal_sum_sse(dot);
    
    // Remainder
    let remainder_start = chunks * 4;
    let dot_rem: f32 = a[remainder_start..]
        .iter()
        .zip(&b[remainder_start..])
        .map(|(&x, &y)| x * y)
        .sum();
    
    dot_sum + dot_rem
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn horizontal_sum_sse(v: __m128) -> f32 {
    let shuf = _mm_movehdup_ps(v);
    let sums = _mm_add_ps(v, shuf);
    let shuf = _mm_movehl_ps(sums, sums);
    let sums = _mm_add_ss(sums, shuf);
    _mm_cvtss_f32(sums)
}

// Scalar fallback implementations (already exist in vector_index.rs)
// These are kept here for reference but will use the ones from vector_index module

// Scalar fallback implementations (for testing and non-SIMD platforms)
pub fn cosine_similarity_scalar_fallback(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

pub fn euclidean_distance_scalar_fallback(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

pub fn inner_product_scalar_fallback(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let result = cosine_similarity_simd(&a, &b);
        assert!((result - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_simd_euclidean_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 1.0, 1.0];
        let result = euclidean_distance_simd(&a, &b);
        let expected = (3.0f32).sqrt();
        assert!((result - expected).abs() < 1e-6);
    }
    
    #[test]
    fn test_simd_inner_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = inner_product_simd(&a, &b);
        let expected = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0;
        assert!((result - expected).abs() < 1e-6);
    }
    
    #[test]
    fn test_simd_vs_scalar_consistency() {
        // Test that SIMD and scalar give same results
        let a: Vec<f32> = (0..384).map(|i| (i as f32) / 1000.0).collect();
        let b: Vec<f32> = (100..484).map(|i| (i as f32) / 1000.0).collect();
        
        let simd_cosine = cosine_similarity_simd(&a, &b);
        let scalar_cosine = cosine_similarity_scalar_fallback(&a, &b);
        assert!((simd_cosine - scalar_cosine).abs() < 1e-5, "SIMD cosine differs: {} vs {}", simd_cosine, scalar_cosine);
        
        let simd_euclidean = euclidean_distance_simd(&a, &b);
        let scalar_euclidean = euclidean_distance_scalar_fallback(&a, &b);
        assert!((simd_euclidean - scalar_euclidean).abs() < 1e-5, "SIMD euclidean differs: {} vs {}", simd_euclidean, scalar_euclidean);
        
        let simd_inner = inner_product_simd(&a, &b);
        let scalar_inner = inner_product_scalar_fallback(&a, &b);
        assert!((simd_inner - scalar_inner).abs() < 1e-5, "SIMD inner product differs: {} vs {}", simd_inner, scalar_inner);
    }
}

