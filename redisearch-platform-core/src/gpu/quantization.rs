//! Vector quantization for memory-efficient storage
//!
//! Implements SQ8 (scalar quantization) and PQ (product quantization).

use serde::{Deserialize, Serialize};

/// Quantizer type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizerType {
    /// No quantization (float32)
    None,

    /// Scalar quantization to int8 (4x compression)
    SQ8,

    /// Scalar quantization to int4 (8x compression)
    SQ4,

    /// Product quantization
    PQ {
        /// Number of subvectors
        m: usize,
        /// Bits per subvector (typically 8)
        nbits: usize,
    },
}

impl Default for QuantizerType {
    fn default() -> Self {
        Self::None
    }
}

/// SQ8 (Scalar Quantization to int8) quantizer
#[derive(Debug, Clone)]
pub struct Sq8Quantizer {
    /// Vector dimension
    dim: usize,

    /// Scale per dimension (for dequantization)
    scales: Vec<f32>,

    /// Offset per dimension
    offsets: Vec<f32>,

    /// Whether quantizer is trained
    trained: bool,
}

impl Sq8Quantizer {
    /// Create new SQ8 quantizer
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            scales: vec![1.0; dim],
            offsets: vec![0.0; dim],
            trained: false,
        }
    }

    /// Train quantizer on data
    pub fn train(&mut self, vectors: &[f32]) {
        let n_vectors = vectors.len() / self.dim;
        if n_vectors == 0 {
            return;
        }

        // Find min/max per dimension
        let mut mins = vec![f32::INFINITY; self.dim];
        let mut maxs = vec![f32::NEG_INFINITY; self.dim];

        for i in 0..n_vectors {
            for d in 0..self.dim {
                let v = vectors[i * self.dim + d];
                mins[d] = mins[d].min(v);
                maxs[d] = maxs[d].max(v);
            }
        }

        // Compute scales and offsets
        for d in 0..self.dim {
            let range = maxs[d] - mins[d];
            if range > 0.0 {
                self.scales[d] = range / 254.0; // Map to [-127, 127]
                self.offsets[d] = mins[d] + range / 2.0; // Center at 0
            } else {
                self.scales[d] = 1.0;
                self.offsets[d] = mins[d];
            }
        }

        self.trained = true;
    }

    /// Quantize vectors to int8
    pub fn encode(&self, vectors: &[f32]) -> Vec<i8> {
        let n_vectors = vectors.len() / self.dim;
        let mut codes = Vec::with_capacity(vectors.len());

        for i in 0..n_vectors {
            for d in 0..self.dim {
                let v = vectors[i * self.dim + d];
                let centered = v - self.offsets[d];
                let scaled = centered / self.scales[d];
                let quantized = scaled.round().clamp(-127.0, 127.0) as i8;
                codes.push(quantized);
            }
        }

        codes
    }

    /// Dequantize int8 codes to float32
    pub fn decode(&self, codes: &[i8]) -> Vec<f32> {
        let n_vectors = codes.len() / self.dim;
        let mut vectors = Vec::with_capacity(codes.len());

        for i in 0..n_vectors {
            for d in 0..self.dim {
                let code = codes[i * self.dim + d];
                let scaled = code as f32 * self.scales[d];
                let v = scaled + self.offsets[d];
                vectors.push(v);
            }
        }

        vectors
    }

    /// Compute approximate L2 squared distance between query and quantized vectors
    pub fn compute_l2_squared(&self, query: &[f32], codes: &[i8]) -> f32 {
        let mut sum_sq = 0.0f32;

        for d in 0..self.dim {
            let code = codes[d];
            let decoded = code as f32 * self.scales[d] + self.offsets[d];
            let diff = query[d] - decoded;
            sum_sq += diff * diff;
        }

        sum_sq
    }

    /// Compute approximate inner product
    pub fn compute_inner_product(&self, query: &[f32], codes: &[i8]) -> f32 {
        let mut dot = 0.0f32;

        for d in 0..self.dim {
            let code = codes[d];
            let decoded = code as f32 * self.scales[d] + self.offsets[d];
            dot += query[d] * decoded;
        }

        dot
    }

    /// Get dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Check if trained
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Bytes per vector
    pub fn bytes_per_vector(&self) -> usize {
        self.dim // 1 byte per dimension
    }
}

/// Product Quantization (PQ) quantizer
#[derive(Debug, Clone)]
pub struct PqQuantizer {
    /// Vector dimension
    dim: usize,

    /// Number of subvectors
    m: usize,

    /// Dimension per subvector
    dsub: usize,

    /// Number of centroids per subvector (2^nbits)
    k: usize,

    /// Codebooks: [m][k][dsub]
    codebooks: Vec<Vec<Vec<f32>>>,

    /// Whether trained
    trained: bool,
}

impl PqQuantizer {
    /// Create new PQ quantizer
    ///
    /// - `dim`: Vector dimension (must be divisible by m)
    /// - `m`: Number of subvectors (typically 8, 16, 32)
    /// - `nbits`: Bits per subvector (typically 8 for k=256 centroids)
    pub fn new(dim: usize, m: usize, nbits: usize) -> Self {
        assert!(dim % m == 0, "Dimension must be divisible by m");
        let dsub = dim / m;
        let k = 1 << nbits;

        Self {
            dim,
            m,
            dsub,
            k,
            codebooks: vec![vec![vec![0.0; dsub]; k]; m],
            trained: false,
        }
    }

    /// Train PQ codebooks using k-means
    pub fn train(&mut self, vectors: &[f32], n_iter: usize) {
        let n_vectors = vectors.len() / self.dim;
        if n_vectors < self.k {
            panic!("Need at least {} training vectors", self.k);
        }

        // Train each subquantizer independently
        for sub in 0..self.m {
            let sub_start = sub * self.dsub;

            // Extract subvectors
            let mut subvectors: Vec<f32> = Vec::with_capacity(n_vectors * self.dsub);
            for i in 0..n_vectors {
                let vec_start = i * self.dim + sub_start;
                subvectors.extend_from_slice(&vectors[vec_start..vec_start + self.dsub]);
            }

            // Run k-means
            let centroids = kmeans(&subvectors, self.dsub, self.k, n_iter);
            self.codebooks[sub] = centroids;
        }

        self.trained = true;
    }

    /// Encode vectors to PQ codes
    pub fn encode(&self, vectors: &[f32]) -> Vec<u8> {
        let n_vectors = vectors.len() / self.dim;
        let mut codes = Vec::with_capacity(n_vectors * self.m);

        for i in 0..n_vectors {
            for sub in 0..self.m {
                let sub_start = i * self.dim + sub * self.dsub;
                let subvec = &vectors[sub_start..sub_start + self.dsub];

                // Find nearest centroid
                let mut best_dist = f32::INFINITY;
                let mut best_idx = 0u8;

                for (j, centroid) in self.codebooks[sub].iter().enumerate() {
                    let dist: f32 = subvec
                        .iter()
                        .zip(centroid.iter())
                        .map(|(&a, &b)| (a - b).powi(2))
                        .sum();

                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = j as u8;
                    }
                }

                codes.push(best_idx);
            }
        }

        codes
    }

    /// Decode PQ codes to approximate vectors
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let n_vectors = codes.len() / self.m;
        let mut vectors = Vec::with_capacity(n_vectors * self.dim);

        for i in 0..n_vectors {
            for sub in 0..self.m {
                let code = codes[i * self.m + sub] as usize;
                vectors.extend_from_slice(&self.codebooks[sub][code]);
            }
        }

        vectors
    }

    /// Compute distance table for a query (for fast ADC)
    pub fn compute_distance_table(&self, query: &[f32]) -> Vec<Vec<f32>> {
        let mut table = vec![vec![0.0f32; self.k]; self.m];

        for sub in 0..self.m {
            let sub_start = sub * self.dsub;
            let subquery = &query[sub_start..sub_start + self.dsub];

            for (j, centroid) in self.codebooks[sub].iter().enumerate() {
                let dist: f32 = subquery
                    .iter()
                    .zip(centroid.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum();
                table[sub][j] = dist;
            }
        }

        table
    }

    /// Compute asymmetric distance using precomputed table (ADC)
    pub fn compute_distance_adc(&self, table: &[Vec<f32>], codes: &[u8]) -> f32 {
        let mut dist = 0.0f32;

        for sub in 0..self.m {
            let code = codes[sub] as usize;
            dist += table[sub][code];
        }

        dist
    }

    /// Bytes per vector
    pub fn bytes_per_vector(&self) -> usize {
        self.m // 1 byte per subvector (for k=256)
    }

    /// Get dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Check if trained
    pub fn is_trained(&self) -> bool {
        self.trained
    }
}

/// Simple k-means clustering
fn kmeans(data: &[f32], dim: usize, k: usize, n_iter: usize) -> Vec<Vec<f32>> {
    let n = data.len() / dim;

    // Initialize centroids randomly (using first k points)
    let mut centroids: Vec<Vec<f32>> = (0..k)
        .map(|i| {
            let idx = (i * n / k) % n;
            data[idx * dim..(idx + 1) * dim].to_vec()
        })
        .collect();

    let mut assignments = vec![0usize; n];

    for _ in 0..n_iter {
        // Assign points to nearest centroid
        for i in 0..n {
            let point = &data[i * dim..(i + 1) * dim];
            let mut best_dist = f32::INFINITY;
            let mut best_idx = 0;

            for (j, centroid) in centroids.iter().enumerate() {
                let dist: f32 = point
                    .iter()
                    .zip(centroid.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum();

                if dist < best_dist {
                    best_dist = dist;
                    best_idx = j;
                }
            }

            assignments[i] = best_idx;
        }

        // Update centroids
        let mut counts = vec![0usize; k];
        let mut sums = vec![vec![0.0f32; dim]; k];

        for i in 0..n {
            let cluster = assignments[i];
            counts[cluster] += 1;
            for d in 0..dim {
                sums[cluster][d] += data[i * dim + d];
            }
        }

        for j in 0..k {
            if counts[j] > 0 {
                for d in 0..dim {
                    centroids[j][d] = sums[j][d] / counts[j] as f32;
                }
            }
        }
    }

    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sq8_quantizer() {
        let mut quantizer = Sq8Quantizer::new(4);

        // Training data
        let training = vec![
            0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0,
            0.5, 0.5, 0.5, 0.5,
        ];

        quantizer.train(&training);
        assert!(quantizer.is_trained());

        // Encode and decode
        let test_vec = vec![0.25, 0.75, 0.5, 0.0];
        let codes = quantizer.encode(&test_vec);
        let decoded = quantizer.decode(&codes);

        // Check approximate reconstruction
        for i in 0..4 {
            assert!(
                (test_vec[i] - decoded[i]).abs() < 0.1,
                "dim {}: {} vs {}",
                i,
                test_vec[i],
                decoded[i]
            );
        }
    }

    #[test]
    fn test_pq_quantizer() {
        let mut quantizer = PqQuantizer::new(8, 2, 8); // 8 dims, 2 subvectors, 256 centroids

        // Generate training data
        let n_train = 1000;
        let mut training = Vec::with_capacity(n_train * 8);
        for i in 0..n_train {
            for d in 0..8 {
                training.push((i as f32 / n_train as f32) + (d as f32 / 8.0));
            }
        }

        quantizer.train(&training, 10);
        assert!(quantizer.is_trained());

        // Encode and decode
        let test_vec = vec![0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4];
        let codes = quantizer.encode(&test_vec);
        assert_eq!(codes.len(), 2); // 2 subvectors

        let decoded = quantizer.decode(&codes);
        assert_eq!(decoded.len(), 8);
    }
}

