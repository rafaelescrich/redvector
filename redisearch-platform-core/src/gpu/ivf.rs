//! IVF (Inverted File) Index for scalable approximate nearest neighbor search
//!
//! IVF partitions vectors into clusters and only searches relevant clusters.
//! Combines with SQ8/PQ for memory efficiency.

use std::collections::HashMap;
use std::sync::RwLock;

use super::{
    DistanceMetric, GpuConfig, GpuError, GpuVectorIndex, Result, SearchResult,
    quantization::{Sq8Quantizer, PqQuantizer, QuantizerType},
    traits::TrainableGpuIndex,
};

/// IVF index configuration
#[derive(Debug, Clone)]
pub struct IvfConfig {
    /// Number of clusters (nlist)
    pub n_clusters: usize,

    /// Number of clusters to search (nprobe)
    pub n_probe: usize,

    /// Quantizer type for vectors
    pub quantizer: QuantizerType,

    /// Training iterations for k-means
    pub training_iterations: usize,
}

impl Default for IvfConfig {
    fn default() -> Self {
        Self {
            n_clusters: 1024,
            n_probe: 32,
            quantizer: QuantizerType::SQ8,
            training_iterations: 25,
        }
    }
}

impl IvfConfig {
    /// Create IVF-Flat config (no quantization)
    pub fn flat(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            n_probe: n_clusters.min(64),
            quantizer: QuantizerType::None,
            training_iterations: 25,
        }
    }

    /// Create IVF-SQ8 config
    pub fn sq8(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            n_probe: n_clusters.min(32),
            quantizer: QuantizerType::SQ8,
            training_iterations: 25,
        }
    }

    /// Create IVF-PQ config
    pub fn pq(n_clusters: usize, m: usize) -> Self {
        Self {
            n_clusters,
            n_probe: n_clusters.min(32),
            quantizer: QuantizerType::PQ { m, nbits: 8 },
            training_iterations: 25,
        }
    }

    /// Set nprobe
    pub fn with_nprobe(mut self, n_probe: usize) -> Self {
        self.n_probe = n_probe;
        self
    }
}

/// Inverted list (one per cluster)
struct InvertedList {
    /// Vector IDs in this cluster
    ids: Vec<u64>,

    /// Encoded vectors (format depends on quantizer)
    codes: Vec<u8>,

    /// Original float vectors (for IVF-Flat only)
    vectors: Option<Vec<f32>>,
}

impl InvertedList {
    fn new(use_float: bool) -> Self {
        Self {
            ids: Vec::new(),
            codes: Vec::new(),
            vectors: if use_float { Some(Vec::new()) } else { None },
        }
    }

    fn len(&self) -> usize {
        self.ids.len()
    }

    fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// GPU IVF Index
pub struct GpuIvfIndex {
    /// Vector dimension
    dim: usize,

    /// Distance metric
    metric: DistanceMetric,

    /// GPU configuration
    gpu_config: GpuConfig,

    /// IVF configuration
    ivf_config: IvfConfig,

    /// Cluster centroids [n_clusters × dim]
    centroids: Vec<f32>,

    /// Inverted lists (one per cluster)
    inverted_lists: Vec<RwLock<InvertedList>>,

    /// SQ8 quantizer (if using SQ8)
    sq8_quantizer: Option<Sq8Quantizer>,

    /// PQ quantizer (if using PQ)
    pq_quantizer: Option<PqQuantizer>,

    /// Whether index is trained
    trained: bool,

    /// Total vector count
    total_vectors: RwLock<usize>,
}

impl GpuIvfIndex {
    /// Create new IVF index
    pub fn new(
        dim: usize,
        metric: DistanceMetric,
        gpu_config: GpuConfig,
        ivf_config: IvfConfig,
    ) -> Result<Self> {
        let use_float = ivf_config.quantizer == QuantizerType::None;

        let inverted_lists: Vec<_> = (0..ivf_config.n_clusters)
            .map(|_| RwLock::new(InvertedList::new(use_float)))
            .collect();

        let sq8_quantizer = match ivf_config.quantizer {
            QuantizerType::SQ8 => Some(Sq8Quantizer::new(dim)),
            _ => None,
        };

        let pq_quantizer = match ivf_config.quantizer {
            QuantizerType::PQ { m, nbits } => Some(PqQuantizer::new(dim, m, nbits)),
            _ => None,
        };

        Ok(Self {
            dim,
            metric,
            gpu_config,
            ivf_config,
            centroids: Vec::new(),
            inverted_lists,
            sq8_quantizer,
            pq_quantizer,
            trained: false,
            total_vectors: RwLock::new(0),
        })
    }

    /// Find nearest cluster for a vector
    fn find_nearest_cluster(&self, vector: &[f32]) -> usize {
        let mut best_dist = f32::INFINITY;
        let mut best_idx = 0;

        for i in 0..self.ivf_config.n_clusters {
            let centroid = &self.centroids[i * self.dim..(i + 1) * self.dim];
            let dist = self.metric.compute(vector, centroid);

            // For similarity metrics, we want maximum; for distance, minimum
            let is_better = if self.metric.is_similarity() {
                dist > best_dist || best_dist == f32::INFINITY
            } else {
                dist < best_dist
            };

            if is_better {
                best_dist = dist;
                best_idx = i;
            }
        }

        best_idx
    }

    /// Find k nearest clusters for a query
    fn find_nearest_clusters(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut cluster_dists: Vec<(usize, f32)> = (0..self.ivf_config.n_clusters)
            .map(|i| {
                let centroid = &self.centroids[i * self.dim..(i + 1) * self.dim];
                (i, self.metric.compute(query, centroid))
            })
            .collect();

        // Sort by distance/similarity
        if self.metric.is_similarity() {
            cluster_dists.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        } else {
            cluster_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        }

        cluster_dists.truncate(k);
        cluster_dists
    }

    /// Search within a single cluster
    fn search_cluster(
        &self,
        cluster_idx: usize,
        query: &[f32],
        k: usize,
    ) -> Vec<SearchResult> {
        let list = self.inverted_lists[cluster_idx].read().unwrap();

        if list.is_empty() {
            return Vec::new();
        }

        let scores: Vec<f32> = match &self.ivf_config.quantizer {
            QuantizerType::None => {
                // IVF-Flat: compute exact distances
                let vectors = list.vectors.as_ref().unwrap();
                (0..list.len())
                    .map(|i| {
                        let vec = &vectors[i * self.dim..(i + 1) * self.dim];
                        self.metric.compute(query, vec)
                    })
                    .collect()
            }
            QuantizerType::SQ8 => {
                // IVF-SQ8: compute approximate distances
                let quantizer = self.sq8_quantizer.as_ref().unwrap();
                (0..list.len())
                    .map(|i| {
                        let codes = &list.codes[i * self.dim..(i + 1) * self.dim];
                        let codes_i8: Vec<i8> = codes.iter().map(|&c| c as i8).collect();

                        match self.metric {
                            DistanceMetric::L2 => quantizer.compute_l2_squared(query, &codes_i8).sqrt(),
                            DistanceMetric::InnerProduct => quantizer.compute_inner_product(query, &codes_i8),
                            DistanceMetric::Cosine => {
                                // Approximate cosine using inner product on normalized vectors
                                quantizer.compute_inner_product(query, &codes_i8)
                            }
                        }
                    })
                    .collect()
            }
            QuantizerType::PQ { m, .. } => {
                // IVF-PQ: compute approximate distances using ADC
                let quantizer = self.pq_quantizer.as_ref().unwrap();
                let dist_table = quantizer.compute_distance_table(query);

                (0..list.len())
                    .map(|i| {
                        let codes = &list.codes[i * m..(i + 1) * m];
                        quantizer.compute_distance_adc(&dist_table, codes)
                    })
                    .collect()
            }
            _ => Vec::new(),
        };

        // Build results
        let mut results: Vec<SearchResult> = scores
            .into_iter()
            .enumerate()
            .map(|(i, score)| SearchResult::new(list.ids[i], score))
            .collect();

        // Sort
        if self.metric.is_similarity() {
            results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        } else {
            results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        }

        results.truncate(k);
        results
    }
}

impl GpuVectorIndex for GpuIvfIndex {
    fn add(&mut self, ids: &[u64], vectors: &[f32]) -> Result<()> {
        if !self.trained {
            return Err(GpuError::NotTrained);
        }

        let n_vectors = vectors.len() / self.dim;
        if ids.len() != n_vectors {
            return Err(GpuError::DimensionMismatch {
                expected: n_vectors,
                got: ids.len(),
            });
        }

        for i in 0..n_vectors {
            let vector = &vectors[i * self.dim..(i + 1) * self.dim];
            let cluster_idx = self.find_nearest_cluster(vector);

            let mut list = self.inverted_lists[cluster_idx].write().unwrap();
            list.ids.push(ids[i]);

            match &self.ivf_config.quantizer {
                QuantizerType::None => {
                    list.vectors.as_mut().unwrap().extend_from_slice(vector);
                }
                QuantizerType::SQ8 => {
                    let quantizer = self.sq8_quantizer.as_ref().unwrap();
                    let codes = quantizer.encode(vector);
                    list.codes.extend(codes.iter().map(|&c| c as u8));
                }
                QuantizerType::PQ { .. } => {
                    let quantizer = self.pq_quantizer.as_ref().unwrap();
                    let codes = quantizer.encode(vector);
                    list.codes.extend(codes);
                }
                _ => {}
            }
        }

        *self.total_vectors.write().unwrap() += n_vectors;
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if !self.trained {
            return Err(GpuError::NotTrained);
        }

        if query.len() != self.dim {
            return Err(GpuError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }

        // Find nearest clusters
        let clusters = self.find_nearest_clusters(query, self.ivf_config.n_probe);

        // Search each cluster and merge results
        let mut all_results: Vec<SearchResult> = clusters
            .iter()
            .flat_map(|(cluster_idx, _)| self.search_cluster(*cluster_idx, query, k))
            .collect();

        // Final sort and truncate
        if self.metric.is_similarity() {
            all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        } else {
            all_results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        }

        all_results.truncate(k);
        Ok(all_results)
    }

    fn batch_search(&self, queries: &[f32], k: usize) -> Result<Vec<Vec<SearchResult>>> {
        let n_queries = queries.len() / self.dim;
        let mut results = Vec::with_capacity(n_queries);

        for i in 0..n_queries {
            let query = &queries[i * self.dim..(i + 1) * self.dim];
            results.push(self.search(query, k)?);
        }

        Ok(results)
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn len(&self) -> usize {
        *self.total_vectors.read().unwrap()
    }

    fn metric(&self) -> DistanceMetric {
        self.metric
    }

    fn config(&self) -> &GpuConfig {
        &self.gpu_config
    }

    fn remove(&mut self, ids: &[u64]) -> Result<usize> {
        // IVF removal is expensive; would need to search all lists
        // For now, return 0 (not implemented)
        Ok(0)
    }

    fn clear(&mut self) -> Result<()> {
        for list in &self.inverted_lists {
            let mut l = list.write().unwrap();
            l.ids.clear();
            l.codes.clear();
            if let Some(ref mut v) = l.vectors {
                v.clear();
            }
        }
        *self.total_vectors.write().unwrap() = 0;
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }

    fn memory_usage(&self) -> usize {
        let mut total = self.centroids.len() * std::mem::size_of::<f32>();

        for list in &self.inverted_lists {
            let l = list.read().unwrap();
            total += l.ids.len() * std::mem::size_of::<u64>();
            total += l.codes.len();
            if let Some(ref v) = l.vectors {
                total += v.len() * std::mem::size_of::<f32>();
            }
        }

        total
    }
}

impl TrainableGpuIndex for GpuIvfIndex {
    fn train(&mut self, training_data: &[f32]) -> Result<()> {
        let n_vectors = training_data.len() / self.dim;

        if n_vectors < self.ivf_config.n_clusters {
            return Err(GpuError::ComputeError(format!(
                "Need at least {} training vectors, got {}",
                self.ivf_config.n_clusters, n_vectors
            )));
        }

        // Train centroids using k-means
        self.centroids = kmeans_train(
            training_data,
            self.dim,
            self.ivf_config.n_clusters,
            self.ivf_config.training_iterations,
        );

        // Train quantizer on residuals (difference from centroid)
        match &self.ivf_config.quantizer {
            QuantizerType::SQ8 => {
                if let Some(ref mut quantizer) = self.sq8_quantizer {
                    quantizer.train(training_data);
                }
            }
            QuantizerType::PQ { .. } => {
                if let Some(ref mut quantizer) = self.pq_quantizer {
                    quantizer.train(training_data, self.ivf_config.training_iterations);
                }
            }
            _ => {}
        }

        self.trained = true;
        Ok(())
    }

    fn is_trained(&self) -> bool {
        self.trained
    }

    fn training_params(&self) -> super::traits::TrainingParams {
        super::traits::TrainingParams {
            n_clusters: self.ivf_config.n_clusters,
            n_iter: self.ivf_config.training_iterations,
            min_samples: self.ivf_config.n_clusters,
            seed: 42,
        }
    }
}

/// K-means training for centroids
fn kmeans_train(data: &[f32], dim: usize, k: usize, n_iter: usize) -> Vec<f32> {
    let n = data.len() / dim;

    // Initialize centroids (k-means++ would be better)
    let mut centroids: Vec<f32> = (0..k)
        .flat_map(|i| {
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

            for j in 0..k {
                let centroid = &centroids[j * dim..(j + 1) * dim];
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
        let mut sums = vec![0.0f32; k * dim];

        for i in 0..n {
            let cluster = assignments[i];
            counts[cluster] += 1;
            for d in 0..dim {
                sums[cluster * dim + d] += data[i * dim + d];
            }
        }

        for j in 0..k {
            if counts[j] > 0 {
                for d in 0..dim {
                    centroids[j * dim + d] = sums[j * dim + d] / counts[j] as f32;
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
    fn test_ivf_flat() {
        let gpu_config = GpuConfig::cpu();
        let ivf_config = IvfConfig::flat(4);

        let mut index = GpuIvfIndex::new(4, DistanceMetric::L2, gpu_config, ivf_config).unwrap();

        // Generate training data
        let training: Vec<f32> = (0..100)
            .flat_map(|i| {
                vec![
                    (i as f32 / 10.0).sin(),
                    (i as f32 / 10.0).cos(),
                    (i as f32 / 20.0).sin(),
                    (i as f32 / 20.0).cos(),
                ]
            })
            .collect();

        // Train
        index.train(&training).unwrap();
        assert!(index.is_trained());

        // Add vectors
        let ids: Vec<u64> = (0..50).collect();
        let vectors = training[..200].to_vec();
        index.add(&ids, &vectors).unwrap();

        assert_eq!(index.len(), 50);

        // Search
        let query = vec![0.0, 1.0, 0.0, 1.0];
        let results = index.search(&query, 5).unwrap();

        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_ivf_sq8() {
        let gpu_config = GpuConfig::cpu();
        let ivf_config = IvfConfig::sq8(4);

        let mut index = GpuIvfIndex::new(4, DistanceMetric::L2, gpu_config, ivf_config).unwrap();

        let training: Vec<f32> = (0..100)
            .flat_map(|i| vec![i as f32 / 100.0; 4])
            .collect();

        index.train(&training).unwrap();

        let ids: Vec<u64> = (0..50).collect();
        index.add(&ids, &training[..200]).unwrap();

        let query = vec![0.5, 0.5, 0.5, 0.5];
        let results = index.search(&query, 5).unwrap();

        assert!(!results.is_empty());
    }
}

