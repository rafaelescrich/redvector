//! GPU vector index traits

use super::{DistanceMetric, GpuConfig, Result};

/// Search result from GPU index
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Document/vector ID
    pub id: u64,

    /// Distance or similarity score
    pub score: f32,
}

impl SearchResult {
    /// Create new search result
    pub fn new(id: u64, score: f32) -> Self {
        Self { id, score }
    }
}

/// Trait for GPU-accelerated vector indexes
pub trait GpuVectorIndex: Send + Sync {
    /// Add vectors to the index
    ///
    /// - `ids`: Vector IDs
    /// - `vectors`: Flattened vector data [n_vectors × dim]
    fn add(&mut self, ids: &[u64], vectors: &[f32]) -> Result<()>;

    /// Search for k nearest neighbors
    ///
    /// - `query`: Query vector [dim]
    /// - `k`: Number of results to return
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>>;

    /// Batch search for multiple queries
    ///
    /// - `queries`: Flattened query vectors [n_queries × dim]
    /// - `k`: Number of results per query
    fn batch_search(&self, queries: &[f32], k: usize) -> Result<Vec<Vec<SearchResult>>>;

    /// Get the dimension of vectors in this index
    fn dimension(&self) -> usize;

    /// Get the number of vectors in this index
    fn len(&self) -> usize;

    /// Check if index is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the distance metric
    fn metric(&self) -> DistanceMetric;

    /// Get GPU config
    fn config(&self) -> &GpuConfig;

    /// Remove vectors by ID
    fn remove(&mut self, ids: &[u64]) -> Result<usize>;

    /// Clear all vectors
    fn clear(&mut self) -> Result<()>;

    /// Flush pending operations to GPU
    fn flush(&mut self) -> Result<()>;

    /// Get memory usage in bytes
    fn memory_usage(&self) -> usize;
}

/// Trait for trainable GPU indexes (IVF, PQ, etc.)
pub trait TrainableGpuIndex: GpuVectorIndex {
    /// Train the index on representative data
    ///
    /// - `training_data`: Flattened training vectors [n_vectors × dim]
    fn train(&mut self, training_data: &[f32]) -> Result<()>;

    /// Check if index is trained
    fn is_trained(&self) -> bool;

    /// Get training parameters
    fn training_params(&self) -> TrainingParams;
}

/// Training parameters for trainable indexes
#[derive(Debug, Clone)]
pub struct TrainingParams {
    /// Number of clusters (for IVF)
    pub n_clusters: usize,

    /// Number of iterations for k-means
    pub n_iter: usize,

    /// Minimum training samples
    pub min_samples: usize,

    /// Random seed
    pub seed: u64,
}

impl Default for TrainingParams {
    fn default() -> Self {
        Self {
            n_clusters: 1024,
            n_iter: 25,
            min_samples: 1000,
            seed: 42,
        }
    }
}

/// Batch operations helper
pub struct BatchOps;

impl BatchOps {
    /// Split vectors into batches for GPU processing
    pub fn batch_vectors(vectors: &[f32], dim: usize, batch_size: usize) -> Vec<&[f32]> {
        let n_vectors = vectors.len() / dim;
        let n_batches = (n_vectors + batch_size - 1) / batch_size;

        (0..n_batches)
            .map(|i| {
                let start = i * batch_size * dim;
                let end = ((i + 1) * batch_size * dim).min(vectors.len());
                &vectors[start..end]
            })
            .collect()
    }

    /// Split IDs into batches
    pub fn batch_ids(ids: &[u64], batch_size: usize) -> Vec<&[u64]> {
        ids.chunks(batch_size).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_vectors() {
        let vectors: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let batches = BatchOps::batch_vectors(&vectors, 10, 3);

        assert_eq!(batches.len(), 4); // 10 vectors / 3 = 4 batches
        assert_eq!(batches[0].len(), 30); // 3 vectors × 10 dims
        assert_eq!(batches[3].len(), 10); // 1 vector × 10 dims (remainder)
    }
}

