//! Product Quantization (PQ) for vector compression
//!
//! This module provides a pure-Rust implementation of Product Quantization,
//! allowing high-dimensional vectors to be compressed into small codes
//! for memory-efficient search.

use anyhow::Result;
use rand::seq::SliceRandom;
use rand::thread_rng;
use crate::simd_metrics::euclidean_distance_simd;

/// PQ configuration and codebooks
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct ProductQuantizer {
    pub dimension: usize,
    pub num_subspaces: usize,
    pub num_clusters: usize,
    pub subspace_dim: usize,
    /// Codebooks for each subspace: [num_subspaces][num_clusters][subspace_dim]
    pub codebooks: Vec<Vec<Vec<f32>>>,
}

impl ProductQuantizer {
    /// Create a new PQ instance by training on a sample of vectors
    pub fn train(vectors: &[Vec<f32>], num_subspaces: usize, num_clusters: usize) -> Result<Self> {
        if vectors.is_empty() {
            anyhow::bail!("Cannot train PQ on empty vector set");
        }
        
        let dimension = vectors[0].len();
        if dimension % num_subspaces != 0 {
            anyhow::bail!("Dimension {} must be divisible by num_subspaces {}", dimension, num_subspaces);
        }
        
        let subspace_dim = dimension / num_subspaces;
        let mut codebooks = Vec::with_capacity(num_subspaces);
        
        for m in 0..num_subspaces {
            let start = m * subspace_dim;
            let end = start + subspace_dim;
            
            // Extract subspace vectors for training
            let subspace_vectors: Vec<Vec<f32>> = vectors.iter()
                .map(|v| v[start..end].to_vec())
                .collect();
            
            // Train k-means for this subspace
            let centroids = train_kmeans(&subspace_vectors, num_clusters, 10)?;
            codebooks.push(centroids);
        }
        
        Ok(Self {
            dimension,
            num_subspaces,
            num_clusters,
            subspace_dim,
            codebooks,
        })
    }
    
    /// Encode a full vector into a PQ code
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let mut code = Vec::with_capacity(self.num_subspaces);
        
        for m in 0..self.num_subspaces {
            let start = m * self.subspace_dim;
            let end = start + self.subspace_dim;
            let subspace_vec = &vector[start..end];
            
            // Find nearest centroid in this subspace
            let mut min_dist = f32::MAX;
            let mut best_cluster = 0;
            
            for (i, centroid) in self.codebooks[m].iter().enumerate() {
                let dist = euclidean_distance_simd(subspace_vec, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = i;
                }
            }
            
            code.push(best_cluster as u8);
        }
        
        code
    }
    
    /// Compute distance between a query vector and a PQ code (ADC: Asymmetric Distance Computation)
    pub fn distance_adc(&self, query: &[f32], code: &[u8]) -> f32 {
        let mut total_dist_sq = 0.0;
        
        for m in 0..self.num_subspaces {
            let start = m * self.subspace_dim;
            let end = start + self.subspace_dim;
            let query_subspace = &query[start..end];
            let centroid = &self.codebooks[m][code[m] as usize];
            
            // Use squared distance for speed, take sqrt at the end if needed
            let d = euclidean_distance_simd(query_subspace, centroid);
            total_dist_sq += d * d;
        }
        
        total_dist_sq.sqrt()
    }

    /// Precompute distance table for a query to speed up many ADC lookups
    pub fn compute_distance_table(&self, query: &[f32]) -> Vec<Vec<f32>> {
        let mut table = Vec::with_capacity(self.num_subspaces);
        for m in 0..self.num_subspaces {
            let start = m * self.subspace_dim;
            let end = start + self.subspace_dim;
            let query_subspace = &query[start..end];
            
            let mut subspace_dists = Vec::with_capacity(self.num_clusters);
            for centroid in &self.codebooks[m] {
                subspace_dists.push(euclidean_distance_simd(query_subspace, centroid));
            }
            table.push(subspace_dists);
        }
        table
    }

    /// Compute distance using a precomputed table (very fast)
    pub fn distance_with_table(&self, table: &[Vec<f32>], code: &[u8]) -> f32 {
        let mut total_dist_sq = 0.0;
        for m in 0..self.num_subspaces {
            let d = table[m][code[m] as usize];
            total_dist_sq += d * d;
        }
        total_dist_sq.sqrt()
    }
}

/// Simple K-means implementation for codebook training
fn train_kmeans(data: &[Vec<f32>], k: usize, max_iters: usize) -> Result<Vec<Vec<f32>>> {
    if data.len() < k {
        anyhow::bail!("Not enough data to train k-means (needed {}, got {})", k, data.len());
    }
    
    let dim = data[0].len();
    let mut rng = thread_rng();
    
    // Initialize centroids by picking random points
    let mut centroids: Vec<Vec<f32>> = data.choose_multiple(&mut rng, k)
        .cloned()
        .collect();
    
    for _ in 0..max_iters {
        let mut new_centroids = vec![vec![0.0; dim]; k];
        let mut counts = vec![0; k];
        
        // Assignment step
        for point in data {
            let mut min_dist = f32::MAX;
            let mut best_k = 0;
            
            for (i, centroid) in centroids.iter().enumerate() {
                let dist = euclidean_distance_simd(point, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    best_k = i;
                }
            }
            
            for d in 0..dim {
                new_centroids[best_k][d] += point[d];
            }
            counts[best_k] += 1;
        }
        
        // Update step
        let mut changed = false;
        for i in 0..k {
            if counts[i] > 0 {
                for d in 0..dim {
                    let old_val = centroids[i][d];
                    centroids[i][d] = new_centroids[i][d] / counts[i] as f32;
                    if (old_val - centroids[i][d]).abs() > 1e-4 {
                        changed = true;
                    }
                }
            }
        }
        
        if !changed {
            break;
        }
    }
    
    Ok(centroids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq_basic() {
        // Generate some synthetic data
        let mut data = Vec::new();
        for i in 0..100 {
            let mut v = vec![0.0; 128];
            v[i % 128] = 1.0;
            data.push(v);
        }

        // Train PQ
        let pq = ProductQuantizer::train(&data, 8, 16).unwrap();
        assert_eq!(pq.dimension, 128);
        assert_eq!(pq.num_subspaces, 8);
        assert_eq!(pq.subspace_dim, 16);

        // Encode and check distance
        let v = &data[0];
        let code = pq.encode(v);
        assert_eq!(code.len(), 8);

        let dist = pq.distance_adc(v, &code);
        assert!(dist < 1.0); // Should be very close to 0

        // Test with distance table
        let table = pq.compute_distance_table(v);
        let dist_table = pq.distance_with_table(&table, &code);
        assert!((dist - dist_table).abs() < 1e-5);
    }
}
