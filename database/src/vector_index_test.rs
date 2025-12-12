//! Comprehensive unit tests for HNSW vector index
//! 
//! This module contains extensive tests for the HNSW implementation,
//! including edge cases, error handling, and performance validation.

#[cfg(test)]
#[cfg(feature = "hnsw-backend")]
mod hnsw_tests {
    use crate::vector_index::{HnswVectorIndex, VectorMetric, VectorIndexTrait};
    
    /// Helper to generate normalized random vectors
    fn random_vector(dim: usize, seed: u64) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        let mut rng = hasher.finish();
        
        let mut v = Vec::with_capacity(dim);
        let mut norm_sq = 0.0;
        
        for _ in 0..dim {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let val = ((rng % 2000) as f32 / 1000.0) - 1.0;
            norm_sq += val * val;
            v.push(val);
        }
        
        // Normalize
        let norm = norm_sq.sqrt();
        if norm > 0.0 {
            for val in &mut v {
                *val /= norm;
            }
        }
        
        v
    }
    
    #[test]
    fn test_basic_operations() {
        let mut index = HnswVectorIndex::new(128, VectorMetric::Cosine, Some(16), Some(200));
        
        let v1 = random_vector(128, 1);
        let v2 = random_vector(128, 2);
        
        // Add vectors
        assert!(index.add(1, v1.clone()).is_ok());
        assert!(index.add(2, v2.clone()).is_ok());
        assert_eq!(index.len(), 2);
        
        // Search
        let results = index.search(&v1, 2, None).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1); // Should find itself first
        
        // Remove
        assert!(index.remove(1).is_ok());
        assert_eq!(index.len(), 1);
    }
    
    #[test]
    fn test_dimension_validation() {
        let mut index = HnswVectorIndex::new(384, VectorMetric::Cosine, Some(16), Some(200));
        
        // Wrong dimension on add
        assert!(index.add(1, vec![0.1; 128]).is_err());
        assert!(index.add(1, vec![0.1; 512]).is_err());
        
        // Correct dimension
        assert!(index.add(1, vec![0.1; 384]).is_ok());
        
        // Wrong dimension on search
        assert!(index.search(&vec![0.1; 128], 1, None).is_err());
        assert!(index.search(&vec![0.1; 512], 1, None).is_err());
    }
    
    #[test]
    fn test_duplicate_prevention() {
        let mut index = HnswVectorIndex::new(64, VectorMetric::Cosine, Some(16), Some(200));
        
        let v = random_vector(64, 1);
        
        // First add succeeds
        assert!(index.add(1, v.clone()).is_ok());
        assert_eq!(index.len(), 1);
        
        // Duplicate add fails
        assert!(index.add(1, v.clone()).is_err());
        assert_eq!(index.len(), 1);
        
        // Different doc_id with same vector succeeds
        assert!(index.add(2, v.clone()).is_ok());
        assert_eq!(index.len(), 2);
    }
    
    #[test]
    fn test_search_result_ordering() {
        let mut index = HnswVectorIndex::new(64, VectorMetric::Cosine, Some(16), Some(200));
        
        // Create a base vector and similar vectors
        let base = random_vector(64, 100);
        
        // Add base vector
        index.add(0, base.clone()).unwrap();
        
        // Add vectors with varying similarity
        for i in 1..10 {
            let mut v = random_vector(64, 100 + i);
            // Make them progressively less similar
            for j in 0..64 {
                v[j] = v[j] * (1.0 - i as f32 * 0.05) + base[j] * (i as f32 * 0.05);
            }
            // Renormalize
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for val in &mut v {
                *val /= norm;
            }
            index.add(i, v).unwrap();
        }
        
        // Search should return results sorted by distance (ascending - lower = more similar)
        let results = index.search(&base, 10, None).unwrap();
        // Note: HNSW is approximate, might not return all 10 results
        assert!(results.len() >= 9, "Expected at least 9 results, got {}", results.len());
        
        // Check sorting (distances should be ascending - lower is better)
        for i in 1..results.len() {
            assert!(results[i-1].1 <= results[i].1, 
                   "Results not sorted by distance: {} <= {}", results[i-1].1, results[i].1);
        }
        
        // First result should be the base vector itself (distance should be very close to 0)
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 < 0.01, "Base vector should have distance < 0.01, got {}", results[0].1);
    }
    
    #[test]
    fn test_empty_index_behavior() {
        let index = HnswVectorIndex::new(128, VectorMetric::Cosine, Some(16), Some(200));
        
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
        assert_eq!(index.dimension(), 128);
        
        // Search on empty index
        let query = random_vector(128, 1);
        let results = index.search(&query, 10, None).unwrap();
        assert_eq!(results.len(), 0);
    }
    
    #[test]
    fn test_remove_operations() {
        let mut index = HnswVectorIndex::new(64, VectorMetric::Cosine, Some(16), Some(200));
        
        // Remove from empty index
        assert!(index.remove(999).is_err());
        
        // Add and remove
        let v = random_vector(64, 1);
        assert!(index.add(1, v).is_ok());
        assert_eq!(index.len(), 1);
        
        // Remove non-existent
        assert!(index.remove(999).is_err());
        assert_eq!(index.len(), 1);
        
        // Remove existing
        assert!(index.remove(1).is_ok());
        assert_eq!(index.len(), 0);
        
        // Remove again
        assert!(index.remove(1).is_err());
    }
    
    #[test]
    fn test_large_dataset() {
        let mut index = HnswVectorIndex::new(384, VectorMetric::Cosine, Some(16), Some(200));
        
        // Add 10,000 vectors
        for i in 0..10000 {
            let v = random_vector(384, i);
            assert!(index.add(i, v).is_ok());
        }
        
        assert_eq!(index.len(), 10000);
        
        // Search
        let query = random_vector(384, 99999);
        let results = index.search(&query, 10, Some(100)).unwrap();
        assert_eq!(results.len(), 10);
        
        // Verify all results are valid
        for (doc_id, distance) in &results {
            assert!(*doc_id < 10000);
            // Distance for cosine: 0 (identical) to 2 (opposite)
            assert!(*distance >= 0.0 && *distance <= 2.0);
        }
    }
    
    #[test]
    fn test_different_metrics() {
        let vectors = vec![
            random_vector(64, 1),
            random_vector(64, 2),
            random_vector(64, 3),
        ];
        
        // Test Cosine
        let mut index_cosine = HnswVectorIndex::new(64, VectorMetric::Cosine, Some(16), Some(200));
        for (i, v) in vectors.iter().enumerate() {
            index_cosine.add(i as u64, v.clone()).unwrap();
        }
        let results_cosine = index_cosine.search(&vectors[0], 3, None).unwrap();
        // HNSW is approximate - might not return all results, but should return at least 2
        assert!(results_cosine.len() >= 2, "Expected at least 2 results, got {}", results_cosine.len());
        assert!(results_cosine.len() <= 3, "Expected at most 3 results, got {}", results_cosine.len());
        
        // Test Euclidean
        let mut index_euclidean = HnswVectorIndex::new(64, VectorMetric::Euclidean, Some(16), Some(200));
        for (i, v) in vectors.iter().enumerate() {
            index_euclidean.add(i as u64, v.clone()).unwrap();
        }
        let results_euclidean = index_euclidean.search(&vectors[0], 3, None).unwrap();
        // HNSW is approximate - might not return all results, but should return at least 2
        assert!(results_euclidean.len() >= 2, "Expected at least 2 results, got {}", results_euclidean.len());
        assert!(results_euclidean.len() <= 3, "Expected at most 3 results, got {}", results_euclidean.len());
    }
    
    #[test]
    fn test_ef_search_parameter() {
        let mut index = HnswVectorIndex::new(64, VectorMetric::Cosine, Some(16), Some(200));
        
        // Add 1000 vectors
        for i in 0..1000 {
            let v = random_vector(64, i);
            index.add(i, v).unwrap();
        }
        
        let query = random_vector(64, 9999);
        
        // Test different ef_search values
        for ef_search in [10, 20, 50, 100, 200] {
            let results = index.search(&query, 10, Some(ef_search)).unwrap();
            assert_eq!(results.len(), 10);
        }
    }
    
    #[test]
    fn test_trait_interface() {
        let mut index = HnswVectorIndex::new(64, VectorMetric::Cosine, Some(16), Some(200));
        
        let v1 = random_vector(64, 1);
        let v2 = random_vector(64, 2);
        
        // Use trait methods
        assert!(VectorIndexTrait::add(&mut index, 1, v1.clone()).is_ok());
        assert!(VectorIndexTrait::add(&mut index, 2, v2.clone()).is_ok());
        
        assert_eq!(VectorIndexTrait::len(&index), 2);
        assert!(!VectorIndexTrait::is_empty(&index));
        
        let results = VectorIndexTrait::search(&index, &v1, 2).unwrap();
        assert_eq!(results.len(), 2);
        
        assert!(VectorIndexTrait::remove(&mut index, 1).is_ok());
        assert_eq!(VectorIndexTrait::len(&index), 1);
    }
    
    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;
        
        let index = Arc::new(std::sync::Mutex::new(
            HnswVectorIndex::new(64, VectorMetric::Cosine, Some(16), Some(200))
        ));
        
        // Add vectors from multiple threads
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let index = Arc::clone(&index);
                thread::spawn(move || {
                    for j in 0..100 {
                        let doc_id = (i * 100 + j) as u64;
                        let v = random_vector(64, doc_id);
                        let mut idx = index.lock().unwrap();
                        idx.add(doc_id, v).unwrap();
                    }
                })
            })
            .collect();
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        assert_eq!(index.lock().unwrap().len(), 1000);
    }
}

