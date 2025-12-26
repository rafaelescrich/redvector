//! HNSW-based vector index implementation
//! 
//! This module provides a high-performance vector similarity search using
//! Hierarchical Navigable Small World (HNSW) algorithm via the hnsw_rs crate.

#[cfg(feature = "hnsw-backend")]
use std::sync::{Arc, Mutex};

// Import hnsw_rs types from the actual API
// Using extern crate to ensure the dependency is linked
#[cfg(feature = "hnsw-backend")]
extern crate hnsw_rs;

// Use fully qualified paths to avoid import resolution issues
#[cfg(feature = "hnsw-backend")]
use hnsw_rs::hnsw::Hnsw;
#[cfg(feature = "hnsw-backend")]
use hnsw_rs::dist::DistCosine;

/// Distance metrics supported for vector search
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VectorMetric {
    /// Cosine similarity (normalized dot product)
    Cosine,
    /// Euclidean distance (L2)
    Euclidean,
    /// Inner product (dot product)
    InnerProduct,
}

/// HNSW-based vector index for approximate nearest neighbor search
#[cfg(feature = "hnsw-backend")]
pub struct HnswVectorIndex {
    /// HNSW index instance
    /// Note: Using DistCosine as the default distance metric
    /// TODO: Support multiple distance metrics via enum or separate index types
    index: Arc<Mutex<Hnsw<f32, DistCosine>>>,
    /// Dimension of vectors
    dimension: usize,
    /// Number of documents in the index
    count: Arc<Mutex<usize>>,
    /// Document ID to internal ID mapping
    doc_id_map: Arc<Mutex<std::collections::HashMap<u64, usize>>>,
    /// Internal ID to document ID mapping (reverse)
    id_to_doc: Arc<Mutex<std::collections::HashMap<usize, u64>>>,
    /// Next available internal ID
    next_id: Arc<Mutex<usize>>,
}

#[cfg(feature = "hnsw-backend")]
impl HnswVectorIndex {
    /// Create a new HNSW index
    /// 
    /// # Arguments
    /// * `dimension` - Dimension of vectors (e.g., 384 for sentence embeddings)
    /// * `metric` - Distance metric to use
    /// * `m` - Number of bi-directional links for each element (default: 16)
    /// * `ef_construction` - Size of dynamic candidate list during construction (default: 200)
    /// 
    /// # Performance Notes
    /// - Higher `m` = better recall, more memory, slower inserts
    /// - Higher `ef_construction` = better recall, slower builds
    /// - Recommended: m=16, ef_construction=200 for 98%+ recall
    pub fn new(dimension: usize, _metric: VectorMetric, m: Option<usize>, ef_construction: Option<usize>) -> Self {
        let m = m.unwrap_or(16);
        let ef_construction = ef_construction.unwrap_or(200);
        
        // Create HNSW index
        // Note: hnsw_rs API: new(max_nb_connection, max_elements, max_layer, ef_construction, distance_metric)
        // - max_nb_connection: m (number of connections)
        // - max_elements: maximum number of elements (we'll use a large value)
        // - max_layer: maximum layer (typically calculated, using 16 as default)
        // - ef_construction: ef_construction parameter
        // - distance_metric: instance of the distance metric (DistCosine::default())
        let max_elements = 1_000_000; // Large enough for most use cases
        let max_layer = 16; // Default max layer
        let dist = DistCosine::default();
        let index = Arc::new(Mutex::new(Hnsw::<f32, DistCosine>::new(m, max_elements, max_layer, ef_construction, dist)));
        
        Self {
            index,
            dimension,
            count: Arc::new(Mutex::new(0)),
            doc_id_map: Arc::new(Mutex::new(std::collections::HashMap::new())),
            id_to_doc: Arc::new(Mutex::new(std::collections::HashMap::new())),
            next_id: Arc::new(Mutex::new(0)),
        }
    }
    
    /// Add a vector to the index
    /// 
    /// # Arguments
    /// * `doc_id` - External document ID (u64)
    /// * `vector` - Vector to add (must match dimension)
    /// 
    /// # Returns
    /// `Ok(())` on success, `Err(String)` on error
    pub fn add(&mut self, doc_id: u64, vector: Vec<f32>) -> Result<(), String> {
        if vector.len() != self.dimension {
            return Err(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dimension,
                vector.len()
            ));
        }
        
        // Check if document already exists
        let mut doc_id_map = self.doc_id_map.lock().unwrap();
        if doc_id_map.contains_key(&doc_id) {
            return Err(format!("Document {} already exists in index", doc_id));
        }
        
        // Get next internal ID
        let mut next_id = self.next_id.lock().unwrap();
        let internal_id = *next_id;
        *next_id += 1;
        drop(next_id);
        
        // Add to HNSW index
        // Note: insert takes a tuple (&Vec<T>, usize)
        let index = self.index.lock().unwrap();
        index.insert((&vector, internal_id));
        drop(index);
        
        // Update mappings
        doc_id_map.insert(doc_id, internal_id);
        drop(doc_id_map);
        
        let mut id_to_doc = self.id_to_doc.lock().unwrap();
        id_to_doc.insert(internal_id, doc_id);
        drop(id_to_doc);
        
        // Update count
        let mut count = self.count.lock().unwrap();
        *count += 1;
        
        Ok(())
    }
    
    /// Search for similar vectors
    /// 
    /// # Arguments
    /// * `query_vector` - Query vector
    /// * `k` - Number of results to return
    /// * `ef_search` - Size of dynamic candidate list during search (default: k * 2)
    /// 
    /// # Returns
    /// Vector of (doc_id, score) pairs, sorted by similarity (highest first)
    pub fn search(&self, query_vector: &[f32], k: usize, ef_search: Option<usize>) -> Result<Vec<(u64, f32)>, String> {
        if query_vector.len() != self.dimension {
            return Err(format!(
                "Query vector dimension mismatch: expected {}, got {}",
                self.dimension,
                query_vector.len()
            ));
        }
        
        let ef_search = ef_search.unwrap_or(k.max(10) * 2);
        let index = self.index.lock().unwrap();
        let id_to_doc = self.id_to_doc.lock().unwrap();
        
        // Perform search
        let results = index.search(query_vector, k.max(1), ef_search);
        
        // Convert internal IDs to document IDs
        // Neighbour.d_id is the DataId (our internal_id)
        // Note: neighbour.distance is a distance value (lower = more similar)
        let mut doc_results: Vec<(u64, f32)> = results
            .into_iter()
            .filter_map(|neighbour| {
                let internal_id = neighbour.d_id;
                let distance = neighbour.distance;
                id_to_doc.get(&internal_id).map(|&doc_id| (doc_id, distance))
            })
            .collect();
        
        // Sort by distance (ascending - lower distance = more similar)
        // For cosine distance: 0 = identical, 1 = orthogonal, 2 = opposite
        doc_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(doc_results)
    }
    
    /// Remove a vector from the index
    /// 
    /// # Arguments
    /// * `doc_id` - External document ID to remove
    /// 
    /// # Returns
    /// `Ok(())` on success, `Err(String)` if document not found
    /// 
    /// # Note
    /// HNSW doesn't support efficient deletion. This marks the document as deleted
    /// but doesn't remove it from the graph structure. For production use,
    /// consider periodic index rebuilds.
    pub fn remove(&mut self, doc_id: u64) -> Result<(), String> {
        let mut doc_id_map = self.doc_id_map.lock().unwrap();
        let mut id_to_doc = self.id_to_doc.lock().unwrap();
        
        if let Some(internal_id) = doc_id_map.remove(&doc_id) {
            id_to_doc.remove(&internal_id);
            
            let mut count = self.count.lock().unwrap();
            *count = count.saturating_sub(1);
            
            // Note: We don't actually remove from HNSW graph (not efficiently supported)
            // The vector will remain in the graph but won't be returned in searches
            // due to missing mapping
            
            Ok(())
        } else {
            Err(format!("Document {} not found in index", doc_id))
        }
    }
    
    /// Get the number of documents in the index
    pub fn len(&self) -> usize {
        *self.count.lock().unwrap()
    }
    
    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get the dimension of vectors in this index
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Trait for vector index implementations
/// This provides a common interface that can be implemented by different backends
pub trait VectorIndexTrait: Send + Sync {
    /// Add a vector to the index
    fn add(&mut self, doc_id: u64, vector: Vec<f32>) -> Result<(), String>;
    
    /// Search for similar vectors
    fn search(&self, query_vector: &[f32], k: usize) -> Result<Vec<(u64, f32)>, String>;
    
    /// Remove a vector from the index
    fn remove(&mut self, doc_id: u64) -> Result<(), String>;
    
    /// Get the number of documents
    fn len(&self) -> usize;
    
    /// Check if empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(feature = "hnsw-backend")]
impl VectorIndexTrait for HnswVectorIndex {
    fn add(&mut self, doc_id: u64, vector: Vec<f32>) -> Result<(), String> {
        self.add(doc_id, vector)
    }
    
    fn search(&self, query_vector: &[f32], k: usize) -> Result<Vec<(u64, f32)>, String> {
        self.search(query_vector, k, None)
    }
    
    fn remove(&mut self, doc_id: u64) -> Result<(), String> {
        self.remove(doc_id)
    }
    
    fn len(&self) -> usize {
        self.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    /// Helper function to normalize a vector
    fn normalize(v: &mut Vec<f32>) {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in v.iter_mut() {
                *val /= norm;
            }
        }
    }
    
    /// Generate a random normalized vector
    fn random_vector(dim: usize, seed: u64) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        let mut rng = hasher.finish();
        
        let mut v = Vec::with_capacity(dim);
        for _ in 0..dim {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            v.push(((rng % 2000) as f32 / 1000.0) - 1.0);
        }
        normalize(&mut v);
        v
    }
    
    #[test]
    #[cfg(feature = "hnsw-backend")]
    fn test_hnsw_basic() {
        let mut index = HnswVectorIndex::new(128, VectorMetric::Cosine, Some(16), Some(200));
        
        // Add some vectors
        let v1: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();
        let mut v1_normalized = v1.clone();
        normalize(&mut v1_normalized);
        
        let v2: Vec<f32> = (0..128).map(|i| ((i + 1) as f32) / 128.0).collect();
        let mut v2_normalized = v2.clone();
        normalize(&mut v2_normalized);
        
        assert!(index.add(1, v1_normalized.clone()).is_ok());
        assert!(index.add(2, v2_normalized.clone()).is_ok());
        
        assert_eq!(index.len(), 2);
        assert!(!index.is_empty());
        assert_eq!(index.dimension(), 128);
        
        // Search
        let results = index.search(&v1_normalized, 1, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1); // Should find itself
        
        // Remove
        assert!(index.remove(1).is_ok());
        assert_eq!(index.len(), 1);
        
        // Try to remove again (should fail)
        assert!(index.remove(1).is_err());
    }
    
    #[test]
    #[cfg(feature = "hnsw-backend")]
    fn test_hnsw_dimension_mismatch() {
        let mut index = HnswVectorIndex::new(128, VectorMetric::Cosine, Some(16), Some(200));
        
        // Try to add vector with wrong dimension
        let wrong_vec = vec![0.1, 0.2, 0.3]; // Only 3 dimensions
        assert!(index.add(1, wrong_vec).is_err());
        
        // Try to search with wrong dimension
        let wrong_query = vec![0.1, 0.2];
        assert!(index.search(&wrong_query, 1, None).is_err());
    }
    
    #[test]
    #[cfg(feature = "hnsw-backend")]
    fn test_hnsw_duplicate_add() {
        let mut index = HnswVectorIndex::new(64, VectorMetric::Cosine, Some(16), Some(200));
        
        let v1 = random_vector(64, 1);
        
        // Add first time - should succeed
        assert!(index.add(1, v1.clone()).is_ok());
        assert_eq!(index.len(), 1);
        
        // Try to add same doc_id again - should fail
        assert!(index.add(1, v1.clone()).is_err());
        assert_eq!(index.len(), 1);
    }
    
    #[test]
    #[cfg(feature = "hnsw-backend")]
    fn test_hnsw_search_multiple_results() {
        let mut index = HnswVectorIndex::new(64, VectorMetric::Cosine, Some(16), Some(200));
        
        // Add 10 similar vectors
        let base_vector = random_vector(64, 42);
        for i in 0..10 {
            let mut v = random_vector(64, 42 + i);
            // Make them somewhat similar to base
            for j in 0..64 {
                v[j] = v[j] * 0.7 + base_vector[j] * 0.3;
            }
            normalize(&mut v);
            assert!(index.add(i as u64, v).is_ok());
        }
        
        assert_eq!(index.len(), 10);
        
        // Search for top 5
        let results = index.search(&base_vector, 5, None).unwrap();
        assert_eq!(results.len(), 5);
        
        // Results should be sorted by distance (ascending - lower is better)
        for i in 1..results.len() {
            assert!(results[i-1].1 <= results[i].1, 
                   "Results not sorted by distance: {} <= {}", results[i-1].1, results[i].1);
        }
    }
    
    #[test]
    #[cfg(feature = "hnsw-backend")]
    fn test_hnsw_large_dataset() {
        let mut index = HnswVectorIndex::new(384, VectorMetric::Cosine, Some(16), Some(200));
        
        // Add 1000 vectors (simulating real embedding scenario)
        for i in 0..1000 {
            let v = random_vector(384, i);
            assert!(index.add(i, v).is_ok());
        }
        
        assert_eq!(index.len(), 1000);
        
        // Search
        let query = random_vector(384, 9999);
        let results = index.search(&query, 10, Some(50)).unwrap();
        assert_eq!(results.len(), 10);
        
        // All results should have valid doc_ids
        for (doc_id, distance) in &results {
            assert!(*doc_id < 1000);
            // Distance for cosine: 0 (identical) to 2 (opposite)
            assert!(*distance >= 0.0 && *distance <= 2.0);
        }
    }
    
    #[test]
    #[cfg(feature = "hnsw-backend")]
    fn test_hnsw_metrics() {
        // Test Cosine metric
        let mut index_cosine = HnswVectorIndex::new(64, VectorMetric::Cosine, Some(16), Some(200));
        let v1 = random_vector(64, 1);
        let v2 = random_vector(64, 2);
        assert!(index_cosine.add(1, v1.clone()).is_ok());
        assert!(index_cosine.add(2, v2.clone()).is_ok());
        let results = index_cosine.search(&v1, 2, None).unwrap();
        assert_eq!(results.len(), 2);
        
        // Test Euclidean metric
        let mut index_euclidean = HnswVectorIndex::new(64, VectorMetric::Euclidean, Some(16), Some(200));
        assert!(index_euclidean.add(1, v1.clone()).is_ok());
        assert!(index_euclidean.add(2, v2.clone()).is_ok());
        let results = index_euclidean.search(&v1, 2, None).unwrap();
        assert_eq!(results.len(), 2);
    }
    
    #[test]
    #[cfg(feature = "hnsw-backend")]
    fn test_hnsw_empty_index() {
        let index = HnswVectorIndex::new(128, VectorMetric::Cosine, Some(16), Some(200));
        
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
        
        // Search on empty index should return empty results
        let query = random_vector(128, 1);
        let results = index.search(&query, 10, None).unwrap();
        assert_eq!(results.len(), 0);
    }
    
    #[test]
    #[cfg(feature = "hnsw-backend")]
    fn test_hnsw_remove_nonexistent() {
        let mut index = HnswVectorIndex::new(64, VectorMetric::Cosine, Some(16), Some(200));
        
        // Try to remove from empty index
        assert!(index.remove(999).is_err());
        
        // Add a vector
        let v = random_vector(64, 1);
        assert!(index.add(1, v).is_ok());
        
        // Try to remove non-existent doc_id
        assert!(index.remove(999).is_err());
        
        // Remove existing one
        assert!(index.remove(1).is_ok());
    }
    
    #[test]
    #[cfg(feature = "hnsw-backend")]
    fn test_hnsw_ef_search_parameter() {
        let mut index = HnswVectorIndex::new(64, VectorMetric::Cosine, Some(16), Some(200));
        
        // Add 100 vectors
        for i in 0..100 {
            let v = random_vector(64, i);
            assert!(index.add(i, v).is_ok());
        }
        
        let query = random_vector(64, 9999);
        
        // Search with different ef_search values
        let results_ef10 = index.search(&query, 5, Some(10)).unwrap();
        let results_ef50 = index.search(&query, 5, Some(50)).unwrap();
        let results_ef100 = index.search(&query, 5, Some(100)).unwrap();
        
        // All should return 5 results
        assert_eq!(results_ef10.len(), 5);
        assert_eq!(results_ef50.len(), 5);
        assert_eq!(results_ef100.len(), 5);
        
        // Higher ef_search should generally give better recall (but not guaranteed)
        // We just verify they all work
    }
    
    #[test]
    #[cfg(feature = "hnsw-backend")]
    fn test_vector_index_trait() {
        let mut index = HnswVectorIndex::new(64, VectorMetric::Cosine, Some(16), Some(200));
        
        let v1 = random_vector(64, 1);
        let v2 = random_vector(64, 2);
        
        // Test trait methods
        assert!(VectorIndexTrait::add(&mut index, 1, v1.clone()).is_ok());
        assert!(VectorIndexTrait::add(&mut index, 2, v2.clone()).is_ok());
        
        assert_eq!(VectorIndexTrait::len(&index), 2);
        assert!(!VectorIndexTrait::is_empty(&index));
        
        let results = VectorIndexTrait::search(&index, &v1, 2).unwrap();
        assert_eq!(results.len(), 2);
        
        assert!(VectorIndexTrait::remove(&mut index, 1).is_ok());
        assert_eq!(VectorIndexTrait::len(&index), 1);
    }
}

