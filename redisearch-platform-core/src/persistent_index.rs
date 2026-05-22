use crate::storage::{RedbVectorStorage, IndexMetadata};
use crate::vector_index::{VectorIndex, VectorMetric};
use crate::quantization::ProductQuantizer;
use std::path::Path;
use std::sync::Arc;
use anyhow::Result;

/// Persistent vector index with redb backing
pub struct PersistentVectorIndex {
    index: VectorIndex,
    storage: Arc<RedbVectorStorage>,
    index_name: String,
    snapshot_counter: usize,
    snapshot_interval: usize,
    /// Product Quantizer for memory-efficient search
    pub quantizer: Option<ProductQuantizer>,
    /// Quantized vectors stored in RAM: [internal_id] -> PQ code
    pub quantized_vectors: std::collections::HashMap<u64, Vec<u8>>,
}

impl PersistentVectorIndex {
    /// Create a new persistent vector index
    pub fn new(
        index_name: String,
        dimension: usize,
        metric: VectorMetric,
        storage_path: Option<&Path>,
        snapshot_interval: usize,
    ) -> Result<Self> {
        // Open or create storage
        let storage = Arc::new(if let Some(path) = storage_path {
            RedbVectorStorage::open(path)?
        } else {
            RedbVectorStorage::open_in_memory()?
        });
        
        // Try to recover metadata
        let index = if let Some(metadata) = storage.get_index_metadata(&index_name)? {
            // Recover from metadata
            VectorIndex::with_threshold(
                metadata.dimension,
                match metadata.metric.as_str() {
                    "cosine" => VectorMetric::Cosine,
                    "euclidean" => VectorMetric::Euclidean,
                    "inner_product" => VectorMetric::InnerProduct,
                    _ => VectorMetric::Cosine,
                },
                10_000, // Default threshold
            )
        } else {
            // Create new index
            VectorIndex::new(dimension, metric)
        };
        
        Ok(Self {
            index,
            storage,
            index_name,
            snapshot_counter: 0,
            snapshot_interval,
            quantizer: None,
            quantized_vectors: std::collections::HashMap::new(),
        })
    }
    
    /// Add a vector (with persistence and quantization)
    pub fn add(&mut self, doc_id: u64, vector: Vec<f32>) -> Result<()> {
        // Add to in-memory index
        self.index.add(doc_id, vector.clone())?;
        
        // Persist to redb
        self.storage.store_vector(doc_id, &vector)?;
        
        // Quantize if quantizer is available
        if let Some(pq) = &self.quantizer {
            let code = pq.encode(&vector);
            self.quantized_vectors.insert(doc_id, code);
        }
        
        // Check if we need to snapshot
        self.snapshot_counter += 1;
        if self.snapshot_counter >= self.snapshot_interval {
            self.snapshot()?;
            self.snapshot_counter = 0;
        }
        
        Ok(())
    }
    
    /// Batch add vectors (more efficient)
    pub fn add_batch(&mut self, vectors: Vec<(u64, Vec<f32>)>) -> Result<()> {
        // Add to in-memory index
        for (doc_id, vector) in &vectors {
            self.index.add(*doc_id, vector.clone())?;
            
            // Quantize if quantizer is available
            if let Some(pq) = &self.quantizer {
                let code = pq.encode(vector);
                self.quantized_vectors.insert(*doc_id, code);
            }
        }
        
        // Batch persist to redb
        let vectors_ref: Vec<(u64, Vec<f32>)> = vectors.iter()
            .map(|(id, vec)| (*id, vec.clone()))
            .collect();
        self.storage.store_vectors_batch(&vectors_ref)?;
        
        // Update snapshot counter
        self.snapshot_counter += vectors.len();
        if self.snapshot_counter >= self.snapshot_interval {
            self.snapshot()?;
            self.snapshot_counter = 0;
        }
        
        Ok(())
    }
    
    /// Search for similar vectors (Two-stage: Quantized search + Full re-ranking)
    pub fn search(&self, query_vector: &[f32], top_k: usize) -> Result<Vec<(u64, f32)>> {
        // If we have quantization, use the two-stage approach
        if let Some(_pq) = &self.quantizer {
            // 1. Quantized search (Stage 1)
            // For now, we still rely on the HNSW index which has its own nodes.
            // In a full DiskANN implementation, the HNSW graph would navigate PQ vectors.
            // Here we'll simulate the refinement by getting candidates from HNSW 
            // and re-ranking them with full vectors from disk.
            
            let candidates = self.index.search(query_vector, top_k * 5)?; // Get more candidates
            
            // 2. High-precision re-ranking (Stage 2)
            let mut reranked = Vec::with_capacity(candidates.len());
            for (doc_id, _) in candidates {
                if let Some(full_vec) = self.storage.get_vector(doc_id)? {
                    let score = match self.index.metric() {
                        VectorMetric::Cosine => crate::simd_metrics::cosine_similarity_simd(query_vector, &full_vec),
                        VectorMetric::Euclidean => -crate::simd_metrics::euclidean_distance_simd(query_vector, &full_vec),
                        VectorMetric::InnerProduct => crate::simd_metrics::inner_product_simd(query_vector, &full_vec),
                    };
                    reranked.push((doc_id, score));
                }
            }
            
            reranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            return Ok(reranked.into_iter().take(top_k).collect());
        }

        // Fallback to standard HNSW search
        self.index.search(query_vector, top_k)
    }

    /// Train Product Quantizer using existing vectors
    pub fn train_quantizer(&mut self, num_subspaces: usize, num_clusters: usize, sample_size: usize) -> Result<()> {
        // Collect a sample of vectors from storage
        let mut sample = Vec::new();
        // This is a bit inefficient for very large datasets, should use a reservoir or random IDs
        for i in 0..sample_size.min(self.index.len()) {
            // Placeholder: need a way to iterate IDs. 
            // For now, assume IDs are sequential for training sample
            if let Ok(Some(v)) = self.storage.get_vector(i as u64) {
                sample.push(v);
            }
        }
        
        if sample.is_empty() {
            anyhow::bail!("No vectors found for training");
        }

        let pq = ProductQuantizer::train(&sample, num_subspaces, num_clusters)?;
        
        // Re-encode all existing vectors (Stage 1 initialization)
        // In production, this would be done in batches
        for i in 0..self.index.len() {
            if let Ok(Some(v)) = self.storage.get_vector(i as u64) {
                let code = pq.encode(&v);
                self.quantized_vectors.insert(i as u64, code);
            }
        }
        
        self.quantizer = Some(pq);
        Ok(())
    }

    /// Remove a vector
    pub fn remove(&mut self, doc_id: u64) -> Result<()> {
        self.index.remove(doc_id)
    }

    /// Get the number of vectors
    pub fn len(&self) -> usize {
        self.index.len()
    }
    
    /// Create HNSW snapshot
    pub fn snapshot(&mut self) -> Result<()> {
        // For now, just update metadata
        // Full HNSW snapshot would require serializing the graph structure
        let metadata = IndexMetadata {
            dimension: self.index.dimension(),
            metric: match self.index.metric() {
                VectorMetric::Cosine => "cosine".to_string(),
                VectorMetric::Euclidean => "euclidean".to_string(),
                VectorMetric::InnerProduct => "inner_product".to_string(),
            },
            num_vectors: self.index.len(),
            m: 16,
            ef_construction: 150,
            ef_search: 50,
        };
        
        self.storage.store_index_metadata(&self.index_name, &metadata)?;
        
        Ok(())
    }
    
    /// Get storage reference
    pub fn storage(&self) -> &Arc<RedbVectorStorage> {
        &self.storage
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    
    #[test]
    fn test_persistent_index() {
        use std::time::{SystemTime, UNIX_EPOCH};
        let temp_dir = env::temp_dir();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let db_path = temp_dir.join(format!("test_persistent_index_{}.db", timestamp));
        let _ = std::fs::remove_file(&db_path);
        
        println!("\n=== Testing Persistent Vector Index ===\n");
        
        // Create persistent index
        let mut pindex = PersistentVectorIndex::new(
            "test_index".to_string(),
            384,
            VectorMetric::Cosine,
            Some(&db_path),
            10, // Snapshot every 10 vectors
        ).unwrap();
        
        // Add vectors
        let vectors: Vec<(u64, Vec<f32>)> = (0..20)
            .map(|i| {
                let mut vec = vec![0.0; 384];
                vec[i % 384] = 1.0;
                (i as u64, vec)
            })
            .collect();
        
        pindex.add_batch(vectors.clone()).unwrap();
        
        // Verify search works
        let query = vec![1.0; 384];
        let results = pindex.search(&query, 5).unwrap();
        assert!(!results.is_empty());
        
        // Drop the index to close the database
        drop(pindex);
        
        // Verify persistence: create new index and check recovery
        let pindex2 = PersistentVectorIndex::new(
            "test_index".to_string(),
            384,
            VectorMetric::Cosine,
            Some(&db_path),
            10,
        ).unwrap();
        
        let metadata = pindex2.storage().get_index_metadata("test_index").unwrap().unwrap();
        assert_eq!(metadata.num_vectors, 20);
        
        let _ = std::fs::remove_file(&db_path);
        println!("✅ Persistent index test passed!");
    }
}

