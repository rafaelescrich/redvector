use crate::storage::{RedbVectorStorage, IndexMetadata};
use crate::vector_index::{VectorIndex, VectorMetric};
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
}

impl PersistentVectorIndex {
    /// Create a new persistent vector index
    pub fn new(
        index_name: String,
        dimension: usize,
        metric: VectorMetric,
        storage_path: &Path,
        snapshot_interval: usize,
    ) -> Result<Self> {
        // Open or create storage
        let storage = Arc::new(RedbVectorStorage::open(storage_path)?);
        
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
        })
    }
    
    /// Add a vector (with persistence)
    pub fn add(&mut self, doc_id: u64, vector: Vec<f32>) -> Result<()> {
        // Add to in-memory index
        self.index.add(doc_id, vector.clone())?;
        
        // Persist to redb
        self.storage.store_vector(doc_id, &vector)?;
        
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
    
    /// Search for similar vectors
    pub fn search(&self, query_vector: &[f32], top_k: usize) -> Result<Vec<(u64, f32)>> {
        self.index.search(query_vector, top_k)
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
            &db_path,
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
            &db_path,
            10,
        ).unwrap();
        
        let metadata = pindex2.storage().get_index_metadata("test_index").unwrap().unwrap();
        assert_eq!(metadata.num_vectors, 20);
        
        let _ = std::fs::remove_file(&db_path);
        println!("✅ Persistent index test passed!");
    }
}

