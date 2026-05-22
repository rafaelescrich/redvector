//! Storage module for persistent vector storage
//!
//! Provides redb-based persistence for vectors, metadata, and HNSW snapshots
//! with LRU caching for performance.

use anyhow::{Context, Result};
use redb::{Database, ReadableTable, TableDefinition, backends::InMemoryBackend};
use std::path::Path;
use std::sync::{Arc, Mutex};
use bincode;
use lru::LruCache;
use std::num::NonZeroUsize;

// Table definitions
const VECTORS_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("vectors");
const METADATA_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("metadata");
const PAYLOAD_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("payload");
const HNSW_SNAPSHOT_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("hnsw_snapshot");
const INDEX_CONFIG_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("index_config");

/// Index metadata stored in redb
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IndexMetadata {
    pub dimension: usize,
    pub metric: String,
    pub num_vectors: usize,
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
}

/// redb-backed vector storage with LRU cache
pub struct RedbVectorStorage {
    db: Database,
    /// LRU cache for hot vectors (thread-safe)
    cache: Arc<Mutex<LruCache<u64, Vec<f32>>>>,
    /// Cache size
    cache_size: usize,
}

impl RedbVectorStorage {
    /// Open or create redb database
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_with_cache_size(path, 10_000)
    }

    /// Open an in-memory redb database
    pub fn open_in_memory() -> Result<Self> {
        Self::open_in_memory_with_cache_size(10_000)
    }
    
    /// Open with custom cache size
    pub fn open_with_cache_size(path: &Path, cache_size: usize) -> Result<Self> {
        let db = Database::create(path)
            .context("Failed to create redb database")?;
        Self::init_with_db(db, cache_size)
    }

    /// Open in-memory with custom cache size
    pub fn open_in_memory_with_cache_size(cache_size: usize) -> Result<Self> {
        let backend = InMemoryBackend::new();
        let db = Database::builder()
            .create_with_backend(backend)
            .context("Failed to create in-memory redb database")?;
        Self::init_with_db(db, cache_size)
    }

    fn init_with_db(db: Database, cache_size: usize) -> Result<Self> {
        // Initialize tables
        let write_txn = db.begin_write()
            .context("Failed to begin write transaction")?;
        
        {
            let _ = write_txn.open_table(VECTORS_TABLE)?;
            let _ = write_txn.open_table(METADATA_TABLE)?;
            let _ = write_txn.open_table(PAYLOAD_TABLE)?;
            let _ = write_txn.open_table(HNSW_SNAPSHOT_TABLE)?;
            let _ = write_txn.open_table(INDEX_CONFIG_TABLE)?;
        }
        
        write_txn.commit()
            .context("Failed to commit transaction")?;
        
        let cache = Arc::new(Mutex::new(
            LruCache::new(NonZeroUsize::new(cache_size).unwrap())
        ));
        
        Ok(Self {
            db,
            cache,
            cache_size,
        })
    }
    
    /// Store a vector (with caching)
    pub fn store_vector(&self, doc_id: u64, vector: &[f32]) -> Result<()> {
        // Update cache
        {
            let mut cache = self.cache.lock().unwrap();
            cache.put(doc_id, vector.to_vec());
        }
        
        // Persist to redb
        let serialized = bincode::serialize(vector)
            .context("Failed to serialize vector")?;
        
        let write_txn = self.db.begin_write()
            .context("Failed to begin write transaction")?;
        
        {
            let mut table = write_txn.open_table(VECTORS_TABLE)
                .context("Failed to open vectors table")?;
            
            table.insert(doc_id, serialized.as_slice())
                .context("Failed to insert vector")?;
        }
        
        write_txn.commit()
            .context("Failed to commit transaction")?;
        
        Ok(())
    }
    
    /// Batch store vectors (more efficient)
    pub fn store_vectors_batch(&self, vectors: &[(u64, Vec<f32>)]) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }
        
        // Update cache
        {
            let mut cache = self.cache.lock().unwrap();
            for (doc_id, vector) in vectors {
                cache.put(*doc_id, vector.clone());
            }
        }
        
        // Batch persist to redb
        let write_txn = self.db.begin_write()
            .context("Failed to begin write transaction")?;
        
        {
            let mut table = write_txn.open_table(VECTORS_TABLE)
                .context("Failed to open vectors table")?;
            
            for (doc_id, vector) in vectors {
                let serialized = bincode::serialize(vector)
                    .context("Failed to serialize vector")?;
                
                table.insert(*doc_id, serialized.as_slice())
                    .context("Failed to insert vector")?;
            }
        }
        
        write_txn.commit()
            .context("Failed to commit transaction")?;
        
        Ok(())
    }
    
    /// Retrieve a vector (checks cache first)
    pub fn get_vector(&self, doc_id: u64) -> Result<Option<Vec<f32>>> {
        // Check cache first
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(cached) = cache.get(&doc_id) {
                return Ok(Some(cached.clone()));
            }
        }
        
        // Load from redb
        let read_txn = self.db.begin_read()
            .context("Failed to begin read transaction")?;
        
        let table = read_txn.open_table(VECTORS_TABLE)
            .context("Failed to open vectors table")?;
        
        let result = if let Some(serialized) = table.get(doc_id)
            .context("Failed to get vector")? {
            // Deserialize while transaction is still alive
            let vector: Vec<f32> = bincode::deserialize(serialized.value())
                .context("Failed to deserialize vector")?;
            
            // Update cache
            {
                let mut cache = self.cache.lock().unwrap();
                cache.put(doc_id, vector.clone());
            }
            
            Some(vector)
        } else {
            None
        };
        
        Ok(result)
    }
    
    /// Store index metadata
    pub fn store_index_metadata(&self, index_name: &str, metadata: &IndexMetadata) -> Result<()> {
        let serialized = bincode::serialize(metadata)
            .context("Failed to serialize metadata")?;
        
        let write_txn = self.db.begin_write()
            .context("Failed to begin write transaction")?;
        
        {
            let mut table = write_txn.open_table(INDEX_CONFIG_TABLE)
                .context("Failed to open index config table")?;
            
            table.insert(index_name, serialized.as_slice())
                .context("Failed to insert metadata")?;
        }
        
        write_txn.commit()
            .context("Failed to commit transaction")?;
        
        Ok(())
    }
    
    /// Retrieve index metadata
    pub fn get_index_metadata(&self, index_name: &str) -> Result<Option<IndexMetadata>> {
        let read_txn = self.db.begin_read()
            .context("Failed to begin read transaction")?;
        
        let table = read_txn.open_table(INDEX_CONFIG_TABLE)
            .context("Failed to open index config table")?;
        
        let result = if let Some(serialized) = table.get(index_name)
            .context("Failed to get metadata")? {
            let metadata: IndexMetadata = bincode::deserialize(serialized.value())
                .context("Failed to deserialize metadata")?;
            Some(metadata)
        } else {
            None
        };
        
        Ok(result)
    }
    
    /// Store HNSW snapshot (for recovery)
    pub fn store_hnsw_snapshot(&self, snapshot_id: &str, snapshot_data: &[u8]) -> Result<()> {
        let write_txn = self.db.begin_write()
            .context("Failed to begin write transaction")?;
        
        {
            let mut table = write_txn.open_table(HNSW_SNAPSHOT_TABLE)
                .context("Failed to open HNSW snapshot table")?;
            
            table.insert(snapshot_id, snapshot_data)
                .context("Failed to insert snapshot")?;
        }
        
        write_txn.commit()
            .context("Failed to commit transaction")?;
        
        Ok(())
    }
    
    /// Retrieve HNSW snapshot
    pub fn get_hnsw_snapshot(&self, snapshot_id: &str) -> Result<Option<Vec<u8>>> {
        let read_txn = self.db.begin_read()
            .context("Failed to begin read transaction")?;
        
        let table = read_txn.open_table(HNSW_SNAPSHOT_TABLE)
            .context("Failed to open HNSW snapshot table")?;
        
        let result = if let Some(data) = table.get(snapshot_id)
            .context("Failed to get snapshot")? {
            Some(data.value().to_vec())
        } else {
            None
        };
        
        Ok(result)
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        let cache = self.cache.lock().unwrap();
        (cache.len(), self.cache_size)
    }
    
    /// Clear cache
    pub fn clear_cache(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();
    }
}

