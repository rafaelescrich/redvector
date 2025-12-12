//! Redisearch Platform Core
//!
//! Unified platform combining rsedis (Redis-compatible server) and
//! redisearch-rust-port (full-text and vector search) into a single,
//! high-performance search and vector database.

pub mod vector_index;
pub mod storage;
pub mod integration;
pub mod simd_metrics;
pub mod persistent_index;

use anyhow::Result;

/// Platform configuration
#[derive(Debug, Clone)]
pub struct PlatformConfig {
    /// Redis server port
    pub redis_port: u16,
    /// redb database path
    pub redb_path: Option<std::path::PathBuf>,
    /// Enable persistence
    pub persistence_enabled: bool,
    /// Hot cache size (number of vectors)
    pub hot_cache_size: usize,
    /// HNSW snapshot interval
    pub hnsw_snapshot_interval: usize,
}

impl Default for PlatformConfig {
    fn default() -> Self {
        Self {
            redis_port: 6379,
            redb_path: Some(std::path::PathBuf::from("./data/platform.redb")),
            persistence_enabled: false,
            hot_cache_size: 10_000,
            hnsw_snapshot_interval: 10_000,
        }
    }
}

/// Main platform instance
pub struct RedisearchPlatform {
    config: PlatformConfig,
    // Will be added as we implement
}

impl RedisearchPlatform {
    /// Create a new platform instance
    pub fn new(config: PlatformConfig) -> Result<Self> {
        Ok(Self {
            config,
        })
    }
    
    /// Start the platform
    pub async fn start(&mut self) -> Result<()> {
        // Implementation will be added
        Ok(())
    }
}

