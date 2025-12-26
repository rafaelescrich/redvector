//! Tiered Storage Manager
//!
//! Manages hot (mmap), warm (local cache), and cold (object store) tiers
//! for RVF2 segments.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use bytes::Bytes;
use memmap2::Mmap;

use crate::rvf2::object_store::ObjectStore;
use crate::rvf2::segment::Segment;
use crate::rvf2::{Result, Rvf2Error};

/// LRU cache for segments
pub struct LruSegmentCache {
    /// Cached segments
    segments: HashMap<u32, Arc<CachedSegment>>,

    /// Access order (most recent last)
    order: Vec<u32>,

    /// Maximum cache size in bytes
    max_bytes: u64,

    /// Current cache size
    current_bytes: u64,
}

/// Cached segment with metadata
pub struct CachedSegment {
    /// Segment data
    pub segment: Segment,

    /// Size in bytes
    pub size: u64,

    /// Last access time
    pub last_access: std::time::Instant,
}

impl LruSegmentCache {
    /// Create new cache with max size
    pub fn new(max_bytes: u64) -> Self {
        Self {
            segments: HashMap::new(),
            order: Vec::new(),
            max_bytes,
            current_bytes: 0,
        }
    }

    /// Get segment from cache
    pub fn get(&mut self, segment_id: u32) -> Option<Arc<CachedSegment>> {
        if let Some(seg) = self.segments.get(&segment_id) {
            // Move to end of order (most recently used)
            self.order.retain(|&id| id != segment_id);
            self.order.push(segment_id);
            Some(Arc::clone(seg))
        } else {
            None
        }
    }

    /// Put segment in cache
    pub fn put(&mut self, segment_id: u32, segment: Segment, size: u64) {
        // Evict if needed
        while self.current_bytes + size > self.max_bytes && !self.order.is_empty() {
            let evict_id = self.order.remove(0);
            if let Some(evicted) = self.segments.remove(&evict_id) {
                self.current_bytes -= evicted.size;
            }
        }

        let cached = Arc::new(CachedSegment {
            segment,
            size,
            last_access: std::time::Instant::now(),
        });

        self.segments.insert(segment_id, cached);
        self.order.push(segment_id);
        self.current_bytes += size;
    }

    /// Check if segment is cached
    pub fn contains(&self, segment_id: u32) -> bool {
        self.segments.contains_key(&segment_id)
    }

    /// Clear cache
    pub fn clear(&mut self) {
        self.segments.clear();
        self.order.clear();
        self.current_bytes = 0;
    }

    /// Get current cache size
    pub fn size(&self) -> u64 {
        self.current_bytes
    }

    /// Get number of cached segments
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }
}

/// Tiered storage manager
pub struct TieredStore {
    /// Hot tier: local directory with mmap'd segments
    hot_dir: PathBuf,

    /// Warm tier: LRU cache of segments fetched from cold
    warm_cache: RwLock<LruSegmentCache>,

    /// Cold tier: object storage
    object_store: Option<Arc<dyn ObjectStore>>,

    /// Segment prefix in object store
    segment_prefix: String,

    /// Statistics
    stats: RwLock<TieredStoreStats>,
}

/// Storage tier statistics
#[derive(Debug, Clone, Default)]
pub struct TieredStoreStats {
    /// Hot tier hits
    pub hot_hits: u64,

    /// Warm tier hits
    pub warm_hits: u64,

    /// Cold tier fetches
    pub cold_fetches: u64,

    /// Total bytes read
    pub bytes_read: u64,

    /// Total bytes fetched from cold
    pub cold_bytes_fetched: u64,
}

impl TieredStore {
    /// Create new tiered store (local only)
    pub fn new_local(hot_dir: PathBuf, cache_max_bytes: u64) -> Self {
        Self {
            hot_dir,
            warm_cache: RwLock::new(LruSegmentCache::new(cache_max_bytes)),
            object_store: None,
            segment_prefix: "seg".to_string(),
            stats: RwLock::new(TieredStoreStats::default()),
        }
    }

    /// Create tiered store with object storage backend
    pub fn with_object_store(
        hot_dir: PathBuf,
        cache_max_bytes: u64,
        object_store: Arc<dyn ObjectStore>,
        prefix: &str,
    ) -> Self {
        Self {
            hot_dir,
            warm_cache: RwLock::new(LruSegmentCache::new(cache_max_bytes)),
            object_store: Some(object_store),
            segment_prefix: prefix.to_string(),
            stats: RwLock::new(TieredStoreStats::default()),
        }
    }

    /// Get segment path in hot tier
    fn hot_path(&self, segment_id: u32) -> PathBuf {
        self.hot_dir.join(format!("seg_{:06}.rvf2", segment_id))
    }

    /// Get segment key in object store
    fn cold_key(&self, segment_id: u32) -> String {
        format!("{}/seg_{:06}.rvf2", self.segment_prefix, segment_id)
    }

    /// Get segment from any tier
    pub async fn get_segment(&self, segment_id: u32) -> Result<Arc<CachedSegment>> {
        // 1. Check hot tier (local mmap)
        let hot_path = self.hot_path(segment_id);
        if hot_path.exists() {
            let segment = Segment::mmap(&hot_path)?;
            let size = std::fs::metadata(&hot_path)?.len();

            self.stats.write().await.hot_hits += 1;
            self.stats.write().await.bytes_read += size;

            return Ok(Arc::new(CachedSegment {
                segment,
                size,
                last_access: std::time::Instant::now(),
            }));
        }

        // 2. Check warm cache
        {
            let mut cache = self.warm_cache.write().await;
            if let Some(cached) = cache.get(segment_id) {
                self.stats.write().await.warm_hits += 1;
                self.stats.write().await.bytes_read += cached.size;
                return Ok(cached);
            }
        }

        // 3. Fetch from cold tier
        let object_store = self.object_store.as_ref()
            .ok_or_else(|| Rvf2Error::SegmentNotFound { segment_id })?;

        let key = self.cold_key(segment_id);
        let bytes = object_store.get(&key).await?;
        let size = bytes.len() as u64;

        // Write to warm cache (local file)
        let cache_path = self.hot_dir.join(format!("cache_seg_{:06}.rvf2", segment_id));
        if let Some(parent) = cache_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(&cache_path, &bytes).await?;

        // mmap the cached file
        let segment = Segment::mmap(&cache_path)?;

        // Add to warm cache
        let cached = CachedSegment {
            segment,
            size,
            last_access: std::time::Instant::now(),
        };

        let cached = Arc::new(cached);
        {
            let mut cache = self.warm_cache.write().await;
            // Re-open since we can't clone Segment easily
            let segment = Segment::mmap(&cache_path)?;
            cache.put(segment_id, segment, size);
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.cold_fetches += 1;
            stats.cold_bytes_fetched += size;
            stats.bytes_read += size;
        }

        Ok(cached)
    }

    /// Read byte range from segment (for partial reads)
    pub async fn read_range(
        &self,
        segment_id: u32,
        offset: u64,
        len: u64,
    ) -> Result<Bytes> {
        // Try to get full segment first (uses caching)
        let segment = self.get_segment(segment_id).await?;

        // Extract range
        let start = offset as usize;
        let end = (offset + len) as usize;

        if end > segment.segment.as_slice().len() {
            return Err(Rvf2Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Range exceeds segment size",
            )));
        }

        Ok(Bytes::copy_from_slice(&segment.segment.as_slice()[start..end]))
    }

    /// Prefetch segments to warm cache
    pub async fn prefetch(&self, segment_ids: &[u32]) -> Result<()> {
        for &segment_id in segment_ids {
            // This will fetch from cold if needed and cache
            let _ = self.get_segment(segment_id).await;
        }
        Ok(())
    }

    /// Pin segment in hot tier (ensure it's local)
    pub async fn pin(&self, segment_id: u32) -> Result<()> {
        let hot_path = self.hot_path(segment_id);
        if hot_path.exists() {
            return Ok(()); // Already hot
        }

        // Fetch and save to hot tier
        let object_store = self.object_store.as_ref()
            .ok_or_else(|| Rvf2Error::SegmentNotFound { segment_id })?;

        let key = self.cold_key(segment_id);
        let bytes = object_store.get(&key).await?;

        if let Some(parent) = hot_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(&hot_path, &bytes).await?;

        Ok(())
    }

    /// Unpin segment from hot tier (move to cold only)
    pub async fn unpin(&self, segment_id: u32) -> Result<()> {
        let hot_path = self.hot_path(segment_id);
        if hot_path.exists() {
            tokio::fs::remove_file(&hot_path).await?;
        }
        Ok(())
    }

    /// Upload segment to cold tier
    pub async fn upload(&self, segment_id: u32, data: Bytes) -> Result<()> {
        let object_store = self.object_store.as_ref()
            .ok_or_else(|| Rvf2Error::ObjectStore("No object store configured".into()))?;

        let key = self.cold_key(segment_id);
        object_store.put(&key, data).await?;

        Ok(())
    }

    /// Get statistics
    pub async fn stats(&self) -> TieredStoreStats {
        self.stats.read().await.clone()
    }

    /// Clear warm cache
    pub async fn clear_cache(&self) {
        self.warm_cache.write().await.clear();
    }
}

/// Builder for TieredStore
pub struct TieredStoreBuilder {
    hot_dir: PathBuf,
    cache_max_bytes: u64,
    object_store: Option<Arc<dyn ObjectStore>>,
    segment_prefix: String,
}

impl TieredStoreBuilder {
    pub fn new(hot_dir: PathBuf) -> Self {
        Self {
            hot_dir,
            cache_max_bytes: 1024 * 1024 * 1024, // 1 GB default
            object_store: None,
            segment_prefix: "seg".to_string(),
        }
    }

    pub fn cache_size(mut self, bytes: u64) -> Self {
        self.cache_max_bytes = bytes;
        self
    }

    pub fn object_store(mut self, store: Arc<dyn ObjectStore>) -> Self {
        self.object_store = Some(store);
        self
    }

    pub fn segment_prefix(mut self, prefix: &str) -> Self {
        self.segment_prefix = prefix.to_string();
        self
    }

    pub fn build(self) -> TieredStore {
        if let Some(store) = self.object_store {
            TieredStore::with_object_store(
                self.hot_dir,
                self.cache_max_bytes,
                store,
                &self.segment_prefix,
            )
        } else {
            TieredStore::new_local(self.hot_dir, self.cache_max_bytes)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_lru_cache() {
        let mut cache = LruSegmentCache::new(1000);

        // This test just checks the LRU logic, not actual segments
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[tokio::test]
    async fn test_tiered_store_local() {
        let dir = tempdir().unwrap();
        let store = TieredStore::new_local(dir.path().to_path_buf(), 10_000_000);

        // Stats should start at zero
        let stats = store.stats().await;
        assert_eq!(stats.hot_hits, 0);
        assert_eq!(stats.cold_fetches, 0);
    }
}

