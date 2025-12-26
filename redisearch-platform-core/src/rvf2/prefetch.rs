//! Prefetch: Async range-based prefetching for batch reranking
//!
//! Coordinates efficient fetching of document records from tiered storage.

use bytes::Bytes;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Semaphore;

use crate::rvf2::object_store::ObjectStore;
use crate::rvf2::Result;

/// Request for a document record
#[derive(Debug, Clone)]
pub struct FetchRequest {
    /// Document ID
    pub doc_id: u64,
    /// Segment ID
    pub segment_id: u32,
    /// Byte offset in segment
    pub offset: u32,
    /// Byte length
    pub len: u32,
}

/// Fetched document data
#[derive(Debug)]
pub struct FetchedDoc {
    /// Document ID
    pub doc_id: u64,
    /// Raw bytes of the document record
    pub data: Bytes,
}

/// Prefetch coordinator for batch operations
pub struct PrefetchCoordinator {
    /// Object store for cold tier
    object_store: Arc<dyn ObjectStore>,

    /// Max concurrent fetches
    max_concurrency: usize,

    /// Range coalescing threshold (bytes)
    /// Merge ranges within this distance
    coalesce_threshold: u64,

    /// Segment path prefix
    segment_prefix: String,
}

impl PrefetchCoordinator {
    /// Create new coordinator
    pub fn new(object_store: Arc<dyn ObjectStore>) -> Self {
        Self {
            object_store,
            max_concurrency: 64,
            coalesce_threshold: 4096, // 4KB
            segment_prefix: "seg".to_string(),
        }
    }

    /// Set max concurrency
    pub fn with_concurrency(mut self, max: usize) -> Self {
        self.max_concurrency = max;
        self
    }

    /// Set coalesce threshold
    pub fn with_coalesce_threshold(mut self, threshold: u64) -> Self {
        self.coalesce_threshold = threshold;
        self
    }

    /// Set segment prefix
    pub fn with_segment_prefix(mut self, prefix: &str) -> Self {
        self.segment_prefix = prefix.to_string();
        self
    }

    /// Prefetch multiple documents, returning data for each
    pub async fn prefetch(&self, requests: &[FetchRequest]) -> Result<Vec<FetchedDoc>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        // Group by segment
        let mut by_segment: HashMap<u32, Vec<&FetchRequest>> = HashMap::new();
        for req in requests {
            by_segment.entry(req.segment_id).or_default().push(req);
        }

        // Coalesce ranges within each segment
        let mut fetch_tasks = Vec::new();
        let semaphore = Arc::new(Semaphore::new(self.max_concurrency));

        for (segment_id, segment_requests) in by_segment {
            let coalesced = self.coalesce_ranges(&segment_requests);
            let segment_key = self.segment_key(segment_id);

            for (offset, len, doc_ids) in coalesced {
                let store = self.object_store.clone();
                let sem = semaphore.clone();
                let key = segment_key.clone();

                fetch_tasks.push(tokio::spawn(async move {
                    let _permit = sem.acquire().await.unwrap();
                    let data = store.get_range(&key, offset, len).await?;
                    Ok::<_, crate::rvf2::Rvf2Error>((offset, data, doc_ids))
                }));
            }
        }

        // Wait for all fetches
        let results = futures::future::join_all(fetch_tasks).await;

        // Map fetched data back to original requests
        let mut output = Vec::with_capacity(requests.len());
        let mut offset_to_data: HashMap<u64, Bytes> = HashMap::new();

        for result in results {
            let (offset, data, _doc_ids) = result.map_err(|e| {
                crate::rvf2::Rvf2Error::ObjectStore(format!("Task join error: {}", e))
            })??;
            offset_to_data.insert(offset, data);
        }

        // Extract individual doc records from coalesced fetches
        for req in requests {
            // Find the coalesced range containing this request
            for (&coalesced_offset, data) in &offset_to_data {
                let req_offset = req.offset as u64;
                if req_offset >= coalesced_offset
                    && req_offset + req.len as u64 <= coalesced_offset + data.len() as u64
                {
                    let start = (req_offset - coalesced_offset) as usize;
                    let end = start + req.len as usize;
                    output.push(FetchedDoc {
                        doc_id: req.doc_id,
                        data: data.slice(start..end),
                    });
                    break;
                }
            }
        }

        Ok(output)
    }

    /// Coalesce nearby ranges to reduce number of fetches
    ///
    /// Returns: Vec<(offset, len, doc_ids in this range)>
    fn coalesce_ranges(&self, requests: &[&FetchRequest]) -> Vec<(u64, u64, Vec<u64>)> {
        if requests.is_empty() {
            return Vec::new();
        }

        // Sort by offset
        let mut sorted: Vec<_> = requests.iter().collect();
        sorted.sort_by_key(|r| r.offset);

        let mut result = Vec::new();
        let mut current_start = sorted[0].offset as u64;
        let mut current_end = current_start + sorted[0].len as u64;
        let mut current_docs = vec![sorted[0].doc_id];

        for req in sorted.into_iter().skip(1) {
            let req_start = req.offset as u64;
            let req_end = req_start + req.len as u64;

            if req_start < current_end + self.coalesce_threshold {
                // Extend current range
                current_end = current_end.max(req_end);
                current_docs.push(req.doc_id);
            } else {
                // Gap too large, emit current and start new
                result.push((current_start, current_end - current_start, current_docs));
                current_start = req_start;
                current_end = req_end;
                current_docs = vec![req.doc_id];
            }
        }

        // Emit final range
        result.push((current_start, current_end - current_start, current_docs));

        result
    }

    fn segment_key(&self, segment_id: u32) -> String {
        format!("{}/seg_{:06}.rvf2", self.segment_prefix, segment_id)
    }
}

/// Statistics for prefetch operations
#[derive(Debug, Default)]
pub struct PrefetchStats {
    /// Total requests
    pub total_requests: usize,
    /// Requests coalesced into fewer fetches
    pub coalesced_fetches: usize,
    /// Total bytes fetched
    pub bytes_fetched: u64,
    /// Cache hits (warm tier)
    pub cache_hits: usize,
    /// Cache misses (cold tier)
    pub cache_misses: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rvf2::object_store::LocalFsStore;
    use tempfile::tempdir;

    #[test]
    fn test_coalesce_ranges() {
        let dir = tempdir().unwrap();
        let store = Arc::new(LocalFsStore::new(dir.path()));
        let coord = PrefetchCoordinator::new(store).with_coalesce_threshold(100);

        let requests = vec![
            FetchRequest {
                doc_id: 1,
                segment_id: 0,
                offset: 0,
                len: 50,
            },
            FetchRequest {
                doc_id: 2,
                segment_id: 0,
                offset: 50,
                len: 50,
            },
            FetchRequest {
                doc_id: 3,
                segment_id: 0,
                offset: 200,
                len: 50,
            },
        ];

        let refs: Vec<_> = requests.iter().collect();
        let coalesced = coord.coalesce_ranges(&refs);

        // First two should be merged (within threshold)
        // Third is separate (gap > 100)
        assert_eq!(coalesced.len(), 2);
        assert_eq!(coalesced[0].0, 0); // offset
        assert_eq!(coalesced[0].1, 100); // len (merged)
        assert_eq!(coalesced[1].0, 200);
        assert_eq!(coalesced[1].1, 50);
    }
}

