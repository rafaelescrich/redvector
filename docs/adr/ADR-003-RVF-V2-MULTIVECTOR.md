# ADR-003: RVF v2 Storage Format for Multi-Vector Retrieval

## Status

**Proposed**

## Date

2025-12-25

## Decision Makers

- Rafael Escrich

---

## Context

We need scalable multi-vector retrieval (ColPali/ColBERT-style) where each document/page has **P patch vectors** (often ~1024) plus a **pooled/summarized** embedding for ANN candidate generation. At scale, storing and scoring patch matrices is the dominant cost.

### Problem Statement

| Challenge | Impact |
|-----------|--------|
| 1024 patches × 128 dims × 4 bytes = **512KB/doc** | Storage explosion at scale |
| MaxSim requires fetching ALL patches for candidates | I/O bound reranking |
| Object storage latency (50-200ms) | Network bottleneck |
| Billions of documents | Can't fit in RAM |

### Requirements

1. **Two-stage retrieval**: ANN over pooled vectors → MaxSim rerank over patch matrices
2. **Efficient storage**: SQ8/FP16/PQ codecs for 4-32x compression
3. **Cloud-native**: Local mmap + object storage (S3/GCS/Azure/MinIO) with cache
4. **O(1) doc lookup**: Compute offset directly, no B-tree traversal
5. **Immutable segments**: Append-only, compaction-friendly

---

## Decision

Adopt **RVF v2** format with:

1. **Two-tier storage**: local segment store (mmap/NVMe) + object storage backend with cache + range fetch
2. **Segmented immutable data layout**: append-only segments, compacted offline/async
3. **Codec-aware layout**: SQ8 default; FP16 optional for high-accuracy rerank; PQ for cold/candidate-only
4. **Doc index for O(1) lookup**: `doc_id → (segment, offset, len)` in mmapable array

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Two-Stage Retrieval Flow                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Query                                                                       │
│    │                                                                         │
│    └─> embed(query)                                                          │
│          │                                                                   │
│          ├─> query_pooled [1×D]      ──────┐                                │
│          │                                  │                                │
│          └─> query_tokens [T×D]      ──────┼─────────────────────┐          │
│                                             │                     │          │
│                                             ▼                     │          │
│                              ┌──────────────────────────┐        │          │
│                              │  Stage A: ANN Search     │        │          │
│                              │  (HNSW/IVF on pooled)    │        │          │
│                              │  O(log N) or O(√N)       │        │          │
│                              └───────────┬──────────────┘        │          │
│                                          │                        │          │
│                                          ▼                        │          │
│                              candidate doc_ids (Top-K=200)       │          │
│                                          │                        │          │
│                                          ▼                        ▼          │
│                              ┌──────────────────────────────────────────┐   │
│                              │  Stage B: MaxSim Rerank                  │   │
│                              │  ────────────────────────                │   │
│                              │  1. Lookup doc_ids in DocIndex           │   │
│                              │     → (segment_id, offset, len)          │   │
│                              │                                           │   │
│                              │  2. Fetch patch matrices from RVF2       │   │
│                              │     → range GET from cache/object store  │   │
│                              │                                           │   │
│                              │  3. Compute MaxSim(query_tokens, patches)│   │
│                              │     → SIMD/GPU accelerated               │   │
│                              │                                           │   │
│                              │  4. Sort by score, return Top-10         │   │
│                              └──────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Invariants

| Stage | Requirement | Approach |
|-------|-------------|----------|
| A | Fast, ANN-friendly | HNSW/IVF on pooled vectors (single vector/doc) |
| B | Minimize bytes moved | SQ8 compression (4x), range GET coalescing |
| B | Maximize dot-product throughput | SIMD AVX2/NEON, optional GPU |

---

## Data Model

### Assumptions (configurable per index)

```rust
const DEFAULT_DIM: u16 = 128;           // ColPali projection dimension
const DEFAULT_PATCHES: u16 = 1024;       // Patches per doc/page
const QUERY_TOKENS_TYPICAL: usize = 32;  // Query token count
```

### Size Calculations

| Codec | Bytes per Patch | 1024-patch Doc | 1M Docs | Compression |
|-------|-----------------|----------------|---------|-------------|
| FP32 | 512 | 512 KB | 512 GB | 1x |
| FP16 | 256 | 256 KB | 256 GB | 2x |
| **SQ8** | 128 + scales | **~130 KB** | **130 GB** | **4x** |
| PQ16 | 16 | 16 KB | 16 GB | 32x |

---

## File/Object Layout

### Directory Structure

```
index_name/
├── manifest.rvf2                 # Small, atomically replaced (MessagePack)
├── doc_index.rvf2                # Mmapable: doc_id → (segment, offset, len, flags)
├── pooled/                       # Stage A: ANN index on pooled vectors
│   ├── vectors.rvf2              # Pooled vectors (one per doc)
│   └── hnsw_graph/               # HNSW graph files
├── seg/                          # Stage B: Patch matrix segments
│   ├── seg_000000.rvf2           # Immutable segment blobs (256MB-1GB)
│   ├── seg_000001.rvf2
│   └── ...
├── codebooks/                    # PQ codebooks (if using PQ codec)
│   └── pq_m16_k256.bin
└── wal/                          # Write-ahead log (Phase 2)
    └── wal_000000.rvf2
```

### Storage Tiers

| Tier | Location | Latency | Use Case |
|------|----------|---------|----------|
| **Hot** | mmap (NVMe/RAM) | <1ms | Active segments, frequent access |
| **Warm** | Local SSD cache | 1-10ms | Fetched from cold, LRU cached |
| **Cold** | Object store (S3/GCS/MinIO) | 50-200ms | Archive, range GET on demand |

---

## On-Disk Structures

### 1. Manifest (`manifest.rvf2`)

**Encoding**: MessagePack (compact, fast parsing)

```rust
/// Manifest file - small, atomically replaced on updates
#[derive(Serialize, Deserialize)]
pub struct Manifest {
    /// Magic bytes: "RVF2"
    pub magic: [u8; 4],
    
    /// Format version (2 for RVF v2)
    pub version: u32,
    
    /// Vector dimension (e.g., 128 for ColPali)
    pub dims: u16,
    
    /// Default codec for patches
    pub codec: Codec,
    
    /// Expected patches per doc (can vary per doc)
    pub patches_per_doc: u16,
    
    /// Target segment size in bytes (e.g., 512MB)
    pub segment_target_bytes: u32,
    
    /// List of segments
    pub segments: Vec<SegmentMeta>,
    
    /// Creation timestamp (Unix epoch)
    pub created_at_unix: u64,
    
    /// Last update timestamp
    pub updated_at_unix: u64,
}

#[derive(Serialize, Deserialize)]
pub struct SegmentMeta {
    /// Unique segment identifier
    pub segment_id: u32,
    
    /// Filename (e.g., "seg_000123.rvf2")
    pub file_name: String,
    
    /// Total bytes
    pub bytes: u64,
    
    /// Number of documents in segment
    pub doc_count: u32,
    
    /// CRC32C checksum
    pub crc32c: u32,
    
    /// Min doc_id in segment (for range queries)
    pub min_doc_id: u64,
    
    /// Max doc_id in segment
    pub max_doc_id: u64,
    
    /// Codec used (can differ from manifest default)
    pub codec: Codec,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum Codec {
    /// Int8 scalar quantization (4x compression)
    Sq8 = 1,
    
    /// Float16 (2x compression, high accuracy)
    Fp16 = 2,
    
    /// Product quantization m=16 (32x compression)
    Pq16 = 3,
    
    /// Full precision (no compression)
    Fp32 = 0,
}
```

### 2. Doc Index (`doc_index.rvf2`)

**Goal**: O(log N) lookup via binary search, or O(1) with hash index

```rust
/// Doc index file header
#[repr(C, packed)]
pub struct DocIndexHeader {
    /// Magic: "RVDI"
    pub magic: [u8; 4],
    
    /// Version
    pub version: u32,
    
    /// Number of entries
    pub entry_count: u64,
    
    /// Flags (sorted, has_hash_index, etc.)
    pub flags: u32,
    
    /// Reserved for alignment
    pub reserved: u32,
}

/// Single doc entry - 32 bytes, cache-line friendly
#[repr(C, packed)]
pub struct DocEntry {
    /// External document ID (Redis key hash or sequential ID)
    pub doc_id: u64,
    
    /// Segment containing this document
    pub segment_id: u32,
    
    /// Byte offset within segment
    pub offset: u32,
    
    /// Byte length of record
    pub len: u32,
    
    /// Flags: codec override, variable patch count, tombstone, etc.
    pub flags: u16,
    
    /// Reserved for future use
    pub reserved: u16,
}

// Total: 24 bytes per entry
// 10M docs = 240 MB doc index (easily mmapable)
```

**Lookup Strategies**:

| Strategy | Memory | Lookup Time | Use Case |
|----------|--------|-------------|----------|
| Binary search (sorted by doc_id) | 0 extra | O(log N) | Cold start, simple |
| Robin Hood hash (in-memory) | ~4 bytes/entry | O(1) | Hot path, frequent lookups |
| Hybrid (hash for hot, binary for cold) | Adaptive | O(1) average | Production recommended |

### 3. Segment File (`seg_XXXXXX.rvf2`)

#### Segment Header

```rust
/// Segment file header - 64 bytes
#[repr(C, packed)]
pub struct SegmentHeader {
    /// Magic: "RVF2"
    pub magic: [u8; 4],
    
    /// Format version
    pub version: u32,
    
    /// Segment ID
    pub segment_id: u32,
    
    /// Default codec for this segment
    pub codec: u8,
    
    /// Flags (compressed, has_footer_index, etc.)
    pub flags: u8,
    
    /// Header length in bytes
    pub header_len: u16,
    
    /// Number of documents in segment
    pub doc_count: u32,
    
    /// Offset to optional footer index
    pub index_offset: u32,
    
    /// Length of footer index
    pub index_len: u32,
    
    /// CRC32C of header + payload (excluding this field)
    pub crc32c: u32,
    
    /// Dimension
    pub dims: u16,
    
    /// Expected patches per doc
    pub patches_per_doc: u16,
    
    /// Reserved for alignment to 64 bytes
    pub reserved: [u8; 24],
}
```

#### Document Record Layout

Each document is stored contiguously for efficient range GET:

```rust
/// Per-document record header - 32 bytes
#[repr(C, packed)]
pub struct DocRecordHeader {
    /// Document ID
    pub doc_id: u64,
    
    /// Number of patches (can vary from segment default)
    pub n_patches: u16,
    
    /// Dimension (can vary for future flexibility)
    pub dims: u16,
    
    /// Codec for pooled vector
    pub pooled_codec: u8,
    
    /// Codec for patch vectors (may override segment codec)
    pub patches_codec: u8,
    
    /// Reserved
    pub reserved: u16,
    
    /// Length of pooled vector blob
    pub pooled_len: u32,
    
    /// Length of patch codes blob
    pub patches_len: u32,
    
    /// Length of scales/aux data (SQ8 scales, PQ refs, etc.)
    pub scales_len: u32,
    
    /// CRC32C of this record
    pub record_crc32c: u32,
}

// Payload follows header (aligned to 64 bytes):
// 1. pooled_vector: [u8; pooled_len]
// 2. patch_codes:   [u8; patches_len]  (the matrix)
// 3. scales_aux:    [u8; scales_len]   (SQ8 scales or PQ codebook refs)
```

#### Memory Layout Diagram

```
Segment File:
┌──────────────────────────────────────────────────────────────────────────┐
│  SegmentHeader (64 bytes)                                                │
├──────────────────────────────────────────────────────────────────────────┤
│  DocRecord 0:                                                            │
│  ├─ DocRecordHeader (32 bytes)                                          │
│  ├─ pooled_vector (aligned, ~128-512 bytes depending on codec)          │
│  ├─ patch_codes (1024×128×1 = 128KB for SQ8)                            │
│  └─ scales_aux (~64-256 bytes for SQ8 block scales)                     │
├──────────────────────────────────────────────────────────────────────────┤
│  DocRecord 1:                                                            │
│  ├─ DocRecordHeader                                                      │
│  ├─ ...                                                                  │
├──────────────────────────────────────────────────────────────────────────┤
│  ... (more records)                                                      │
├──────────────────────────────────────────────────────────────────────────┤
│  Optional Footer Index (for intra-segment random access)                 │
│  [offset_0, offset_1, ..., offset_N]                                    │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Codec Details

### SQ8 (Recommended Default)

**Representation**: Int8 with per-block scaling

```rust
/// SQ8 block quantization parameters
pub struct Sq8Params {
    /// Patches per scale block (e.g., 32)
    pub block_size: u16,
    
    /// Scale format: f16 recommended
    pub scale_format: ScaleFormat,
}

#[derive(Clone, Copy)]
pub enum ScaleFormat {
    F16,  // 2 bytes per block
    F32,  // 4 bytes per block
}

/// Encode float32 patches to SQ8
pub fn encode_sq8(patches: &[f32], dims: usize, block_size: usize) -> Sq8Encoded {
    let n_patches = patches.len() / dims;
    let n_blocks = (n_patches + block_size - 1) / block_size;
    
    let mut codes = Vec::with_capacity(n_patches * dims);
    let mut scales = Vec::with_capacity(n_blocks);
    
    for block_idx in 0..n_blocks {
        let start = block_idx * block_size;
        let end = (start + block_size).min(n_patches);
        
        // Find max abs value in block
        let mut max_abs = 0.0f32;
        for i in start..end {
            for d in 0..dims {
                max_abs = max_abs.max(patches[i * dims + d].abs());
            }
        }
        
        // Scale factor: map [-max_abs, max_abs] to [-127, 127]
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        let inv_scale = 1.0 / scale;
        scales.push(half::f16::from_f32(scale));
        
        // Quantize
        for i in start..end {
            for d in 0..dims {
                let v = patches[i * dims + d];
                let q = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
                codes.push(q);
            }
        }
    }
    
    Sq8Encoded { codes, scales }
}

/// Decode SQ8 to float32 for dot product
#[inline]
pub fn decode_sq8_block(
    codes: &[i8],
    scale: f16,
    dims: usize,
    output: &mut [f32],
) {
    let scale_f32 = scale.to_f32();
    for (i, &code) in codes.iter().enumerate() {
        output[i] = code as f32 * scale_f32;
    }
}
```

**Storage Breakdown (1024×128 SQ8)**:

| Component | Size |
|-----------|------|
| Patch codes | 128 KB |
| Scales (block=32, f16) | 64 bytes |
| **Total** | **~128 KB** |

### FP16 (High-Accuracy Mode)

For domains where SQ8 quality loss is unacceptable:

```rust
/// Encode float32 patches to FP16
pub fn encode_fp16(patches: &[f32]) -> Vec<half::f16> {
    patches.iter().map(|&v| half::f16::from_f32(v)).collect()
}
```

**Storage**: 1024×128×2 = **256 KB/doc**

### PQ16 (Cold Tier / Candidate Only)

For extreme compression (32x) at cost of accuracy:

```rust
/// PQ16 parameters
pub struct Pq16Params {
    /// Number of subvectors (m=16 for 128-dim → 8 dims per subvector)
    pub m: u8,
    
    /// Codewords per subvector (256 = 8-bit codes)
    pub k: u16,
    
    /// Codebook path
    pub codebook_path: String,
}

// Storage per patch: m=16 × 1 byte = 16 bytes
// 1024 patches = 16 KB/doc
```

**Warning**: PQ is typically too lossy for final MaxSim scoring. Use for:
- Candidate-only storage (rerank with FP16/SQ8)
- Cold tier where cost > quality

---

## Segment Sizing Rules

### Target Sizes

| Codec | Doc Size | 512MB Segment | 1GB Segment |
|-------|----------|---------------|-------------|
| SQ8 | ~130 KB | ~3,900 docs | ~7,800 docs |
| FP16 | ~260 KB | ~1,900 docs | ~3,800 docs |
| PQ16 | ~18 KB | ~28,000 docs | ~56,000 docs |

### Guidelines

```rust
/// Segment sizing configuration
pub struct SegmentConfig {
    /// Target segment size (256MB - 1GB recommended)
    pub target_bytes: u64,
    
    /// Min docs before creating new segment
    pub min_docs_per_segment: u32,
    
    /// Max docs per segment (for metadata overhead)
    pub max_docs_per_segment: u32,
}

impl Default for SegmentConfig {
    fn default() -> Self {
        Self {
            target_bytes: 512 * 1024 * 1024,  // 512 MB
            min_docs_per_segment: 1000,
            max_docs_per_segment: 50_000,
        }
    }
}
```

---

## Compaction Strategy

### Phase 1: Simple (Immediate Implementation)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Append-Only + Periodic Compaction                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Write Path:                                                                 │
│  1. New doc → append to "active segment"                                    │
│  2. Update DocIndex with (segment, offset, len)                             │
│  3. When segment reaches target_bytes → seal and create new                 │
│                                                                              │
│  Delete Path:                                                                │
│  1. Mark doc as tombstone in DocIndex (flags |= TOMBSTONE)                  │
│  2. Actual data remains in segment until compaction                         │
│                                                                              │
│  Compaction:                                                                 │
│  1. Select segments with tombstone_ratio > threshold                        │
│  2. Read live docs, write to new segment                                    │
│  3. Atomically update manifest + doc index                                  │
│  4. Delete old segments                                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 2: WAL-Based (Production Durability)

```rust
/// Write-ahead log entry
#[repr(C, packed)]
pub struct WalEntry {
    /// Entry type
    pub entry_type: WalEntryType,
    
    /// Document ID
    pub doc_id: u64,
    
    /// Length of payload
    pub payload_len: u32,
    
    /// CRC32C
    pub crc32c: u32,
    
    // Followed by payload bytes
}

#[repr(u8)]
pub enum WalEntryType {
    Put = 1,
    Delete = 2,
}
```

**Compaction Triggers**:

| Trigger | Threshold | Action |
|---------|-----------|--------|
| Tombstone ratio | > 20% | Compact segment |
| Segment count | > 100 | Merge small segments |
| Cold storage cost | Periodic | Tier to object store |

---

## Cache + Object Storage

### SegmentStore Trait

```rust
/// Abstraction over local filesystem and object storage
#[async_trait]
pub trait SegmentStore: Send + Sync {
    /// Open segment for mmap access (returns None if remote-only)
    async fn open_local(&self, segment_id: u32) -> Result<Option<MmapSegment>>;
    
    /// Read byte range from segment
    async fn read_range(
        &self,
        segment_id: u32,
        offset: u64,
        len: u64,
    ) -> Result<Bytes>;
    
    /// Read multiple ranges (for batch prefetch)
    async fn read_ranges(
        &self,
        requests: &[(u32, u64, u64)],  // (segment_id, offset, len)
    ) -> Result<Vec<Bytes>>;
    
    /// Prefetch segment to local cache
    async fn prefetch_segment(&self, segment_id: u32) -> Result<()>;
    
    /// Write new segment
    async fn write_segment(
        &self,
        segment_id: u32,
        data: Bytes,
    ) -> Result<()>;
    
    /// Delete segment
    async fn delete_segment(&self, segment_id: u32) -> Result<()>;
}
```

### TieredSegmentStore Implementation

```rust
pub struct TieredSegmentStore {
    /// Hot tier: mmap'd segments on local NVMe
    hot_dir: PathBuf,
    
    /// Warm tier: LRU cache of fetched segments
    warm_cache: Arc<RwLock<LruCache<u32, MmapSegment>>>,
    warm_cache_max_bytes: u64,
    
    /// Cold tier: object storage
    object_store: Arc<dyn ObjectStore>,
    
    /// Metrics
    metrics: StoreMetrics,
}

#[async_trait]
impl SegmentStore for TieredSegmentStore {
    async fn read_range(
        &self,
        segment_id: u32,
        offset: u64,
        len: u64,
    ) -> Result<Bytes> {
        // 1. Check hot tier (local mmap)
        let hot_path = self.hot_dir.join(format!("seg_{:06}.rvf2", segment_id));
        if hot_path.exists() {
            let mmap = self.mmap_segment(&hot_path)?;
            return Ok(Bytes::copy_from_slice(&mmap[offset as usize..(offset + len) as usize]));
        }
        
        // 2. Check warm cache
        if let Some(mmap) = self.warm_cache.read().get(&segment_id) {
            self.metrics.warm_hits.inc();
            return Ok(Bytes::copy_from_slice(&mmap[offset as usize..(offset + len) as usize]));
        }
        
        // 3. Fetch from cold (object store)
        self.metrics.cold_fetches.inc();
        let bytes = self.object_store
            .get_range(
                &format!("seg/seg_{:06}.rvf2", segment_id),
                offset,
                len,
            )
            .await?;
        
        Ok(bytes)
    }
    
    async fn prefetch_segment(&self, segment_id: u32) -> Result<()> {
        // Download full segment to warm cache
        let bytes = self.object_store
            .get(&format!("seg/seg_{:06}.rvf2", segment_id))
            .await?;
        
        // Write to local cache
        let cache_path = self.warm_cache_path(segment_id);
        tokio::fs::write(&cache_path, &bytes).await?;
        
        // mmap and add to warm cache
        let mmap = unsafe { Mmap::map(&File::open(&cache_path)?)? };
        self.warm_cache.write().put(segment_id, mmap);
        
        Ok(())
    }
}
```

### Object Store Implementations

```rust
/// S3-compatible object store (AWS S3, MinIO, GCS, Azure Blob)
pub struct S3ObjectStore {
    client: aws_sdk_s3::Client,
    bucket: String,
    prefix: String,
}

impl S3ObjectStore {
    pub async fn get_range(&self, key: &str, offset: u64, len: u64) -> Result<Bytes> {
        let range = format!("bytes={}-{}", offset, offset + len - 1);
        
        let resp = self.client
            .get_object()
            .bucket(&self.bucket)
            .key(format!("{}/{}", self.prefix, key))
            .range(range)
            .send()
            .await?;
        
        Ok(resp.body.collect().await?.into_bytes())
    }
}

/// GCS object store
pub struct GcsObjectStore { /* similar */ }

/// Azure Blob store
pub struct AzureBlobStore { /* similar */ }

/// Local filesystem "object store" (for testing/development)
pub struct LocalFsStore {
    base_path: PathBuf,
}
```

---

## Async Prefetch Design

### Prefetch Strategy

```rust
/// Prefetch coordinator for batch reranking
pub struct PrefetchCoordinator {
    segment_store: Arc<dyn SegmentStore>,
    
    /// Max concurrent range fetches
    max_concurrency: usize,
    
    /// Range coalescing threshold (merge ranges within N bytes)
    coalesce_threshold: u64,
}

impl PrefetchCoordinator {
    /// Prefetch doc records for reranking
    pub async fn prefetch_docs(
        &self,
        candidates: &[(u64, u32, u32, u32)],  // (doc_id, segment, offset, len)
    ) -> Result<Vec<Bytes>> {
        // 1. Group by segment
        let mut by_segment: HashMap<u32, Vec<(u64, u32)>> = HashMap::new();
        for &(doc_id, seg, off, len) in candidates {
            by_segment.entry(seg).or_default().push((off as u64, len as u64));
        }
        
        // 2. Coalesce ranges within each segment
        let mut requests = Vec::new();
        for (seg_id, mut ranges) in by_segment {
            ranges.sort_by_key(|r| r.0);
            
            let coalesced = self.coalesce_ranges(&ranges);
            for (offset, len) in coalesced {
                requests.push((seg_id, offset, len));
            }
        }
        
        // 3. Fetch with bounded concurrency
        let semaphore = Arc::new(Semaphore::new(self.max_concurrency));
        let futures: Vec<_> = requests
            .into_iter()
            .map(|(seg, off, len)| {
                let store = self.segment_store.clone();
                let sem = semaphore.clone();
                async move {
                    let _permit = sem.acquire().await?;
                    store.read_range(seg, off, len).await
                }
            })
            .collect();
        
        futures::future::try_join_all(futures).await
    }
    
    fn coalesce_ranges(&self, ranges: &[(u64, u64)]) -> Vec<(u64, u64)> {
        let mut result = Vec::new();
        let mut current: Option<(u64, u64)> = None;
        
        for &(offset, len) in ranges {
            match current {
                None => current = Some((offset, len)),
                Some((cur_off, cur_len)) => {
                    let cur_end = cur_off + cur_len;
                    if offset <= cur_end + self.coalesce_threshold {
                        // Merge ranges
                        let new_end = (offset + len).max(cur_end);
                        current = Some((cur_off, new_end - cur_off));
                    } else {
                        // Gap too large, emit current and start new
                        result.push((cur_off, cur_len));
                        current = Some((offset, len));
                    }
                }
            }
        }
        
        if let Some(r) = current {
            result.push(r);
        }
        
        result
    }
}
```

### Concurrency Guidelines

| Backend | Recommended Concurrency | Notes |
|---------|------------------------|-------|
| Local NVMe | 32-64 | io_uring for best perf |
| S3/GCS | 64-128 | Per-connection limits |
| MinIO | 32-64 | Depends on deployment |

---

## SIMD MaxSim Implementation

### MaxSim Algorithm

```
MaxSim(Q, D) = Σ max(q · d) for each query token q, over all doc patches d
              q∈Q   d∈D
```

### Fast CPU Implementation (SQ8)

```rust
use std::arch::x86_64::*;

/// MaxSim with SQ8 patches using AVX2
pub fn maxsim_sq8_avx2(
    query_tokens: &[f32],    // [T × D], already normalized
    patch_codes: &[i8],      // [P × D]
    scales: &[f16],          // [P / block_size]
    dims: usize,
    n_patches: usize,
    block_size: usize,
) -> f32 {
    let n_tokens = query_tokens.len() / dims;
    let mut total_score = 0.0f32;
    
    for t in 0..n_tokens {
        let q = &query_tokens[t * dims..(t + 1) * dims];
        let mut max_sim = f32::NEG_INFINITY;
        
        for p in 0..n_patches {
            let block_idx = p / block_size;
            let scale = scales[block_idx].to_f32();
            
            let patch_start = p * dims;
            let patch = &patch_codes[patch_start..patch_start + dims];
            
            // Compute dot product with SIMD
            let sim = dot_f32_i8_avx2(q, patch) * scale;
            max_sim = max_sim.max(sim);
        }
        
        total_score += max_sim;
    }
    
    total_score
}

/// AVX2 dot product: f32 query × i8 codes
#[target_feature(enable = "avx2")]
unsafe fn dot_f32_i8_avx2(a: &[f32], b: &[i8]) -> f32 {
    let mut sum = _mm256_setzero_ps();
    
    for i in (0..a.len()).step_by(8) {
        // Load 8 f32 values
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        
        // Load 8 i8 values, convert to f32
        let vb_i8 = _mm_loadl_epi64(b.as_ptr().add(i) as *const __m128i);
        let vb_i16 = _mm256_cvtepi8_epi16(vb_i8);
        let vb_lo = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(vb_i16)));
        
        // Multiply and accumulate
        sum = _mm256_fmadd_ps(va, vb_lo, sum);
    }
    
    // Horizontal sum
    let sum128 = _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1));
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    _mm_cvtss_f32(sum32)
}
```

### ARM NEON (Apple Silicon)

```rust
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// MaxSim with SQ8 patches using NEON
#[cfg(target_arch = "aarch64")]
pub fn maxsim_sq8_neon(
    query_tokens: &[f32],
    patch_codes: &[i8],
    scales: &[f16],
    dims: usize,
    n_patches: usize,
    block_size: usize,
) -> f32 {
    let n_tokens = query_tokens.len() / dims;
    let mut total_score = 0.0f32;
    
    for t in 0..n_tokens {
        let q = &query_tokens[t * dims..(t + 1) * dims];
        let mut max_sim = f32::NEG_INFINITY;
        
        for p in 0..n_patches {
            let block_idx = p / block_size;
            let scale = scales[block_idx].to_f32();
            
            let patch_start = p * dims;
            let patch = &patch_codes[patch_start..patch_start + dims];
            
            let sim = unsafe { dot_f32_i8_neon(q, patch) } * scale;
            max_sim = max_sim.max(sim);
        }
        
        total_score += max_sim;
    }
    
    total_score
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_f32_i8_neon(a: &[f32], b: &[i8]) -> f32 {
    let mut sum = vdupq_n_f32(0.0);
    
    for i in (0..a.len()).step_by(4) {
        let va = vld1q_f32(a.as_ptr().add(i));
        
        // Load 4 i8, widen to i16, then to i32, then to f32
        let vb_i8 = vld1_s8(b.as_ptr().add(i));
        let vb_i16 = vmovl_s8(vb_i8);
        let vb_i32 = vmovl_s16(vget_low_s16(vb_i16));
        let vb_f32 = vcvtq_f32_s32(vb_i32);
        
        sum = vfmaq_f32(sum, va, vb_f32);
    }
    
    vaddvq_f32(sum)
}
```

### GPU Path (Optional, High-Throughput)

```rust
/// GPU MaxSim using cuBLAS GEMM
pub struct GpuMaxSimReranker {
    cublas: CudaBlas,
    stream: CudaStream,
    
    /// Pre-allocated buffers
    query_buffer: CudaBuffer<f32>,
    patch_buffer: CudaBuffer<i8>,
    result_buffer: CudaBuffer<f32>,
}

impl GpuMaxSimReranker {
    /// Batch MaxSim for multiple candidates
    /// 
    /// Returns scores for each candidate
    pub async fn batch_maxsim(
        &mut self,
        query_tokens: &[f32],        // [T × D]
        candidates: &[CandidatePatches],  // Vec of [P × D] matrices
    ) -> Vec<f32> {
        // 1. Upload query tokens
        self.query_buffer.copy_from_host(query_tokens)?;
        
        let mut scores = Vec::with_capacity(candidates.len());
        
        for candidate in candidates {
            // 2. Upload candidate patches (or use persistent buffer)
            self.patch_buffer.copy_from_host(&candidate.codes)?;
            
            // 3. GEMM: [T × D] × [D × P] = [T × P]
            //    Gives similarity of each query token to each patch
            self.cublas.gemm(
                &self.query_buffer,      // [T × D]
                &self.patch_buffer,      // [P × D], transposed
                &mut self.result_buffer, // [T × P]
            )?;
            
            // 4. Reduce: for each of T rows, find max across P columns
            //    Then sum the T max values
            let score = self.reduce_maxsim(&self.result_buffer)?;
            scores.push(score);
        }
        
        scores
    }
}
```

---

## Redis Command API

### FT.CREATEMV - Create Multi-Vector Index

```redis
FT.CREATEMV <index_name>
  ON HASH PREFIX <count> <prefix> [<prefix> ...]
  SCHEMA
    <pooled_field> VECTOR <algo> <attr_count> [<attr_name> <attr_value>]...
    <patches_field> MULTIVECTOR <attr_count> [<attr_name> <attr_value>]...
    [<field> <type> ...]
```

**Example**:

```redis
FT.CREATEMV colpali_index
  ON HASH PREFIX 1 page:
  SCHEMA
    # Pooled vector for ANN (Stage A)
    pooled VECTOR HNSW 6
      TYPE FLOAT32
      DIM 128
      DISTANCE_METRIC COSINE
    
    # Patch matrix for reranking (Stage B)
    patches MULTIVECTOR 10
      TYPE INT8              # SQ8 storage
      DIM 128
      MAX_PATCHES 1024
      CODEC SQ8
      BLOCK_SIZE 32
    
    # Metadata
    title TEXT
    page_num NUMERIC
    pdf_path TAG
```

### FT.ADDMV - Add Multi-Vector Document

```redis
FT.ADDMV <index_name> <doc_id>
  POOLED <pooled_blob>
  PATCHES <patches_blob>
  [CODEC <SQ8|FP16|PQ16>]
  [META <json_meta>]
  [FIELDS <field> <value> ...]
```

**Example**:

```redis
FT.ADDMV colpali_index page:doc123:7
  POOLED "\x00\x01\x02..."          # 128 × 4 bytes (float32)
  PATCHES "\x00\x01\x02..."         # Pre-encoded SQ8: 1024 × 128 bytes + scales
  CODEC SQ8
  FIELDS title "Architecture Overview" page_num 7 pdf_path "docs/manual.pdf"
```

### FT.SEARCHMV - Two-Stage Search with Rerank

```redis
FT.SEARCHMV <index_name> <query_text_or_*>
  KNN <k> @<pooled_field> $<query_pooled_param>
  RERANK MAXSIM @<patches_field> $<query_tokens_param>
  [LIMIT <offset> <num>]
  PARAMS <count> <name> <value> ...
  [RETURN <count> <field> ...]
  [DIALECT 2]
```

**Example**:

```redis
FT.SEARCHMV colpali_index "*"
  KNN 200 @pooled $query_pooled        # Stage A: top 200 candidates
  RERANK MAXSIM @patches $query_tokens # Stage B: rerank with MaxSim
  LIMIT 0 10                           # Return top 10
  PARAMS 4 
    query_pooled "\x00\x01..."         # 128 × 4 bytes (float32)
    query_tokens "\x00\x01..."         # 32 × 128 × 4 bytes (32 query tokens)
  RETURN 3 title page_num pdf_path
```

### Response Format

```
1) (integer) 10
2) "page:doc123:7"
3) 1) "maxsim_score"
   2) "0.847"
   3) "title"
   4) "Architecture Overview"
   5) "page_num"
   6) "7"
4) "page:doc456:3"
5) 1) "maxsim_score"
   2) "0.823"
   ...
```

---

## Benchmark Targets

### Stage A: ANN Search (HNSW on Pooled Vectors)

| Docs | p50 | p99 | QPS (8 cores) |
|------|-----|-----|---------------|
| 1M | 2ms | 10ms | 1,000 |
| 10M | 5ms | 25ms | 400 |
| 100M | 15ms | 50ms | 150 |

### Stage B: MaxSim Rerank

**Configuration**: K=200 candidates, P=1024 patches, T=32 query tokens, D=128

| Storage | CPU (8×AVX2) | Apple M2 | RTX 4090 |
|---------|--------------|----------|----------|
| Local mmap (SQ8) | 50ms | 35ms | 3ms |
| Warm cache (SQ8) | 55ms | 40ms | 5ms |
| Cold (S3 fetch) | 150-500ms | - | - |

### End-to-End (Stage A + Stage B)

| Scenario | p50 | p99 |
|----------|-----|-----|
| All hot (mmap) | 55ms | 80ms |
| Warm cache hit | 60ms | 100ms |
| Cold cache miss | 200ms | 600ms |

### Storage

| Codec | Per Doc | 1M Docs | 10M Docs |
|-------|---------|---------|----------|
| SQ8 | 130 KB | 130 GB | 1.3 TB |
| FP16 | 260 KB | 260 GB | 2.6 TB |
| PQ16 | 18 KB | 18 GB | 180 GB |

---

## Consequences

### Pros

✅ Works at enterprise scale with object storage backing
✅ Minimizes bytes moved via range GET + coalescing
✅ Maintains Redis-like O(1) doc lookup via doc index
✅ Codec flexibility for cost/quality tuning
✅ SIMD-optimized MaxSim for fast CPU reranking
✅ GPU acceleration path for high-throughput scenarios
✅ Immutable segments enable simple durability + cache

### Cons

⚠️ More complexity than "store vectors in KV DB"
⚠️ Requires careful caching and cache warming strategy
⚠️ Multi-vector MaxSim is inherently expensive (O(T × P × K))
⚠️ Cold cache misses add significant latency
⚠️ Compaction requires careful scheduling

---

## Implementation Plan

| Phase | Scope | Effort | Dependencies |
|-------|-------|--------|--------------|
| **1** | RVF2 segments + doc index (SQ8 only, no WAL) | 2 weeks | - |
| **2** | Two-stage retrieval wired through Redis API | 1 week | Phase 1 |
| **3** | SIMD MaxSim (AVX2 + NEON) | 1 week | Phase 2 |
| **4** | Async range prefetch for object store | 2 weeks | Phase 1-3 |
| **5** | WAL + compaction | 2 weeks | Phase 1-4 |
| **6** | FP16 codec, PQ16 cold tier | 1 week | Phase 1 |
| **7** | GPU MaxSim (optional) | 2 weeks | Phase 3, ADR-001 |

### Phase 1 Deliverables

- [ ] `rvf2/manifest.rs` - Manifest read/write (MessagePack)
- [ ] `rvf2/doc_index.rs` - Doc index with binary search
- [ ] `rvf2/segment.rs` - Segment read/write (SQ8)
- [ ] `rvf2/codec.rs` - SQ8 encode/decode
- [ ] `rvf2/mod.rs` - Public API

### Phase 2 Deliverables

- [ ] `FT.CREATEMV` command parsing
- [ ] `FT.ADDMV` command + segment writer
- [ ] `FT.SEARCHMV` command + two-stage flow

### Phase 3 Deliverables

- [ ] `rvf2/maxsim.rs` - SIMD MaxSim (AVX2 + NEON)
- [ ] Benchmarks vs scalar

### Phase 4 Deliverables

- [ ] `rvf2/object_store.rs` - S3/GCS trait + impl
- [ ] `rvf2/tiered_store.rs` - Hot/Warm/Cold tiering
- [ ] `rvf2/prefetch.rs` - Async prefetch coordinator

---

## References

1. [ColPali Paper](https://arxiv.org/abs/2407.01449) - Visual Document Retrieval with Vision Language Models
2. [ColBERT Paper](https://arxiv.org/abs/2004.12832) - Efficient and Effective Passage Search via Contextualized Late Interaction
3. [Qdrant Multi-Vector Tutorial](https://qdrant.tech/documentation/advanced-tutorials/pdf-retrieval-at-scale/)
4. [FAISS IVF-SQ](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
5. [MinIO Range Requests](https://min.io/docs/minio/linux/developers/go/API.html)

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2025-12-25 | Initial draft |


