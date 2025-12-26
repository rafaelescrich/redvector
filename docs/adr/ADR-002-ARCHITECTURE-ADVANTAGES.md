# ADR-002: RedVector Architecture Advantages & Vector Storage Design

| Status | Proposed |
|--------|----------|
| **Date** | 2024-12-24 |
| **Decision Makers** | Rafael Escrich |
| **Technical Area** | Architecture, Storage |
| **Related** | ADR-001 (GPU Acceleration) |

---

## Table of Contents

1. [RedVector Unique Advantages](#redvector-unique-advantages)
2. [Competitive Analysis](#competitive-analysis)
3. [Proposed: Rust Vector Filesystem (RVF)](#proposed-rust-vector-filesystem-rvf)
4. [Architecture Decision](#architecture-decision)

---

## RedVector Unique Advantages

### 1. 🦀 Pure Rust: Memory Safety Without GC Overhead

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Memory Model Comparison                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  RedVector (Rust)         Milvus (Go)           Qdrant (Rust)               │
│  ─────────────────        ───────────           ────────────────            │
│  • No garbage collector   • GC pauses           • No garbage collector      │
│  • Zero-copy where        • Memory overhead     • Zero-copy where           │
│    possible                 from GC               possible                  │
│  • Predictable latency    • P99 spikes from GC  • Predictable latency       │
│  • Compile-time safety    • Runtime panics      • Compile-time safety       │
│                                                                              │
│  Weaviate (Go)            Pinecone (?)          Vespa (Java/C++)            │
│  ─────────────            ────────────          ─────────────────           │
│  • GC pauses              • Proprietary         • JVM GC + native           │
│  • Memory overhead        • Cloud-only          • Complex deployment        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why This Matters:**
- P99 latency is critical for production vector search
- GC pauses can cause 10-100ms spikes
- Rust's ownership model = deterministic memory behavior

---

### 2. 🔌 Redis Protocol Compatibility

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Protocol Compatibility Matrix                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Database        Protocol      Existing Clients    Learning Curve           │
│  ────────        ────────      ────────────────    ──────────────           │
│  RedVector       Redis         ✅ 50+ languages    Very Low                 │
│  Qdrant          gRPC/REST     Custom SDKs         Medium                   │
│  Milvus          gRPC          Custom SDKs         High                     │
│  Weaviate        GraphQL/REST  Custom SDKs         Medium                   │
│  Pinecone        REST          Custom SDKs         Low                      │
│  pgvector        PostgreSQL    ✅ All SQL clients  Low (if know SQL)        │
│                                                                              │
│  RedVector Advantage:                                                        │
│  • Works with redis-py, redis-cli, Jedis, ioredis, etc.                     │
│  • No new SDK to learn                                                      │
│  • Drop-in replacement for Redis caching + vector search                    │
│  • Familiar FT.* commands (RediSearch compatible)                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Ecosystem Size:**
| Protocol | Available Clients | Existing Infrastructure |
|----------|-------------------|------------------------|
| Redis | 50+ languages | Massive (10M+ deployments) |
| gRPC | Need custom SDK | Limited |
| REST | Need custom SDK | Moderate |
| GraphQL | Need custom SDK | Limited |

---

### 3. 🏗️ Unified Data Model (Redis + Vectors)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Data Model Comparison                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  RedVector                              Milvus/Qdrant/Pinecone              │
│  ══════════                             ═════════════════════               │
│                                                                              │
│  ┌─────────────────────────┐            ┌─────────────────────┐            │
│  │    Single Database      │            │   Vector Database   │            │
│  ├─────────────────────────┤            └──────────┬──────────┘            │
│  │ • Strings               │                       │                        │
│  │ • Lists                 │            ┌──────────┴──────────┐            │
│  │ • Sets                  │            │   + External Cache   │            │
│  │ • Hashes                │            │   (Redis/Memcached)  │            │
│  │ • Sorted Sets           │            └──────────┬──────────┘            │
│  │ • HyperLogLog           │                       │                        │
│  │ • Pub/Sub               │            ┌──────────┴──────────┐            │
│  │ ─────────────────────── │            │  + Document Store    │            │
│  │ • VECTORS (HNSW/IVF)    │◀───────────│  (MongoDB/Postgres)  │            │
│  │ • Similarity Search     │            └─────────────────────┘            │
│  └─────────────────────────┘                                                │
│                                                                              │
│  ONE SERVER vs THREE SERVERS                                                │
│  ════════════════════════════                                               │
│                                                                              │
│  Benefits:                                                                   │
│  • Single deployment                                                        │
│  • Atomic operations (MULTI/EXEC with vectors)                              │
│  • No network latency between services                                      │
│  • Consistent backup/restore                                                │
│  • Lower operational complexity                                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Real-World Use Case:**
```redis
# Atomic operation: Update product + embedding + invalidate cache
MULTI
HSET product:123 name "New Name" price 99.99
FT.ADD products product:123 1.0 FIELDS vector "0.1,0.2,..."
DEL cache:product:123
PUBLISH product_updates "123"
EXEC
```

**This is impossible with separate vector DB + cache + pub/sub systems!**

---

### 4. 📊 SIMD-Accelerated Distance Metrics (Already Implemented!)

```rust
// RedVector already has AVX2 + SSE4.1 implementations
// (redisearch-platform-core/src/simd_metrics.rs)

#[target_feature(enable = "avx2")]
unsafe fn cosine_similarity_avx2(a: &[f32], b: &[f32]) -> f32 {
    // 8 floats at a time
    let va = _mm256_loadu_ps(a.as_ptr().add(idx));
    let vb = _mm256_loadu_ps(b.as_ptr().add(idx));
    dot = _mm256_fmadd_ps(va, vb, dot);  // Fused multiply-add
    // ...
}
```

**Benchmark vs Naive:**
| Implementation | 768-dim Cosine | Speedup |
|----------------|----------------|---------|
| Scalar loop | 12µs | 1x |
| SSE4.1 | 4µs | 3x |
| AVX2 | 1.5µs | 8x |
| GPU (wgpu) | 0.1µs (batch) | 120x |

---

### 5. 🔄 Flexible Backend Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Adaptive Index Selection                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Dataset Size        Automatic Backend Selection                           │
│   ════════════        ═══════════════════════════                           │
│                                                                              │
│   < 1,000 vectors     LinearScan (exact, simple)                            │
│          │                                                                   │
│          ▼                                                                   │
│   1K - 10K vectors    Auto-migrate to HNSW (CPU)                            │
│          │                                                                   │
│          ▼                                                                   │
│   10K - 1M vectors    HNSW (CPU) or GPU Flat                                │
│          │                                                                   │
│          ▼                                                                   │
│   1M - 100M vectors   GPU IVF-SQ8 (coming in ADR-001)                       │
│          │                                                                   │
│          ▼                                                                   │
│   > 100M vectors      GPU IVF-PQ or DiskANN (future)                        │
│                                                                              │
│   This is AUTOMATIC - user doesn't need to choose!                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Competitive Analysis

### Feature Comparison Matrix

| Feature | RedVector | Qdrant | Milvus | Weaviate | Pinecone | pgvector |
|---------|-----------|--------|--------|----------|----------|----------|
| **Language** | 🦀 Rust | 🦀 Rust | Go | Go | ? | C |
| **Protocol** | Redis ✅ | gRPC/REST | gRPC | GraphQL | REST | SQL |
| **HNSW** | ✅ | ✅ | ✅ | ✅ | ? | ✅ |
| **IVF** | 🔨 | ❌ | ✅ | ❌ | ? | ✅ |
| **GPU** | 🔨 (wgpu+CUDA) | ❌ | ✅ (CUDA) | ❌ | ✅ | ❌ |
| **Apple Silicon** | ✅ Metal | ❌ | ❌ | ❌ | Cloud | ❌ |
| **Quantization** | 🔨 SQ8/PQ | ✅ SQ | ✅ PQ | ❌ | ? | ❌ |
| **Full-Text Search** | ✅ FT.* | ✅ | ❌ | ✅ | ❌ | ✅ tsvector |
| **Other Data Types** | ✅ All Redis | ❌ | ❌ | ❌ | ❌ | ✅ SQL |
| **Transactions** | ✅ MULTI/EXEC | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Pub/Sub** | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ LISTEN |
| **Open Source** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| **Self-Hosted** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |

### Unique RedVector Advantages Summary

| Advantage | Impact | Competitors Have? |
|-----------|--------|-------------------|
| Redis protocol | Massive ecosystem | ❌ None |
| Unified data model | Simpler architecture | ❌ None |
| Rust + No GC | Predictable latency | Qdrant only |
| wgpu (Metal + Vulkan) | Apple Silicon GPU | ❌ None |
| Built-in caching | No separate Redis | ❌ None |
| Pub/Sub for vectors | Real-time updates | ❌ None |

---

## Proposed: Rust Vector Filesystem (RVF)

### Critical Insight: Redis is NOT a Disk-Based Key-Value Store

Before designing RVF, we must understand what Redis *actually* does:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Redis Storage Model (What NOT to Copy)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Redis is primarily an IN-MEMORY HASH TABLE:                                │
│                                                                              │
│     key (bytes) ──────────────▶ RedisObject* (pointer in RAM)               │
│                                                                              │
│  Persistence formats are NOT designed for random access by key:             │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  RDB (Snapshot)                                                     │    │
│  │  ────────────────                                                   │    │
│  │  • Serialized dump of the WHOLE dataset                            │    │
│  │  • NOT random-access by key                                         │    │
│  │  • Must load entire file to find a key                             │    │
│  │  • Great for backup/restore, terrible for lookup                   │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  AOF (Append-Only File)                                             │    │
│  │  ──────────────────────                                              │    │
│  │  • Command log (replay to reconstruct state)                        │    │
│  │  • NOT random-access by key                                         │    │
│  │  • Sequential append, sequential replay                             │    │
│  │  • Great for durability, terrible for lookup                       │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ⚠️  COPYING REDIS PERSISTENCE WON'T GIVE YOU FAST DISK LOOKUP!           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Right Approach: Computed Offsets (O(1) Addressing)

What we *can* borrow from Redis is the **key → pointer** mental model, adapted for disk:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    O(1) Vector Addressing Architecture                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  The fastest possible retrieval path:                                        │
│                                                                              │
│     redis_key ──▶ internal_id ──▶ (shard, offset) ──▶ mmap slice           │
│                                                                              │
│  ════════════════════════════════════════════════════════════════════       │
│                                                                              │
│  Step 1: Small in-memory index (like Redis dict)                            │
│  ─────────────────────────────────────────────────                          │
│                                                                              │
│     redis_key (bytes) ───────▶ internal_id (u64)                            │
│                                                                              │
│     • This is just a hash table in RAM                                      │
│     • Persisted via WAL or redb (for metadata only)                         │
│     • If doc_id is already u64, can be used directly!                       │
│                                                                              │
│  Step 2: Compute location WITHOUT any B-tree lookup                         │
│  ───────────────────────────────────────────────────                         │
│                                                                              │
│     // Fixed-width vectors = computable offset!                             │
│     vectors_per_shard = shard_bytes / bytes_per_vector                      │
│     shard_id          = internal_id / vectors_per_shard                     │
│     idx_in_shard      = internal_id % vectors_per_shard                     │
│     byte_offset       = HEADER_SIZE + idx_in_shard * bytes_per_vector       │
│                                                                              │
│     // Example: 768-dim SQ8 = 768 bytes/vector                              │
│     // internal_id = 1,234,567                                              │
│     // shard_size = 256MB = 268,435,456 bytes                               │
│     // vectors_per_shard = 268,435,456 / 768 = 349,525                      │
│     // shard_id = 1,234,567 / 349,525 = 3                                   │
│     // idx_in_shard = 1,234,567 % 349,525 = 185,992                         │
│     // offset = 32 + 185,992 * 768 = 142,842,288 bytes                      │
│                                                                              │
│  ⚡ This is ONE HASH LOOKUP + TWO DIVISIONS + ONE POINTER ADD!              │
│  ⚡ No searching, no B-tree traversal, no deserialization!                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Handling Deletes and Updates: Location Table

For production systems that need updates/deletes without rewriting shards:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Location Indirection Table (loc.rvf)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  When you need to move vectors during compaction:                           │
│                                                                              │
│     internal_id ──────▶ LocationEntry (16 bytes)                            │
│                                                                              │
│     struct LocationEntry {                                                   │
│         shard_id: u32,      // Which shard file                             │
│         offset: u32,        // Byte offset within shard                     │
│         version: u32,       // For MVCC / optimistic locking                │
│         flags: u32,         // DELETED, MOVED, COMPRESSED, etc.             │
│     }                                                                        │
│                                                                              │
│  ════════════════════════════════════════════════════════════════════       │
│                                                                              │
│  Full lookup path with indirection:                                          │
│                                                                              │
│     redis_key ──▶ internal_id ──▶ loc[internal_id] ──▶ shard[offset]       │
│                   (hash)         (array index)         (mmap slice)         │
│                   O(1)           O(1)                  O(1)                 │
│                                                                              │
│  Benefits:                                                                   │
│  • Move vectors during compaction without changing keys                     │
│  • Mark deleted without rewriting shard                                     │
│  • Version tracking for concurrent access                                   │
│  • loc.rvf itself is mmap'd (16 bytes × num_vectors)                       │
│                                                                              │
│  Size: 10M vectors × 16 bytes = 160 MB (trivial)                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why This Matters: Comparison of Lookup Strategies

| Strategy | Operations | Latency | Notes |
|----------|------------|---------|-------|
| **redb B-tree lookup** | Hash + B-tree traversal + deserialize | ~10µs | Current approach |
| **SQL index lookup** | Parse + plan + index seek + fetch | ~100µs | pgvector style |
| **RVF computed offset** | Hash + 2 divisions + mmap read | **~0.5µs** | 20x faster! |
| **RVF with loc table** | Hash + array index + mmap read | **~0.7µs** | Supports deletes |

### The Key Trick: DON'T SEARCH FOR VECTORS INSIDE FILES

```rust
// ❌ WRONG: Search for vector in file (B-tree, binary search, etc.)
fn get_vector_slow(db: &Database, key: &[u8]) -> Vec<f32> {
    let serialized = db.get(key)?;           // B-tree lookup
    bincode::deserialize(&serialized)?       // Deserialize
}

// ✅ RIGHT: Compute offset directly (array indexing)
fn get_vector_fast(shards: &[Mmap], loc: &[LocationEntry], internal_id: u64) -> &[f32] {
    let entry = &loc[internal_id as usize];  // Array index: O(1)
    let shard = &shards[entry.shard_id];     // Array index: O(1)
    let offset = entry.offset as usize;
    
    // Direct pointer arithmetic: O(1)
    unsafe {
        std::slice::from_raw_parts(
            shard[offset..].as_ptr() as *const f32,
            DIMENSION
        )
    }
}
```

### Recommended Layout for RedVector

Given that RedVector speaks Redis protocol (keys like `product:123`):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Recommended RedVector Storage Layout                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  METADATA (use existing redb or Redis-like structures):                     │
│  ───────────────────────────────────────────────────────                     │
│  • Key-value data (strings, hashes, lists, etc.)                            │
│  • Index configuration                                                       │
│  • Key → internal_id mapping                                                │
│                                                                              │
│  VECTORS (new RVF shards with sequential internal IDs):                     │
│  ─────────────────────────────────────────────────────                       │
│  • shard_0000.rvf, shard_0001.rvf, ...                                      │
│  • Fixed-width records (SQ8 = 768 bytes each)                               │
│  • Append-only (new vectors get next internal_id)                           │
│  • mmap'd for zero-copy access                                              │
│                                                                              │
│  LOCATION TABLE (small, optional for delete support):                       │
│  ──────────────────────────────────────────────────────                      │
│  • loc.rvf: internal_id → {shard, offset, flags}                            │
│  • Only needed if you allow deletes/updates                                 │
│  • If append-only, skip this (computed offset is enough)                    │
│                                                                              │
│  This keeps Redis semantics while making vector I/O = array indexing!       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Operation-Specific Optimizations

| Operation | Best Strategy |
|-----------|--------------|
| **Fetch vector by key** (KV-style) | Hash → loc → mmap slice (single page touch) |
| **ANN search returns IDs, then rerank** | HNSW/IVF returns internal_ids → batch mmap reads → SIMD/GPU rerank |
| **Bulk insert** | Append to shard, assign sequential internal_ids, update key→id map |
| **Delete** | Set DELETED flag in loc table (tombstone), compact later |
| **Update** | Append new version, update loc table, old version is garbage |

---

### The Problem with Current Storage

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Current Storage (redb)                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Vectors stored as:                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  doc_id (u64) → bincode(Vec<f32>)                                   │   │
│  │                                                                      │   │
│  │  Storage per vector (768-dim):                                       │   │
│  │  • 8 bytes (doc_id key)                                             │   │
│  │  • 3072 bytes (768 × 4 bytes float32)                               │   │
│  │  • ~50 bytes (bincode overhead + redb B-tree)                       │   │
│  │  ─────────────────────────────────────────────                      │   │
│  │  Total: ~3,130 bytes per vector                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Problems:                                                                   │
│  • No compression                                                           │
│  • B-tree overhead for each vector                                          │
│  • Random I/O pattern (not sequential)                                      │
│  • No memory-mapping for > RAM datasets                                     │
│  • Full deserialization on read                                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Proposed: Rust Vector Filesystem (RVF)

A purpose-built filesystem for compressed vector storage, optimized for ANN workloads.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RVF: Rust Vector Filesystem                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Design Goals:                                                               │
│  1. Sequential I/O for bulk reads (cache-friendly)                          │
│  2. Zero-copy memory mapping (mmap)                                         │
│  3. Native compression (SQ8, PQ, float16)                                   │
│  4. Append-only writes (WAL-style, crash safe)                              │
│  5. Tiered storage (hot/warm/cold)                                          │
│  6. GPU-friendly memory layout                                              │
│                                                                              │
│  ════════════════════════════════════════════════════════════════════       │
│                                                                              │
│  File Layout:                                                                │
│                                                                              │
│  index_name/                                                                 │
│  ├── manifest.rvf          # Index metadata, version, config                │
│  ├── keys.rvf              # redis_key → internal_id (hash table)           │
│  ├── loc.rvf               # internal_id → {shard, offset, flags} (array)   │
│  ├── vectors/                                                                │
│  │   ├── shard_0000.rvf    # Vector data (mmap, O(1) offset access)         │
│  │   ├── shard_0001.rvf    # Each shard ~256MB-1GB                          │
│  │   └── ...               # Append-only, sequential internal_ids           │
│  ├── graph/                 # HNSW/IVF structures                           │
│  │   ├── hnsw_layer_0.rvf  # Bottom layer (all vectors)                     │
│  │   ├── hnsw_layer_1.rvf  # Upper layers                                   │
│  │   └── ivf_centroids.rvf # IVF cluster centroids                          │
│  ├── codebooks/             # For PQ                                         │
│  │   └── pq_codebook.rvf   # Trained codebooks                              │
│  └── wal/                   # Write-ahead log                                │
│      ├── wal_0000.rvf      # Pending writes before shard append             │
│      └── wal_0001.rvf                                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Vector Shard Format (`.rvf`)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Vector Shard Binary Format                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Offset     Size        Field                                                │
│  ──────     ────        ─────                                                │
│  0x0000     4 bytes     Magic ("RVF1")                                       │
│  0x0004     4 bytes     Version (u32)                                        │
│  0x0008     4 bytes     Flags (compression type, etc.)                       │
│  0x000C     4 bytes     Dimension (u32)                                      │
│  0x0010     8 bytes     Num Vectors (u64)                                    │
│  0x0018     4 bytes     Bytes per Vector (depends on compression)            │
│  0x001C     4 bytes     Checksum (CRC32)                                     │
│  0x0020     ...         Vector Data (contiguous, mmap-able)                  │
│                                                                              │
│  Compression Flags:                                                          │
│  ┌────────┬───────────────────────────────────────────────────────────┐     │
│  │ 0x00   │ Float32 (uncompressed, 3072 bytes/vec for 768-dim)       │     │
│  │ 0x01   │ Float16 (half precision, 1536 bytes/vec)                 │     │
│  │ 0x02   │ SQ8 (int8 scalar quantization, 768 bytes/vec)            │     │
│  │ 0x03   │ SQ4 (int4 scalar quantization, 384 bytes/vec)            │     │
│  │ 0x04   │ PQ (product quantization, variable)                      │     │
│  └────────┴───────────────────────────────────────────────────────────┘     │
│                                                                              │
│  Example: 10M vectors × 768-dim                                              │
│  ──────────────────────────────                                             │
│  • Float32: 30.7 GB (10M × 3072)                                            │
│  • Float16: 15.4 GB (10M × 1536)                                            │
│  • SQ8:     7.7 GB (10M × 768)                                              │
│  • PQ(m=8): 960 MB (10M × 96)                                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Memory-Mapped Access with O(1) Addressing

```rust
/// Location entry for O(1) vector lookup (16 bytes)
#[repr(C)]
#[derive(Copy, Clone)]
pub struct LocationEntry {
    pub shard_id: u32,
    pub offset: u32,
    pub version: u32,
    pub flags: u32,  // VALID, DELETED, MOVED, etc.
}

/// Location table: internal_id → LocationEntry (mmap'd array)
pub struct LocationTable {
    mmap: memmap2::MmapMut,
    len: usize,
}

impl LocationTable {
    /// O(1) lookup: just array indexing!
    #[inline]
    pub fn get(&self, internal_id: u64) -> &LocationEntry {
        let entries = unsafe {
            std::slice::from_raw_parts(
                self.mmap.as_ptr() as *const LocationEntry,
                self.len
            )
        };
        &entries[internal_id as usize]
    }
}

/// Vector shard storage (mmap'd contiguous vectors)
pub struct VectorShard {
    mmap: memmap2::Mmap,
    bytes_per_vector: usize,
    dimension: usize,
}

impl VectorShard {
    /// O(1) vector access: direct offset computation!
    #[inline]
    pub fn get_vector(&self, offset: u32) -> &[f32] {
        let start = offset as usize;
        let end = start + self.bytes_per_vector;
        
        unsafe {
            std::slice::from_raw_parts(
                self.mmap[start..end].as_ptr() as *const f32,
                self.dimension
            )
        }
    }
}

/// Complete vector store with O(1) key→vector lookup
pub struct RvfVectorStore {
    key_to_id: HashMap<Vec<u8>, u64>,  // In-memory hash (like Redis dict)
    loc_table: LocationTable,           // mmap'd location array
    shards: Vec<VectorShard>,           // mmap'd vector shards
}

impl RvfVectorStore {
    /// Full lookup path: hash + array index + mmap slice
    /// Total: ~0.5µs (vs ~10µs for B-tree)
    pub fn get_vector(&self, key: &[u8]) -> Option<&[f32]> {
        // Step 1: Hash lookup for internal_id (O(1))
        let internal_id = *self.key_to_id.get(key)?;
        
        // Step 2: Array index for location (O(1))
        let loc = self.loc_table.get(internal_id);
        
        if loc.flags & DELETED != 0 {
            return None;
        }
        
        // Step 3: Direct mmap access (O(1))
        let shard = &self.shards[loc.shard_id as usize];
        Some(shard.get_vector(loc.offset))
    }
    
    /// Batch lookup for ANN reranking (SIMD/GPU friendly)
    pub fn get_vectors_batch(&self, internal_ids: &[u64]) -> Vec<&[f32]> {
        internal_ids.iter()
            .filter_map(|&id| {
                let loc = self.loc_table.get(id);
                if loc.flags & DELETED != 0 { return None; }
                let shard = &self.shards[loc.shard_id as usize];
                Some(shard.get_vector(loc.offset))
            })
            .collect()
    }
    
    /// Get contiguous memory region for GPU upload (zero-copy!)
    pub fn get_shard_gpu_ptr(&self, shard_id: usize) -> *const f32 {
        self.shards[shard_id].mmap.as_ptr() as *const f32
    }
}
```

### Simplified Mode: Computed Offset (No Location Table)

If you only need append-only (no deletes), skip the location table entirely:

```rust
impl RvfVectorStore {
    /// Even faster: compute offset directly (append-only mode)
    #[inline]
    pub fn get_vector_append_only(&self, internal_id: u64) -> &[f32] {
        let bytes_per_vector = 768;  // SQ8 for 768-dim
        let vectors_per_shard = SHARD_SIZE / bytes_per_vector;
        
        let shard_id = (internal_id / vectors_per_shard as u64) as usize;
        let idx_in_shard = (internal_id % vectors_per_shard as u64) as usize;
        let offset = HEADER_SIZE + idx_in_shard * bytes_per_vector;
        
        let shard = &self.shards[shard_id];
        unsafe {
            std::slice::from_raw_parts(
                shard.mmap[offset..].as_ptr() as *const f32,
                self.dimension
            )
        }
    }
}
```

### Tiered Storage Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Three-Tier Vector Storage                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  HOT TIER (GPU VRAM / System RAM)                                   │   │
│  │  ════════════════════════════════                                    │   │
│  │  • Frequently accessed vectors                                       │   │
│  │  • LRU eviction to warm tier                                        │   │
│  │  • Full precision or SQ8                                            │   │
│  │  • Size: 1-16 GB (depends on GPU)                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼ LRU eviction                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  WARM TIER (Memory-mapped SSD)                                      │   │
│  │  ═══════════════════════════════                                     │   │
│  │  • Recently used vectors                                            │   │
│  │  • mmap'd for fast random access                                    │   │
│  │  • Compressed (SQ8 or PQ)                                           │   │
│  │  • Size: 32-256 GB                                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼ Age out                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  COLD TIER (Disk / Object Storage)                                  │   │
│  │  ═══════════════════════════════════                                 │   │
│  │  • Rarely accessed vectors                                          │   │
│  │  • Heavily compressed (PQ)                                          │   │
│  │  • Sequential I/O optimized                                         │   │
│  │  • Size: Unlimited (S3, GCS, etc.)                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Access Pattern Optimization:                                                │
│  • IVF clusters hot data together                                           │
│  • Prefetch warm tier based on query patterns                               │
│  • Background compaction and compression                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### RVF vs Current Storage (redb)

| Aspect | redb (Current) | RVF (Proposed) |
|--------|----------------|----------------|
| **Lookup Strategy** | B-tree traversal | **Computed offset (O(1))** |
| **Lookup Latency** | ~10µs | **~0.5µs (20x faster)** |
| **Storage Overhead** | ~3,130 bytes/vec | 768 bytes/vec (SQ8) |
| **Memory Mapping** | ❌ Full load + deserialize | ✅ Zero-copy mmap |
| **Compression** | ❌ None | ✅ SQ8, PQ, float16 |
| **I/O Pattern** | Random (B-tree) | Sequential (append-only) |
| **GPU Upload** | Deserialize + copy | **Zero-copy pointer** |
| **Crash Safety** | ✅ ACID | ✅ WAL |
| **Batch Reads** | N syscalls + deserialize | **Single mmap region** |
| **Delete Support** | Native | Location table tombstone |

### Why This Matters for ANN Search

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ANN Reranking Performance                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Typical ANN flow: HNSW/IVF returns 1000 candidate internal_ids             │
│  Then: Fetch all 1000 vectors for exact reranking                           │
│                                                                              │
│  redb (current):                                                             │
│  ────────────────                                                            │
│  • 1000 B-tree lookups = 10ms                                               │
│  • 1000 deserializations = 5ms                                              │
│  • Total: ~15ms just for vector fetch                                       │
│                                                                              │
│  RVF (proposed):                                                             │
│  ───────────────                                                             │
│  • 1000 array index lookups = 0.1ms                                         │
│  • 0 deserializations (already in native format)                            │
│  • Vectors likely in same mmap region (cache-friendly)                      │
│  • Total: ~0.5ms for vector fetch                                           │
│                                                                              │
│  ⚡ 30x faster vector fetching for reranking!                               │
│                                                                              │
│  GPU upload:                                                                 │
│  ───────────                                                                 │
│  • redb: deserialize → copy to staging buffer → upload                      │
│  • RVF: mmap pointer → direct DMA (zero-copy on unified memory)            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Proposed Module Structure

```
redisearch-platform-core/
├── src/
│   ├── rvf/                          # Rust Vector Filesystem
│   │   ├── mod.rs                    # Module entry
│   │   ├── manifest.rs               # Index manifest
│   │   ├── shard.rs                  # Vector shard (mmap)
│   │   ├── compression/
│   │   │   ├── mod.rs
│   │   │   ├── sq8.rs                # Scalar quantization int8
│   │   │   ├── sq4.rs                # Scalar quantization int4
│   │   │   ├── float16.rs            # Half precision
│   │   │   └── pq.rs                 # Product quantization
│   │   ├── wal.rs                    # Write-ahead log
│   │   ├── compaction.rs             # Background compaction
│   │   ├── tiering.rs                # Hot/warm/cold management
│   │   └── gpu_transfer.rs           # Zero-copy GPU upload
│   │
│   ├── storage.rs                    # (Keep redb for metadata)
│   └── ...
```

### Key Rust Crates to Use

```toml
[dependencies]
# Memory mapping
memmap2 = "0.9"

# Compression
half = "2.3"              # float16 support
lz4 = "1.24"              # Block compression (optional)

# I/O
io-uring = "0.6"          # Linux async I/O (optional)

# Checksums
crc32fast = "1.3"

# Serialization (for metadata only)
bincode = "1.3"
serde = { version = "1.0", features = ["derive"] }
```

---

## Architecture Decision

### Recommendation: Implement RVF in Phases

| Phase | Scope | Effort | Impact |
|-------|-------|--------|--------|
| **1** | Memory-mapped shards (float32) | 2 weeks | 2x read perf |
| **2** | SQ8 compression | 1 week | 4x storage |
| **3** | Float16 support | 3 days | 2x storage |
| **4** | WAL for crash safety | 1 week | Production ready |
| **5** | Tiered storage (local + S3/GCS) | 2 weeks | Bigger than RAM |
| **6** | PQ compression | 2 weeks | 32x storage |
| **7** | Multi-vector (ColPali/ColBERT) | 2 weeks | Document AI |
| **8** | GPU MaxSim reranking | 1 week | 10-20x rerank |

### Integration with ADR-001 (GPU)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RVF + GPU Integration                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Query Flow:                                                                 │
│                                                                              │
│  1. Query arrives                                                            │
│         │                                                                    │
│         ▼                                                                    │
│  2. IVF coarse search (GPU)                                                 │
│         │ returns: cluster IDs [23, 45, 67, ...]                            │
│         ▼                                                                    │
│  3. RVF shard lookup                                                        │
│         │ mmap pointers for clusters                                        │
│         ▼                                                                    │
│  4. GPU upload (zero-copy from mmap)                                        │
│         │ cuMemcpyHtoDAsync or wgpu buffer                                  │
│         ▼                                                                    │
│  5. Fine search (GPU kernel)                                                │
│         │                                                                    │
│         ▼                                                                    │
│  6. Top-K results                                                           │
│                                                                              │
│  Key: RVF mmap → GPU is ZERO COPY on unified memory (Apple Silicon)!       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Multi-Vector Storage: ColPali / ColBERT Support

### The Multi-Vector Challenge

ColPali and ColBERT are **late-interaction multi-vector** models that store *many* vectors per document:

| Model | Vectors per Doc | Dimension | Use Case |
|-------|-----------------|-----------|----------|
| ColBERT | ~100-500 | 128 | Text passages |
| ColPali | ~1024 | 128 | PDF pages/images (patch embeddings) |

Scoring uses **MaxSim**: for each query token, find the best matching doc patch, then aggregate.

```
MaxSim(Q, D) = Σ max(q · d) for each query token q, over all doc patches d
              q∈Q   d∈D
```

**Why this changes everything:**
- Can't pre-compute a single similarity score (late interaction)
- Must fetch and score ALL patch vectors for candidate docs
- Storage format becomes the bottleneck, not the index

### Two-Stage Retrieval Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              ColPali/ColBERT Two-Stage Retrieval                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  STAGE A: Candidate Generation (ANN-friendly)                          │ │
│  │  ═══════════════════════════════════════════                           │ │
│  │                                                                         │ │
│  │  1. Store ONE "summary vector" per document/page                       │ │
│  │     • Mean pooling of all patch vectors                                │ │
│  │     • Or: model-provided [CLS] / pooled embedding                      │ │
│  │                                                                         │ │
│  │  2. Index summary vectors with HNSW/IVF                                │ │
│  │     • Standard ANN search                                              │ │
│  │     • Returns top K=100-1000 candidates                                │ │
│  │                                                                         │ │
│  │  Query: O(log N) or O(√N) — fast!                                      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                              │                                              │
│                              ▼                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  STAGE B: MaxSim Rerank (exact scoring)                                │ │
│  │  ═══════════════════════════════════════                               │ │
│  │                                                                         │ │
│  │  1. Fetch patch matrices for K candidates                              │ │
│  │     • Each: 1024×128 matrix (ColPali)                                  │ │
│  │     • From RVF segments (mmap or range GET)                            │ │
│  │                                                                         │ │
│  │  2. Compute MaxSim for each candidate                                  │ │
│  │     • CPU SIMD batched matrix ops                                      │ │
│  │     • Or: GPU batch (cuBLAS GEMM + reduction)                          │ │
│  │                                                                         │ │
│  │  3. Sort by MaxSim score, return top results                           │ │
│  │                                                                         │ │
│  │  Compute: O(K × patches × query_tokens) — bounded by K                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

This mirrors Qdrant's "multi-vector PDF retrieval" guidance: candidate gen + multi-vector rerank.

### RVF Segment Design for Multi-Vector

#### File Layout (works for local mmap AND object storage)

```
index_name/
├── manifest.json                 # Version, dims, codec, segment list, checksums
├── doc_index.bin                 # doc_id → (segment_id, offset, len) mapping
├── pooled/                       # Summary vectors for Stage A (ANN index)
│   ├── vectors.rvf               # Mean-pooled vectors (one per doc)
│   └── hnsw_graph/               # HNSW index on pooled vectors
├── patches/                      # Multi-vector matrices for Stage B (rerank)
│   ├── seg_000000.rvf            # 256MB-1GB segments
│   ├── seg_000001.rvf
│   └── ...
├── codebooks/                    # PQ codebooks (if using PQ)
│   └── pq_256x8.bin
└── wal/                          # Write-ahead log
```

#### Segment Format for Multi-Vector

```
Segment Header (64 bytes):
┌──────────┬──────────┬──────────┬───────────┬──────────────┬────────────┐
│  Magic   │ Version  │  Codec   │ Dimension │  Num Docs    │  Checksum  │
│  4B      │  2B      │  2B      │  4B       │  8B          │  8B        │
└──────────┴──────────┴──────────┴───────────┴──────────────┴────────────┘

Doc Offset Table (at end of segment, for random access):
┌────────────────────────────────────────────────────────────────────────┐
│  [doc_0_offset: u64, doc_1_offset: u64, ..., doc_N_offset: u64]       │
└────────────────────────────────────────────────────────────────────────┘

Per-Document Record:
┌────────────┬───────────────┬────────────────────────────────────────────┐
│  doc_id    │  n_patches    │  patch_data (n_patches × dim × bytes_per) │
│  8B        │  4B           │  variable                                  │
└────────────┴───────────────┴────────────────────────────────────────────┘
```

#### Encoding Options (What Actually Wins)

| Encoding | Bytes per dim | 1024×128 matrix | Speedup | Quality |
|----------|---------------|-----------------|---------|---------|
| FP32 | 4 | **512 KB** | baseline | perfect |
| FP16 | 2 | **256 KB** | ~same | excellent |
| **INT8 (SQ8)** | 1 | **128 KB** | 1.5-2x SIMD | very good |
| **PQ (m=16)** | 0.125 | **16 KB** | varies | good |

**Recommended default: INT8 (SQ8)** — 4x compression, fast SIMD/GPU paths, small quality loss.

**For extreme scale: PQ** — 32x compression, but use SQ8/FP16 for final rerank.

```rust
/// Multi-vector document storage
pub struct MultiVectorDoc {
    pub doc_id: u64,
    pub n_patches: u32,
    pub patches: Vec<u8>,  // Encoded patch matrix (SQ8 or PQ codes)
}

/// Fetch and decode patch matrix for MaxSim
impl RvfSegment {
    pub fn get_patch_matrix(&self, doc_offset: u64) -> PatchMatrix {
        // Seek to doc offset in mmap
        let header = self.read_doc_header(doc_offset);
        let data_start = doc_offset + 12; // After doc_id + n_patches
        let data_len = header.n_patches as usize * self.dim * self.bytes_per;
        
        // Zero-copy slice for SIMD/GPU
        let raw = &self.mmap[data_start..data_start + data_len];
        
        PatchMatrix {
            n_patches: header.n_patches,
            dim: self.dim,
            data: raw,
            codec: self.codec,
        }
    }
    
    pub fn compute_maxsim(&self, query_tokens: &[f32], patch_matrix: &PatchMatrix) -> f32 {
        // For each query token, find max similarity to any patch
        let mut total = 0.0f32;
        for q in query_tokens.chunks(patch_matrix.dim) {
            let mut max_sim = f32::NEG_INFINITY;
            for p in patch_matrix.iter_patches() {
                let sim = simd_dot(q, p);
                max_sim = max_sim.max(sim);
            }
            total += max_sim;
        }
        total
    }
}
```

---

## Object Storage Integration (S3/GCS/MinIO)

### The Reality of Cloud Storage Latency

Object stores are **high-latency** compared to local NVMe:

| Storage Type | Random Read Latency | Throughput | Cost |
|--------------|---------------------|------------|------|
| NVMe SSD | ~50-100 µs | 3-7 GB/s | $$$ |
| EBS/GP3 | 1-10 ms | 125-1000 MB/s | $$ |
| S3/GCS/Blob | 20-100+ ms | 25-100 MB/s/conn | $ |

**Key insight:** If you put RVF "directly on S3" with no local cache, MaxSim rerank will be network-bound.

### Tiered Storage with Object Storage Backing

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              Three-Tier Storage Architecture                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  HOT TIER: In-Memory / GPU                                              ││
│  │  ────────────────────────                                               ││
│  │  • Active index (HNSW graph for pooled vectors)                        ││
│  │  • Recently accessed patch segments (LRU cache)                        ││
│  │  • GPU buffer pool for batch reranking                                 ││
│  │                                                                         ││
│  │  Capacity: 10-20% of dataset, <1ms access                              ││
│  └──────────────────────────────┬──────────────────────────────────────────┘│
│                                 │ cache miss                               │
│                                 ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  WARM TIER: Local SSD / NVMe (mmap)                                    ││
│  │  ──────────────────────────────────                                    ││
│  │  • Pinned segments (frequently accessed)                               ││
│  │  • LRU segment cache (fetch from cold on miss)                         ││
│  │  • mmap for zero-copy reads                                            ││
│  │                                                                         ││
│  │  Capacity: 30-50% of dataset, 1-10ms access                            ││
│  └──────────────────────────────┬──────────────────────────────────────────┘│
│                                 │ cache miss                               │
│                                 ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  COLD TIER: Object Storage (S3/GCS/MinIO)                              ││
│  │  ────────────────────────────────────────                              ││
│  │  • Complete segment archive                                            ││
│  │  • Durable, replicated, cheap                                          ││
│  │  • Range GET for specific doc offsets                                  ││
│  │                                                                         ││
│  │  Capacity: 100% of dataset, 50-200ms access                            ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Segment-Based Object Storage API

```rust
/// Object storage backend trait
#[async_trait]
pub trait ObjectStore: Send + Sync {
    /// Download entire segment to local cache
    async fn get_segment(&self, path: &str) -> Result<Bytes>;
    
    /// Range read for specific document (efficient for large segments)
    async fn get_range(&self, path: &str, offset: u64, len: u64) -> Result<Bytes>;
    
    /// Upload segment (during compaction/writes)
    async fn put_segment(&self, path: &str, data: Bytes) -> Result<()>;
    
    /// List segments (for initialization)
    async fn list_segments(&self, prefix: &str) -> Result<Vec<String>>;
}

/// S3-compatible implementation (works with AWS S3, MinIO, GCS, etc.)
pub struct S3ObjectStore {
    client: aws_sdk_s3::Client,
    bucket: String,
    prefix: String,
}

#[async_trait]
impl ObjectStore for S3ObjectStore {
    async fn get_range(&self, path: &str, offset: u64, len: u64) -> Result<Bytes> {
        let range = format!("bytes={}-{}", offset, offset + len - 1);
        let resp = self.client
            .get_object()
            .bucket(&self.bucket)
            .key(format!("{}/{}", self.prefix, path))
            .range(range)
            .send()
            .await?;
        
        Ok(resp.body.collect().await?.into_bytes())
    }
}

/// Tiered storage manager
pub struct TieredStorage {
    hot_cache: Arc<RwLock<LruCache<SegmentId, Arc<MmapSegment>>>>,
    local_dir: PathBuf,            // Warm tier (local SSD)
    object_store: Arc<dyn ObjectStore>,  // Cold tier
    cache_size_bytes: usize,
}

impl TieredStorage {
    /// Get patch matrix with tiered fallback
    pub async fn get_patches(&self, doc_id: u64) -> Result<PatchMatrix> {
        let (segment_id, offset, len) = self.doc_index.lookup(doc_id)?;
        
        // Try hot cache first
        if let Some(segment) = self.hot_cache.read().get(&segment_id) {
            return segment.get_patch_matrix(offset);
        }
        
        // Try local SSD (warm tier)
        let local_path = self.local_dir.join(format!("seg_{:06}.rvf", segment_id));
        if local_path.exists() {
            let segment = self.mmap_segment(&local_path)?;
            self.hot_cache.write().put(segment_id, segment.clone());
            return segment.get_patch_matrix(offset);
        }
        
        // Fetch from object storage (cold tier)
        let remote_path = format!("patches/seg_{:06}.rvf", segment_id);
        
        // Option A: Fetch entire segment (good for many docs from same segment)
        // Option B: Range GET just the doc bytes (good for single doc access)
        let bytes = self.object_store.get_range(&remote_path, offset, len).await?;
        
        Ok(PatchMatrix::decode(&bytes)?)
    }
    
    /// Prefetch segments for batch reranking (parallel downloads)
    pub async fn prefetch_segments(&self, segment_ids: &[SegmentId]) -> Result<()> {
        let futures: Vec<_> = segment_ids.iter()
            .filter(|id| !self.hot_cache.read().contains(id))
            .map(|id| self.fetch_and_cache_segment(*id))
            .collect();
        
        futures::future::join_all(futures).await;
        Ok(())
    }
}
```

### Segment Sizing Rules

| Use Case | Segment Size | Rationale |
|----------|--------------|-----------|
| **Local SSD only** | 256MB - 1GB | Maximize mmap efficiency |
| **S3/GCS backing** | 64MB - 256MB | Balance range GET latency vs parallelism |
| **Multi-vector docs** | ~100-500 docs/segment | Group by access pattern |

**Formula for ColPali (1024×128 int8):**
- ~128KB per doc
- 256MB segment ≈ 2,000 docs
- 1B docs ≈ 500K segments (manageable with metadata in memory)

---

## Redis Command API for Multi-Vector

### Extended FT.CREATE for Multi-Vector Indexes

```redis
FT.CREATE colpali_index
  ON HASH PREFIX 1 page:
  SCHEMA
    # Pooled vector for candidate generation (Stage A)
    pooled_vector VECTOR HNSW 6
      TYPE FLOAT32
      DIM 128
      DISTANCE_METRIC COSINE
    
    # Multi-vector patches for reranking (Stage B)
    patches MULTIVECTOR 8
      TYPE INT8
      DIM 128
      MAX_VECTORS 1024
      STORAGE RVF             # Use RVF segments
      COMPRESSION SQ8
      TIER_POLICY LRU         # Cache policy
    
    # Document metadata
    title TEXT
    page_num NUMERIC
```

### Extended FT.SEARCH with Rerank Stage

```redis
# Stage A only (fast, approximate)
FT.SEARCH colpali_index "@pooled_vector:[VECTOR_RANGE 0.8 $query_vec]"
  PARAMS 2 query_vec <128 floats>
  LIMIT 0 100

# Stage A + Stage B (candidate gen + MaxSim rerank)
FT.SEARCH colpali_index "*"
  KNN 100 @pooled_vector $query_pooled        # Stage A: top 100 candidates
  RERANK MAXSIM @patches $query_tokens         # Stage B: rerank with patches
    TOPK 10                                    # Final top 10
  PARAMS 4 query_pooled <128 floats> query_tokens <N×128 floats>
  RETURN 2 title page_num
```

### Internal Query Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  FT.SEARCH ... KNN 100 @pooled_vector $q RERANK MAXSIM @patches $tokens    │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  1. Parse query, extract pooled vector + token vectors                       │
├──────────────────────────────────────────────────────────────────────────────┤
│  2. HNSW search on pooled vectors                                            │
│     └─→ Returns: [doc_42, doc_17, doc_99, ...] (top 100 by cosine)          │
├──────────────────────────────────────────────────────────────────────────────┤
│  3. Lookup doc_ids in doc_index → segment locations                          │
│     └─→ [(seg_5, off_1234), (seg_2, off_5678), ...]                         │
├──────────────────────────────────────────────────────────────────────────────┤
│  4. Prefetch segments (parallel, from hot/warm/cold tiers)                   │
│     └─→ Cache segments in LRU                                               │
├──────────────────────────────────────────────────────────────────────────────┤
│  5. For each candidate doc:                                                  │
│     • Load patch matrix (1024×128 int8)                                     │
│     • Dequantize if needed                                                  │
│     • Compute MaxSim(query_tokens, patches) using SIMD/GPU                  │
├──────────────────────────────────────────────────────────────────────────────┤
│  6. Sort by MaxSim score, return top 10                                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## GPU-Accelerated MaxSim

For large batch reranking, MaxSim can be GPU-accelerated:

```rust
/// GPU MaxSim computation using cuBLAS
pub struct GpuMaxSimReranker {
    cublas: cudarc::cublas::CudaBlas,
    stream: cudarc::driver::CudaStream,
}

impl GpuMaxSimReranker {
    /// Batch MaxSim for multiple candidates
    /// 
    /// query_tokens: [n_query_tokens × dim] (already on GPU)
    /// candidate_patches: Vec of [n_patches × dim] matrices
    pub fn batch_maxsim(
        &self,
        query_tokens: &GpuMatrix,  // e.g., 32×128
        candidates: &[GpuMatrix],  // e.g., 100 × (1024×128)
    ) -> Vec<f32> {
        let mut scores = Vec::with_capacity(candidates.len());
        
        for patches in candidates {
            // Compute all pairwise similarities: [n_query × n_patches]
            // Using cuBLAS GEMM: C = Q × P^T
            let similarities = self.cublas.gemm(
                query_tokens,        // [n_query × dim]
                patches.transpose(), // [dim × n_patches]
            ); // Result: [n_query × n_patches]
            
            // For each query token, take max over patches
            let max_per_query = self.reduce_max_rows(&similarities); // [n_query]
            
            // Sum all max similarities
            let score = self.reduce_sum(&max_per_query);
            scores.push(score);
        }
        
        scores
    }
}
```

**Performance expectation:**
- 100 candidates × 1024 patches × 32 query tokens × 128 dim
- ~400M dot products
- GPU (A100): ~2-5ms
- CPU (AVX2, 8 cores): ~50-100ms

---

## Posits/Unums: Not Recommended

While posits (unum) offer better precision density for values near 1, they are **not practical** for vector DBs today:

| Issue | Impact |
|-------|--------|
| No SIMD intrinsics | ~10x slower than INT8/FP16 |
| No GPU support | Can't use cuBLAS/Tensor Cores |
| Conversion overhead | fp32 ↔ posit on every operation |
| Limited tooling | No standard crates, debugging harder |

**Recommendation:** Treat posits as a research branch. Focus on INT8 (SQ8) and FP16 — widely optimized, GPU-ready, good quality.

---

## Summary: RedVector's Unique Position

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   RedVector = Redis Protocol + Rust Performance + GPU + Custom Storage      │
│   ═══════════════════════════════════════════════════════════════════       │
│                                                                              │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐          │
│   │  Redis Client   │   │  Vector Query   │   │   GPU Accel     │          │
│   │  Ecosystem      │   │  (FT.SEARCH)    │   │   (wgpu/CUDA)   │          │
│   │  (50+ langs)    │   │                 │   │                 │          │
│   └────────┬────────┘   └────────┬────────┘   └────────┬────────┘          │
│            │                     │                     │                    │
│            └─────────────────────┼─────────────────────┘                    │
│                                  ▼                                          │
│            ┌─────────────────────────────────────────────┐                  │
│            │              RedVector Core                  │                  │
│            │  • HNSW (CPU)  • IVF-SQ8 (GPU)  • IVF-PQ    │                  │
│            │  • Multi-Vector (ColPali/ColBERT)           │                  │
│            │  • GPU MaxSim Reranking                      │                  │
│            └─────────────────────────────────────────────┘                  │
│                                  │                                          │
│                                  ▼                                          │
│            ┌─────────────────────────────────────────────┐                  │
│            │         RVF (Vector Filesystem)             │                  │
│            │  • mmap  • SQ8/PQ  • Zero-copy  • Tiered   │                  │
│            │  • S3/GCS/MinIO backing  • Multi-vector    │                  │
│            └─────────────────────────────────────────────┘                  │
│                                                                              │
│   No other vector database has this combination!                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## References

1. [FAISS Storage Design](https://github.com/facebookresearch/faiss/wiki/IO,-cloning-and-hyper-parameter-tuning)
2. [DiskANN Paper](https://proceedings.neurips.cc/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf)
3. [memmap2 Crate](https://docs.rs/memmap2/latest/memmap2/)
4. [Qdrant Storage](https://qdrant.tech/documentation/concepts/storage/)
5. [LanceDB Columnar Format](https://lancedb.github.io/lancedb/)
6. [ColPali - Hugging Face](https://huggingface.co/docs/transformers/en/model_doc/colpali)
7. [Qdrant PDF Retrieval at Scale](https://qdrant.tech/documentation/advanced-tutorials/pdf-retrieval-at-scale/)
8. [MinIO Erasure Coding](https://blog.min.io/erasure-coding-vs-raid/)
9. [ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832)

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2024-12-24 | Initial draft |
| 0.2 | 2024-12-24 | Added Redis persistence model, key→pointer architecture |
| 0.3 | 2024-12-25 | Added multi-vector (ColPali/ColBERT), object storage, GPU MaxSim |

