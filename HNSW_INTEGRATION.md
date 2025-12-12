# HNSW Integration for RedVector

## Overview

This document describes the HNSW (Hierarchical Navigable Small World) integration for RedVector, providing high-performance vector similarity search capabilities.

## Implementation Status

✅ **Completed:**
- HNSW backend wrapper (`database/src/vector_index.rs`)
- Integration with `ft_commands.rs`
- Feature flags for optional compilation
- Support for multiple distance metrics (Cosine, Euclidean, Inner Product)

⚠️ **In Progress:**
- Testing and validation
- Performance benchmarking
- API compatibility verification

## Architecture

### Components

1. **`database/src/vector_index.rs`** - HNSW backend implementation
   - `HnswVectorIndex` - Main index structure
   - `VectorMetric` - Distance metric enum
   - `VectorIndexTrait` - Common interface trait

2. **`command/src/ft_commands.rs`** - Command handlers
   - Updated to use HNSW backend when `hnsw-backend` feature is enabled
   - Falls back to `redisearch-platform-core` if HNSW not available

### Dependencies

- **hnsw_rs** v0.1 - Rust HNSW implementation
  - Multithreaded insert/search
  - Configurable parameters (M, ef_construction, ef_search)
  - Support for f32 vectors

## Usage

### Building with HNSW

```bash
# Build with HNSW backend
cargo build --release --features vector-search

# Or explicitly enable hnsw-backend
cargo build --release --features vector-search,hnsw-backend
```

### Configuration

HNSW parameters (set in `HnswVectorIndex::new()`):
- **m** (default: 16) - Number of bi-directional links per node
  - Higher = better recall, more memory, slower inserts
- **ef_construction** (default: 200) - Candidate list size during construction
  - Higher = better recall, slower builds
- **ef_search** (default: k * 2) - Candidate list size during search
  - Higher = better recall, slower searches

**Recommended settings for 98%+ recall:**
- m = 16
- ef_construction = 200
- ef_search = k * 2 (where k is number of results)

### Example Commands

```redis
# Create index with 384-dimensional vectors
FT.CREATE my_index SCHEMA vector VECTOR(384)

# Add document with vector
FT.ADD my_index doc1 1.0 FIELDS vector "0.1,0.2,0.3,..."

# Search for similar vectors
FT.SEARCH my_index "0.15,0.25,0.35,..." LIMIT 0 10
```

## Performance Expectations

Based on benchmarks with similar implementations:

| Metric | Current (Brute Force) | With HNSW | Improvement |
|--------|----------------------|-----------|-------------|
| Search QPS (1M vectors) | 300-800 | 1000-1500+ | 2-3x |
| P95 Latency | 3-8ms | 1-3ms | 2-3x faster |
| Recall@10 | 95%+ | 98%+ | Similar |
| Memory | ~1.5GB | ~1.5-2GB | Slightly higher |
| Insert Throughput | 2500/sec | 5000+/sec | 2x |

## Known Limitations

1. **Deletion**: HNSW doesn't efficiently support deletion. The current implementation marks documents as deleted but doesn't remove them from the graph structure. For production use, consider periodic index rebuilds.

2. **Real-time Updates**: Frequent inserts can degrade performance over time. Consider batch updates or periodic rebuilds for large-scale deployments.

3. **Memory**: HNSW uses more memory than brute force (approximately 1.5-2x).

## Future Improvements

- [ ] Support for persistence (dump/load index to disk)
- [ ] Batch insert operations
- [ ] Index rebuilding for deleted documents
- [ ] Support for additional distance metrics
- [ ] Memory-mapped storage for large indexes
- [ ] Parallel search operations

## Testing

```bash
# Run tests with HNSW backend
cargo test --features vector-search,hnsw-backend

# Run integration tests
./run_tests.sh
```

## References

- [hnsw_rs crate](https://crates.io/crates/hnsw_rs)
- [HNSW Paper](https://arxiv.org/abs/1603.09320)
- [Qdrant Benchmarks](https://qdrant.tech/benchmarks/)

