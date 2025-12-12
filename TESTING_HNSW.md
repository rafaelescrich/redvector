# Testing and Benchmarking HNSW

This guide explains how to run unit tests and benchmarks for the HNSW vector index implementation.

## Prerequisites

Build with the `hnsw-backend` feature enabled:

```bash
cargo build --features hnsw-backend
```

## Running Unit Tests

### All HNSW Tests

```bash
# Run all tests in the database package with HNSW feature
cargo test --package database --features hnsw-backend

# Run only vector_index tests
cargo test --package database --features hnsw-backend vector_index

# Run with output
cargo test --package database --features hnsw-backend -- --nocapture
```

### Specific Test Categories

```bash
# Test basic operations
cargo test --package database --features hnsw-backend test_basic_operations

# Test dimension validation
cargo test --package database --features hnsw-backend test_dimension_validation

# Test large datasets
cargo test --package database --features hnsw-backend test_large_dataset

# Test concurrent access
cargo test --package database --features hnsw-backend test_concurrent_access
```

### Test Coverage

The unit tests cover:

- ✅ Basic operations (add, search, remove)
- ✅ Dimension validation
- ✅ Duplicate prevention
- ✅ Search result ordering
- ✅ Empty index behavior
- ✅ Remove operations
- ✅ Large datasets (10,000+ vectors)
- ✅ Different distance metrics (Cosine, Euclidean)
- ✅ ef_search parameter variations
- ✅ Trait interface compliance
- ✅ Concurrent access patterns

## Running Benchmarks

### Basic Benchmark

```bash
# Run the default HNSW benchmark
cargo run --release --bin hnsw_benchmark --features hnsw-backend
```

This will:
- Create 10,000 vectors of 384 dimensions
- Insert them into HNSW index
- Run 1,000 search queries
- Report insertion throughput, search QPS, and latency percentiles

### Custom Benchmark Configuration

Edit `benchmarks/hnsw_benchmark.rs` to modify the `BenchmarkConfig`:

```rust
let config = BenchmarkConfig {
    dataset_size: 50000,      // Number of vectors
    vector_dimension: 768,    // Vector dimension
    search_queries: 5000,     // Number of search queries
    k: 10,                    // Results per query
    m: 16,                    // HNSW M parameter
    ef_construction: 200,     // Construction parameter
    ef_search: 50,            // Search parameter
};
```

### Comparison Benchmarks

Uncomment the comparison functions in `hnsw_benchmark.rs` to test:

1. **Different M values** (8, 16, 32)
   - Tests impact of graph connectivity on performance

2. **Different ef_search values** (20, 50, 100, 200)
   - Tests trade-off between recall and speed

3. **Scalability** (1K, 10K, 100K vectors)
   - Tests performance at different scales

```rust
// In main() function, uncomment:
compare_configurations();
benchmark_scalability();
```

## Expected Results

### Unit Tests

All tests should pass. Typical output:

```
running 12 tests
test vector_index::tests::test_hnsw_basic ... ok
test vector_index::tests::test_hnsw_dimension_mismatch ... ok
test vector_index::tests::test_hnsw_duplicate_add ... ok
...
test result: ok. 12 passed; 0 failed; 0 ignored
```

### Benchmarks

Expected performance (on modern hardware):

**10,000 vectors, 384 dimensions:**
- Insertion: 2,000-5,000 vectors/sec
- Search QPS: 800-1,500 queries/sec
- P95 Latency: 1-3ms
- Memory: ~15-25 MB

**100,000 vectors, 384 dimensions:**
- Insertion: 1,500-3,000 vectors/sec
- Search QPS: 500-1,000 queries/sec
- P95 Latency: 2-5ms
- Memory: ~150-250 MB

## Performance Tuning

### For Higher Recall

Increase these parameters:
- `m`: 16 → 32 (more connections per node)
- `ef_construction`: 200 → 400 (more candidates during build)
- `ef_search`: 50 → 100 (more candidates during search)

Trade-off: Slower inserts/searches, more memory

### For Higher Speed

Decrease these parameters:
- `m`: 16 → 8 (fewer connections)
- `ef_construction`: 200 → 100 (fewer candidates)
- `ef_search`: 50 → 20 (fewer candidates)

Trade-off: Lower recall, less memory

### Recommended Settings

**Production (98%+ recall):**
- m = 16
- ef_construction = 200
- ef_search = k * 2 (where k is number of results)

**Development/Testing:**
- m = 8
- ef_construction = 100
- ef_search = k

## Troubleshooting

### Tests Fail to Compile

```bash
# Make sure hnsw-backend feature is enabled
cargo test --package database --features hnsw-backend --no-default-features
```

### Benchmark Runs Slowly

- Reduce `dataset_size` for faster iteration
- Reduce `search_queries` for quicker results
- Use `--release` flag for optimized builds

### Memory Issues

- Reduce `m` parameter (uses less memory)
- Process datasets in batches
- Consider using smaller `ef_construction`

## Continuous Integration

Add to your CI pipeline:

```yaml
# Example GitHub Actions
- name: Test HNSW
  run: cargo test --package database --features hnsw-backend

- name: Benchmark HNSW
  run: cargo run --release --bin hnsw_benchmark --features hnsw-backend
```

## Next Steps

1. Run tests to verify correctness
2. Run benchmarks to measure performance
3. Tune parameters for your use case
4. Compare with baseline (brute force) implementation
5. Test with real embedding data

For more details, see [HNSW_INTEGRATION.md](HNSW_INTEGRATION.md).

