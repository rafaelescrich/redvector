# HNSW API Fix Required

## Current Issue

The `hnsw_rs` crate (v0.1.19) is not being properly imported. The compilation errors indicate:

1. `hnsw_rs` crate not found when feature is enabled
2. Types `Hnsw`, `DistCosine`, `DistL2` not found

## Solution Steps

### Option 1: Verify hnsw_rs API

Check the actual API of `hnsw_rs` v0.1.19:

```bash
# Generate docs
cargo doc --package hnsw_rs --open

# Or check examples
cargo search hnsw_rs
```

### Option 2: Update to Latest Version

The crate may have API changes. Consider updating:

```toml
hnsw_rs = "0.3"  # Latest version
```

### Option 3: Use Alternative Crate

If `hnsw_rs` API is incompatible, consider:
- `hnsw` crate (87k downloads, more stable)
- `small-world-rs` (quantized vectors)
- `similari` (multimodal support)

## Quick Fix

To make tests compile without HNSW:

1. Comment out HNSW-specific code
2. Use feature flags to conditionally compile
3. Create a mock implementation for testing

## Next Steps

1. **Verify API**: Check `hnsw_rs` v0.1.19 documentation
2. **Update Code**: Adjust imports and types to match actual API
3. **Test**: Run `cargo test --features hnsw-backend`
4. **Benchmark**: Run benchmarks once tests pass

## Temporary Workaround

For now, tests can be run without the `hnsw-backend` feature:

```bash
# Run other tests
cargo test --package database

# HNSW tests will be skipped until API is fixed
```

