# Fix HNSW Compilation Issue

## Problem

`hnsw_rs` crate is not being linked when `hnsw-backend` feature is enabled, causing:
```
error[E0433]: failed to resolve: use of unresolved module or unlinked crate `hnsw_rs`
```

## Root Cause Analysis

1. Feature is correctly defined: `hnsw-backend = ["hnsw_rs"]` ✅
2. Dependency is optional: `hnsw_rs = { version = "0.1", optional = true }` ✅  
3. Feature propagation works: Root Cargo.toml has feature ✅
4. **But**: `hnsw_rs` is not being compiled when building `database` package

## Possible Causes

1. **Cargo version issue**: Older Cargo might not handle optional deps correctly
2. **Feature gate timing**: Rust resolver tries to resolve imports before feature gates are evaluated
3. **Build graph issue**: Optional dependency not included in build graph for isolated package builds

## Solutions to Try

### Solution 1: Use `dep:` syntax (Already tried - didn't work)
```toml
hnsw-backend = ["dep:hnsw_rs"]
```

### Solution 2: Build entire workspace
```bash
cargo build --features hnsw-backend  # Builds all packages
cargo test --features hnsw-backend   # Tests all packages
```

### Solution 3: Make dependency non-optional (Temporary)
Remove `optional = true` to test if that's the issue:
```toml
[dependencies]
hnsw_rs = "0.1"  # Not optional
```

Then use `#[cfg(feature = "hnsw-backend")]` only on code, not imports.

### Solution 4: Use a shim/wrapper crate
Create a minimal wrapper that's always included, which conditionally re-exports.

### Solution 5: Check Cargo version
Update Cargo if using old version:
```bash
rustup update
```

## Current Status

- ✅ All test code written
- ✅ All benchmark code written  
- ✅ Feature flags configured
- ⚠️ Compilation fails due to dependency linking issue

## Next Steps

1. **Try Solution 2** (build from root) - most likely to work
2. If that works, update test scripts to use root builds
3. If not, try Solution 3 (non-optional dependency)
4. Once compiling, run tests and benchmarks

## Test Command (Once Fixed)

```bash
# From project root
cargo test --features hnsw-backend --package database

# Run benchmarks
cargo run --release --bin hnsw_benchmark --features hnsw-backend
```

