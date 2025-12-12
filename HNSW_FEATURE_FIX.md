# HNSW Feature Flag Fix

## Current Issue

The `hnsw_rs` crate is not being linked when the `hnsw-backend` feature is enabled, even though:
- ✅ Feature is correctly defined in `database/Cargo.toml`
- ✅ Feature is propagated in root `Cargo.toml`
- ✅ Metadata shows feature is configured correctly
- ✅ `hnsw_rs` is in dependency tree (via `command` package)

## Root Cause

When using `#[cfg(feature = "hnsw-backend")]` on `use` statements, Rust's resolver still tries to resolve the imports at parse time, even if the code is conditionally compiled. If the optional dependency isn't in the build graph for that specific compilation unit, it fails.

## Solutions

### Option 1: Build from Root (Recommended)

Build the entire workspace with the feature enabled:

```bash
# This ensures all dependencies are built
cargo build --features hnsw-backend
cargo test --features hnsw-backend
```

### Option 2: Use `cfg!` Macro Instead

Instead of `#[cfg(feature = "...")]` on imports, use runtime checks:

```rust
// This won't work for imports, but shows the pattern
if cfg!(feature = "hnsw-backend") {
    // Use hnsw types
}
```

### Option 3: Always Include Dependency (Not Recommended)

Make `hnsw_rs` a non-optional dependency, but only use it when feature is enabled. This increases build time when feature is disabled.

### Option 4: Use a Wrapper Module

Create a wrapper that conditionally exports the types:

```rust
// In database/src/lib.rs or a new module
#[cfg(feature = "hnsw-backend")]
mod hnsw_wrapper {
    pub use hnsw_rs::*;
}

#[cfg(not(feature = "hnsw-backend"))]
mod hnsw_wrapper {
    // Empty stub
}
```

## Current Workaround

For now, the tests and benchmarks are written but cannot compile until this issue is resolved. The code structure is correct - it just needs the feature flag propagation fixed.

## Verification

To verify the feature is working:

```bash
# Check feature is recognized
cargo tree --features hnsw-backend | grep hnsw_rs

# Should show hnsw_rs in the tree
```

## Next Steps

1. Try building from root: `cargo build --features hnsw-backend`
2. If that works, use that approach for testing
3. If not, implement Option 4 (wrapper module)
4. Once compiling, run tests: `cargo test --features hnsw-backend`

