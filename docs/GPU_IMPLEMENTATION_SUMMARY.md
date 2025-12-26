# 🚀 RedVector GPU Acceleration Summary

## TL;DR

RedVector will support **GPU-accelerated vector search** on:
- **NVIDIA GPUs** (Linux/Windows) via wgpu + optional CUDA
- **Apple Silicon** (M1/M2/M3) via wgpu → Metal
- **AMD/Intel GPUs** via wgpu → Vulkan

**Expected speedup:** 10-50x over CPU for large-scale vector search.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RedVector                                     │
│                                                                      │
│   ┌────────────────────────────────────────────────────────────┐    │
│   │                 GpuVectorIndex Trait                        │    │
│   │        add() | search() | batch_search()                    │    │
│   └────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│        ┌─────────────────────┼─────────────────────┐                │
│        ▼                     ▼                     ▼                │
│   ┌─────────┐          ┌─────────┐          ┌─────────┐            │
│   │  wgpu   │          │ cudarc  │          │   CPU   │            │
│   │ Backend │          │ Backend │          │ Fallback│            │
│   │(default)│          │(optional│          │ (SIMD)  │            │
│   └────┬────┘          └────┬────┘          └────┬────┘            │
│        │                    │                    │                  │
│   ┌────┴────┐          ┌────┴────┐          ┌────┴────┐            │
│   │ Vulkan  │          │  CUDA   │          │  AVX2   │            │
│   │ Metal   │          │ cuBLAS  │          │  SSE4   │            │
│   │  DX12   │          │         │          │         │            │
│   └─────────┘          └─────────┘          └─────────┘            │
│        │                    │                                       │
│   ┌────┴────────────────────┴────┐                                 │
│   │     Supported Hardware        │                                 │
│   │  • NVIDIA RTX/GTX            │                                 │
│   │  • Apple M1/M2/M3            │                                 │
│   │  • AMD Radeon                │                                 │
│   │  • Intel Arc                 │                                 │
│   └───────────────────────────────┘                                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## FAISS Feature Comparison

| Feature | FAISS | RedVector (Current) | RedVector (Planned) |
|---------|-------|---------------------|---------------------|
| **Flat (Brute Force)** | ✅ CPU/GPU | ✅ CPU (LinearScan) | ✅ GPU (P0) |
| **HNSW** | ✅ CPU only | ✅ CPU | ✅ CPU (keep - NOT GPU) |
| **IVF-Flat** | ✅ CPU/GPU | ❌ | ✅ GPU (P1) |
| **IVF-SQ8** ⭐ | ✅ CPU/GPU | ❌ | ✅ GPU (P1) |
| **IVF-PQ** | ✅ CPU/GPU | ❌ | ✅ GPU (P2) |
| **Scalar Quantization** | ✅ | ❌ | ✅ (P1) |
| **Product Quantization** | ✅ CPU/GPU | ❌ | ✅ CPU/GPU (P2) |
| **GPU Distance (L2)** | ✅ CUDA | ❌ | ✅ wgpu/CUDA |
| **GPU Distance (Cosine)** | ✅ CUDA | ❌ | ✅ wgpu/CUDA |
| **Batch Search** | ✅ | ❌ | ✅ GPU |
| **Apple Silicon** | ❌ | ✅ SIMD | ✅ Metal (wgpu) |
| **Multi-GPU** | ✅ | ❌ | Future (v2.0) |
| **DiskANN** | ❌ | ❌ | Future (v2.0) |
| **MaxSim (ColPali/ColBERT)** | ❌ | ❌ | ✅ GPU (P3) |

### Why HNSW Stays on CPU

**FAISS does NOT offer GPU HNSW** — and for good reason:
- Graph traversal = pointer chasing (irregular memory access)
- Branch divergence kills GPU parallelism
- IVF family = dense matrix ops = GPU-friendly ✅

---

## Implementation Phases

```
Phase 0         Phase 1         Phase 2         Phase 3           Phase 4
Week 1          Week 2-3        Week 4-5        Week 6-8 ⭐       Week 9-12
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌───────────┐     ┌─────────┐
│  Infra  │───▶│  wgpu   │───▶│  CUDA   │───▶│  IVF+SQ8  │────▶│ PQ+     │
│         │    │  Flat   │    │  Flat   │    │ (Priority)│     │ Polish  │
│         │    │         │    │         │    │           │     │         │
│ • Traits│    │ • WGSL  │    │ • cudarc│    │ • IVF-Flat│     │ • IVFPQ │
│ • Config│    │   shaders    │ • cuBLAS│    │ • IVF-SQ8 │     │ • Docs  │
│ • Memory│    │ • Metal │    │ • Kernel│    │ • K-means │     │ • Tuning│
└─────────┘    └─────────┘    └─────────┘    └───────────┘     └─────────┘

Phase 5 (Multi-Vector):
┌─────────────┐
│  MaxSim     │
│  (ColPali)  │
│             │
│ • cuBLAS    │
│   GEMM      │
│ • Batch     │
│   rerank    │
└─────────────┘

Future (v2.0):
┌─────────────────────────────────────────────────────────────────────────┐
│  DiskANN (bigger than RAM)  |  Multi-GPU  |  Hybrid CPU/GPU tiering    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Priority Ladder (What Production Uses)

| Priority | Index | When to Use | Memory (10M × 768d) |
|----------|-------|-------------|---------------------|
| P0 | **GpuFlat** | <1M vectors, exact search | 30 GB |
| P1 | **GpuIVF-Flat** | 1-10M vectors | 30 GB |
| P1 ⭐ | **GpuIVF-SQ8** | 1-100M vectors (sweet spot) | **7.5 GB** |
| P2 | **GpuIVF-PQ** | 100M+ vectors | ~1 GB |
| Future | **DiskANN** | Billions, bigger than RAM | Disk |

---

## Platform Support Matrix

| OS | GPU | Backend | Status |
|----|-----|---------|--------|
| **Linux** | NVIDIA | wgpu (Vulkan) + CUDA | Primary |
| **Linux** | AMD | wgpu (Vulkan) | Supported |
| **Linux** | Intel | wgpu (Vulkan) | Supported |
| **macOS** | Apple Silicon | wgpu (Metal) | Primary |
| **macOS** | AMD (older Macs) | wgpu (Metal) | Supported |
| **Windows** | NVIDIA | wgpu (DX12/Vulkan) + CUDA | Primary |
| **Windows** | AMD | wgpu (DX12/Vulkan) | Supported |
| **Windows** | Intel | wgpu (DX12) | Supported |

---

## Expected Performance

### Single Query Latency (768-dim, k=10)

| Index Type | Dataset | CPU (SIMD) | wgpu (M2 Pro) | CUDA (RTX 4090) |
|------------|---------|------------|---------------|-----------------|
| Flat | 100K | 50ms | 5ms | **1ms** |
| Flat | 1M | 500ms | 50ms | **8ms** |
| IVF-Flat | 1M | 100ms | 10ms | **2ms** |
| **IVF-SQ8** ⭐ | 1M | 80ms | 8ms | **1.5ms** |
| **IVF-SQ8** ⭐ | 10M | 800ms | 80ms | **15ms** |
| IVF-PQ | 10M | 500ms | 50ms | **10ms** |

### Batch Query Throughput (1000 queries, 10M vectors)

| Index Type | CPU | wgpu (M2) | CUDA (RTX 4090) |
|------------|-----|-----------|-----------------|
| IVF-Flat | 20 QPS | 200 QPS | 400 QPS |
| **IVF-SQ8** | 25 QPS | 250 QPS | **500 QPS** |
| IVF-PQ | 30 QPS | 300 QPS | **600 QPS** |

---

## Memory Efficiency

| Index Type | Memory/Vector (768d) | 10M Vectors | Recall@10 |
|------------|---------------------|-------------|-----------|
| Flat (float32) | 3,072 bytes | 30 GB | 100% |
| IVF-Flat | 3,072 bytes | 30 GB | 98-99% |
| **IVF-SQ8** ⭐ | **768 bytes** | **7.5 GB** | **97-98%** |
| IVF-SQ4 | 384 bytes | 3.8 GB | 94-96% |
| IVF-PQ (m=8) | 96 bytes | 1 GB | 90-95% |
| IVF-PQ (m=16) | 48 bytes | 480 MB | 85-92% |

### Why IVF-SQ8 is the Sweet Spot ⭐

```
┌────────────────────────────────────────────────────────────────┐
│  IVF-SQ8 = 4x memory savings + only 1-2% recall loss          │
│                                                                │
│  • Simple int8 quantization (no codebook training)            │
│  • 7.5 GB for 10M vectors vs 30 GB for float32                │
│  • 97-98% recall (vs 99% for full precision)                  │
│  • Best balance before jumping to PQ complexity               │
└────────────────────────────────────────────────────────────────┘
```

---

## Usage Examples

### Basic GPU Flat Search (<1M vectors)

```rust
use redisearch_platform_core::gpu::{GpuFlatIndex, GpuConfig};

// Auto-detect best GPU
let mut index = GpuFlatIndex::new(768, GpuConfig::default())?;

// Add vectors
index.add_vectors(&doc_ids, &vectors)?;

// Search (exact)
let results = index.search(&query, 10)?;
```

### IVF-SQ8 for Production (1M-100M vectors) ⭐

```rust
use redisearch_platform_core::gpu::{IvfSQGpuIndex, GpuConfig, QuantizerType};

// Create IVF-SQ8 index (4x memory savings)
let mut index = IvfSQGpuIndex::new(
    768,                        // dimension
    4096,                       // nlist (clusters)
    QuantizerType::SQ8,         // int8 quantization
    GpuConfig::default()
)?;

// Train on representative data
index.train(&training_vectors)?;

// Add vectors (automatically compressed to int8)
index.add_vectors(&doc_ids, &vectors)?;

// Search with nprobe (clusters to search)
let results = index.search(&query, 10, 32)?;  // nprobe=32
```

### Force Specific Backend

```rust
use redisearch_platform_core::gpu::{GpuConfig, GpuBackend};

// Force wgpu (works on macOS Metal)
let config = GpuConfig {
    backend: GpuBackend::Wgpu,
    ..Default::default()
};

// Force CUDA (NVIDIA only, maximum performance)
let config = GpuConfig {
    backend: GpuBackend::Cuda,
    ..Default::default()
};
```

### Redis Commands

```redis
# Create GPU Flat index (exact search)
FT.CREATE myindex SCHEMA 
    vec VECTOR FLAT 6 TYPE FLOAT32 DIM 768 DISTANCE_METRIC COSINE GPU ON

# Create GPU IVF-SQ8 index (production recommended) ⭐
FT.CREATE myindex_prod SCHEMA
    vec VECTOR IVFSQ 12 
        TYPE FLOAT32 DIM 768 DISTANCE_METRIC L2
        NLIST 4096 NPROBE 32 QUANTIZER SQ8 GPU ON

# Search uses GPU automatically
FT.SEARCH myindex_prod "*=>[KNN 10 @vec $query]" PARAMS 2 query "..."
```

---

## Build Configuration

```toml
# Cargo.toml features
[features]
default = ["wgpu-backend"]      # Cross-platform GPU
wgpu-backend = ["wgpu"]         # wgpu (Vulkan/Metal/DX12)
gpu-cuda = ["cudarc"]           # Optional CUDA for NVIDIA
full-gpu = ["wgpu-backend", "gpu-cuda", "ivf"]
```

```bash
# Build for macOS (Metal via wgpu)
cargo build --release --features wgpu-backend

# Build for NVIDIA (CUDA + wgpu)
cargo build --release --features full-gpu

# Build CPU-only (fallback)
cargo build --release
```

---

## FAQ

### Why wgpu instead of just Metal/CUDA?

**wgpu** is a cross-platform GPU abstraction that maps to:
- **Metal** on macOS
- **Vulkan** on Linux/Windows
- **DX12** on Windows

This means one codebase supports all platforms, including Apple Silicon.

### Will this work on my M1/M2/M3 Mac?

**Yes!** The wgpu backend uses Metal on macOS, which fully supports Apple Silicon.

### Do I need CUDA installed?

**Only if** you want maximum performance on NVIDIA GPUs. The wgpu backend works without CUDA.

### What about AMD GPUs?

**Fully supported** via wgpu → Vulkan on Linux/Windows.

---

## Multi-Vector / ColPali Support

RedVector supports **ColPali/ColBERT-style multi-vector retrieval** with GPU-accelerated MaxSim reranking:

### Two-Stage Retrieval Flow

```
Query → [Stage A: ANN on pooled vectors] → Top K candidates
      → [Stage B: GPU MaxSim rerank] → Final results

Stage A: HNSW/IVF on summary vectors (O(log N))
Stage B: cuBLAS GEMM for MaxSim (O(K × patches × query_tokens))
```

### GPU MaxSim Performance

| Candidates | Patches/Doc | Query Tokens | RTX 4090 | M2 Pro (wgpu) |
|------------|-------------|--------------|----------|---------------|
| 100 | 1024 | 32 | **2ms** | 15ms |
| 500 | 1024 | 32 | **8ms** | 60ms |
| 1000 | 1024 | 32 | **15ms** | 120ms |

### Redis Command Example

```redis
# Create multi-vector index for ColPali
FT.CREATE pdf_pages ON HASH PREFIX 1 page:
  SCHEMA
    pooled_vector VECTOR HNSW 6 TYPE FLOAT32 DIM 128 DISTANCE_METRIC COSINE
    patches MULTIVECTOR 8 TYPE INT8 DIM 128 MAX_VECTORS 1024 COMPRESSION SQ8

# Search with two-stage retrieval + MaxSim rerank
FT.SEARCH pdf_pages "*"
  KNN 100 @pooled_vector $query_pooled
  RERANK MAXSIM @patches $query_tokens TOPK 10
  PARAMS 4 query_pooled <128 floats> query_tokens <32×128 floats>
```

See [ADR-002: Architecture Advantages](./adr/ADR-002-ARCHITECTURE-ADVANTAGES.md) for full multi-vector storage design.

---

## Related Documents

- [ADR-001: GPU Acceleration](./adr/ADR-001-GPU-ACCELERATION.md) - Full technical specification
- [ADR-002: Architecture Advantages](./adr/ADR-002-ARCHITECTURE-ADVANTAGES.md) - RVF storage + multi-vector
- [BENCHMARK_RESULTS.md](../BENCHMARK_RESULTS.md) - Current CPU performance
- [HNSW_INTEGRATION.md](../HNSW_INTEGRATION.md) - Existing HNSW implementation

