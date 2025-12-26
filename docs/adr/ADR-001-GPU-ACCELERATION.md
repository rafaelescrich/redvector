# ADR-001: GPU Acceleration for Vector Search

| Status | Proposed |
|--------|----------|
| **Date** | 2024-12-23 |
| **Decision Makers** | Rafael Escrich |
| **Technical Area** | Vector Search, GPU Computing |
| **Impacted Components** | redisearch-platform-core, database |

---

## Table of Contents

1. [Context and Problem Statement](#context-and-problem-statement)
2. [Decision Drivers](#decision-drivers)
3. [FAISS Feature Analysis](#faiss-feature-analysis)
4. [Considered Options](#considered-options)
5. [Decision Outcome](#decision-outcome)
6. [Architecture Design](#architecture-design)
7. [Implementation Plan](#implementation-plan)
8. [API Design](#api-design)
9. [Performance Targets](#performance-targets)
10. [Risks and Mitigations](#risks-and-mitigations)
11. [References](#references)

---

## Context and Problem Statement

RedVector currently implements vector similarity search using:
- **HNSW** (Hierarchical Navigable Small World) via `hnsw_rs` crate
- **SIMD-accelerated** distance metrics (AVX2, SSE4.1)
- **Linear scan** fallback for small datasets

While these provide good performance on CPU, modern vector databases like **FAISS** achieve 10-100x speedups using GPU acceleration. To compete with production-grade vector databases, RedVector needs GPU support.

### Current Limitations

| Limitation | Impact |
|------------|--------|
| CPU-only distance calculations | Limited throughput for large-scale search |
| No batch search optimization | Inefficient for multiple concurrent queries |
| Memory-bound operations | Can't leverage GPU's parallel architecture |
| Single platform (x86 SIMD) | No acceleration on Apple Silicon |

### Goals

1. **GPU-accelerated similarity search** for NVIDIA and Apple Silicon
2. **FAISS-compatible index types** (Flat, IVF, PQ, IVFPQ)
3. **Cross-platform support** (Linux/Windows + macOS)
4. **Graceful CPU fallback** when GPU unavailable
5. **Redis-compatible API** maintained

---

## Decision Drivers

### Must Have
- [ ] 10x+ speedup over CPU for batch operations
- [ ] Support both NVIDIA CUDA and Apple Metal
- [ ] Maintain existing CPU codepath as fallback
- [ ] No breaking changes to existing API

### Should Have
- [ ] Product Quantization (PQ) for memory efficiency
- [ ] IVF (Inverted File) indexing for large datasets
- [ ] GPU memory management with spilling to RAM
- [ ] Multi-GPU support

### Could Have
- [ ] WebGPU support for browser deployment
- [ ] AMD ROCm support
- [ ] ONNX Runtime integration for embeddings

### Won't Have (Initially)
- Training on GPU (use pre-computed centroids)
- GPU clustering algorithms
- Distributed multi-node GPU

---

## Industry Reality: What Vector Databases Actually Use

Before diving into FAISS specifics, it's critical to understand **what production vector databases actually deploy**:

### Most Used Index Types in Production

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Production Vector Database Index Usage                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   TIER 1: Default / General Purpose (CPU)                                   │
│   ════════════════════════════════════════                                   │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │  HNSW (Graph-based ANN)                                          │       │
│   │  ─────────────────────                                           │       │
│   │  • Qdrant: HNSW is the dense vector index                        │       │
│   │  • Weaviate: "HNSW is the default"                               │       │
│   │  • Elasticsearch: kNN for dense_vector is HNSW-based             │       │
│   │  • Vespa: ANN docs are explicitly HNSW-based                     │       │
│   │  • pgvector: HNSW recommended for robustness/perf                │       │
│   │                                                                   │       │
│   │  WHY: Great speed/recall, dynamic inserts, easy operations       │       │
│   │  GPU STATUS: ❌ NOT GPU-accelerated (irregular memory access)    │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                              │
│   TIER 2: Scale + Compression + GPU (billions of vectors)                   │
│   ═══════════════════════════════════════════════════════                    │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │  IVF Family (Inverted File Index)                                │       │
│   │  ────────────────────────────────                                │       │
│   │  • FAISS GPU: Flat, IVF-Flat, IVF-SQ, IVF-PQ                     │       │
│   │  • Milvus: IVF_FLAT, IVF_SQ8, IVF_PQ + GPU variants              │       │
│   │  • OpenSearch: HNSW + IVF via FAISS engine                       │       │
│   │                                                                   │       │
│   │  WHY: Natural coarse partitioning, GPU-friendly dense math      │       │
│   │  GPU STATUS: ✅ Primary GPU index family                         │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                              │
│   TIER 3: Bigger-than-RAM (future)                                          │
│   ════════════════════════════════                                           │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │  DiskANN                                                         │       │
│   │  ───────                                                         │       │
│   │  • Milvus: First-class DiskANN support                           │       │
│   │  • Microsoft: Original implementation                            │       │
│   │                                                                   │       │
│   │  WHY: SSD-optimized for datasets >> RAM                          │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why HNSW is NOT GPU-Accelerated

**Critical insight:** FAISS does NOT offer GPU HNSW, and for good reason:

| Characteristic | GPU-Friendly? | Explanation |
|----------------|---------------|-------------|
| Pointer chasing | ❌ No | Graph traversal follows unpredictable links |
| Irregular memory access | ❌ No | Can't coalesce memory reads |
| Branch divergence | ❌ No | Different paths per thread |
| Dense matrix ops | ✅ Yes | IVF/Flat excel at this |

**Bottom line:** Keep HNSW on CPU (already implemented ✅), use GPU for Flat + IVF family.

---

## FAISS Feature Analysis

### FAISS GPU Index Types (What We're Implementing)

| Index Type | GPU Support | Memory | Speed | Recall | Priority |
|------------|-------------|--------|-------|--------|----------|
| **GpuIndexFlat** | ✅ FAISS GPU | High | Very Fast | 100% | **P0** |
| **GpuIndexIVFFlat** | ✅ FAISS GPU | Medium | Fast | 95-99% | **P1** |
| **GpuIndexIVFScalarQuantizer** | ✅ FAISS GPU | Low | Fast | 95-98% | **P1** |
| **GpuIndexIVFPQ** | ✅ FAISS GPU | Very Low | Fast | 85-95% | **P2** |
| **HNSW** | ❌ CPU only | Medium | Fast | 98%+ | ✅ **Done (CPU)** |
| **DiskANN** | ❌ Disk-based | Very Low | Medium | 95%+ | **Future** |

### GPU Index Priority Ladder

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     GPU Implementation Priority                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  P0: Foundation                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │  GpuIndexFlat (L2, Cosine, InnerProduct)                        │        │
│  │  • Brute-force exact search on GPU                              │        │
│  │  • Baseline for correctness testing                             │        │
│  │  • Small-medium datasets (<1M vectors)                          │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                           │                                                  │
│                           ▼                                                  │
│  P1: Scale (the "sweet spot" for most production use)                       │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │  GpuIndexIVFFlat                                                │        │
│  │  • Coarse quantizer + full-precision vectors                    │        │
│  │  • 1M-100M vectors                                              │        │
│  │  • nlist=1024-4096, nprobe=16-64                                │        │
│  ├─────────────────────────────────────────────────────────────────┤        │
│  │  GpuIndexIVFScalarQuantizer (IVF-SQ8)  ⭐ EARLY WIN             │        │
│  │  • 4x memory reduction (float32 → int8)                         │        │
│  │  • Minimal recall loss (~1-2%)                                  │        │
│  │  • Often better than jumping straight to PQ                     │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                           │                                                  │
│                           ▼                                                  │
│  P2: Maximum Compression                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │  GpuIndexIVFPQ                                                  │        │
│  │  • 32x+ memory reduction                                        │        │
│  │  • Requires training on representative data                     │        │
│  │  • 100M-1B+ vectors                                             │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                           │                                                  │
│                           ▼                                                  │
│  Future: Bigger-than-RAM                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │  DiskANN                                                        │        │
│  │  • SSD-resident index                                           │        │
│  │  • Graph + disk layout + compression                            │        │
│  │  • 1B+ vectors without RAM constraints                          │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### FAISS GPU Features to Implement

```
┌─────────────────────────────────────────────────────────────────┐
│                    FAISS GPU Feature Map                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────┐ │
│  │  GpuFlat     │  │ GpuIVFFlat   │  │  GpuIVF-SQ8  │  │IVFPQ │ │
│  │  (P0)        │  │  (P1)        │  │  (P1) ⭐     │  │ (P2) │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──┬───┘ │
│         │                 │                 │              │     │
│         ▼                 ▼                 ▼              ▼     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  GpuResources                             │   │
│  │  - Memory Pool     - Temp Memory    - Streams            │   │
│  └──────────────────────────────────────────────────────────┘   │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              GPU Distance Kernels                         │   │
│  │  - L2 Distance    - Cosine    - Inner Product            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ════════════════════════════════════════════════════════════   │
│                                                                  │
│  CPU-ONLY (stays on CPU, already implemented):                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  HNSW (graph traversal not GPU-friendly)                  │   │
│  │  ✅ Already implemented via hnsw_rs                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### RedVector Index Strategy Summary

| Use Case | Index | Platform | Status |
|----------|-------|----------|--------|
| **Default / General** | HNSW | CPU (SIMD) | ✅ Done |
| **GPU Throughput** | Flat → IVF-Flat → IVF-SQ8 | GPU (wgpu/CUDA) | 🔨 Planned |
| **Billion-scale** | IVF-PQ | GPU | 🔨 Planned |
| **Bigger-than-RAM** | DiskANN | Disk + CPU | 📋 Future |

---

## Considered Options

### Option 1: cudarc (CUDA) + metal-rs (Metal)

**Separate libraries for each GPU backend**

| Aspect | CUDA (cudarc) | Metal (metal-rs) |
|--------|---------------|------------------|
| Maturity | ⭐⭐⭐⭐ High | ⭐⭐⭐ Medium |
| Performance | Excellent | Excellent |
| Complexity | Medium | Medium |
| Maintenance | Two codepaths | Two codepaths |

```rust
// Cargo.toml
[dependencies]
cudarc = { version = "0.12", optional = true }
metal = { version = "0.28", optional = true }

[features]
gpu-cuda = ["cudarc"]
gpu-metal = ["metal"]
```

**Pros:**
- Maximum performance on each platform
- Direct access to platform-specific features (Tensor Cores, Apple Neural Engine)
- Mature, well-documented libraries

**Cons:**
- Two separate implementations to maintain
- Different shader languages (CUDA C vs MSL)
- No code sharing between backends

---

### Option 2: wgpu (Cross-Platform WebGPU)

**Single abstraction layer over Vulkan/Metal/DX12**

```rust
// Cargo.toml
[dependencies]
wgpu = "0.20"
```

```
┌─────────────────────────────────────────────────────────┐
│                      RedVector                           │
│                         │                                │
│                    ┌────▼────┐                           │
│                    │  wgpu   │                           │
│                    └────┬────┘                           │
│         ┌───────────────┼───────────────┐                │
│         ▼               ▼               ▼                │
│    ┌─────────┐    ┌─────────┐    ┌─────────┐            │
│    │ Vulkan  │    │  Metal  │    │  DX12   │            │
│    │ (Linux) │    │ (macOS) │    │ (Win)   │            │
│    └─────────┘    └─────────┘    └─────────┘            │
│         │               │               │                │
│         ▼               ▼               ▼                │
│    ┌─────────┐    ┌─────────┐    ┌─────────┐            │
│    │ NVIDIA  │    │ Apple   │    │ NVIDIA/ │            │
│    │   GPU   │    │ Silicon │    │   AMD   │            │
│    └─────────┘    └─────────┘    └─────────┘            │
└─────────────────────────────────────────────────────────┘
```

**Pros:**
- Single codebase for all platforms
- WGSL shaders work everywhere
- WebGPU compatibility (future browser support)
- Excellent Rust support

**Cons:**
- ~10-20% overhead vs native CUDA
- No access to CUDA-specific features (Tensor Cores, cuBLAS)
- Less mature for compute workloads

---

### Option 3: Hybrid Approach (Recommended)

**wgpu for cross-platform + optional CUDA for maximum performance**

```
┌───────────────────────────────────────────────────────────────┐
│                    GPU Abstraction Layer                       │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│   ┌─────────────────────────────────────────────────────┐     │
│   │              GpuVectorIndex (Trait)                  │     │
│   │  - add_vectors()  - search()  - batch_search()      │     │
│   └─────────────────────────────────────────────────────┘     │
│                            │                                   │
│            ┌───────────────┼───────────────┐                  │
│            ▼               ▼               ▼                  │
│   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐         │
│   │  CudaIndex   │ │  WgpuIndex   │ │   CpuIndex   │         │
│   │  (cudarc)    │ │  (wgpu)      │ │  (fallback)  │         │
│   │  [optional]  │ │  [default]   │ │  [always]    │         │
│   └──────────────┘ └──────────────┘ └──────────────┘         │
│         │                   │               │                  │
│   ┌─────▼─────┐     ┌──────▼──────┐  ┌─────▼─────┐           │
│   │  cuBLAS   │     │   Vulkan    │  │   SIMD    │           │
│   │  Kernels  │     │   Metal     │  │  (AVX2)   │           │
│   └───────────┘     │   DX12      │  └───────────┘           │
│                     └─────────────┘                           │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

**Pros:**
- Best of both worlds
- wgpu covers all platforms including Apple Silicon
- Optional CUDA for maximum NVIDIA performance
- CPU fallback always available
- Single API for users

**Cons:**
- More complex build system
- Optional CUDA adds maintenance burden

---

## Decision Outcome

### Chosen Option: **Option 3 - Hybrid Approach**

We will implement:

1. **Primary Backend: wgpu** 
   - Cross-platform (Linux, Windows, macOS)
   - Works on NVIDIA, AMD, Intel, Apple Silicon
   - WGSL compute shaders
   - Default for all platforms

2. **Optional High-Performance Backend: cudarc**
   - NVIDIA GPUs only
   - cuBLAS for matrix operations
   - Custom CUDA kernels for specialized ops
   - Enabled via `gpu-cuda` feature flag

3. **Always Available: CPU Fallback**
   - Existing SIMD implementation
   - Automatic fallback when no GPU available

### Rationale

| Criteria | wgpu | cudarc | Combined |
|----------|------|--------|----------|
| macOS Metal | ✅ | ❌ | ✅ |
| Apple Silicon | ✅ | ❌ | ✅ |
| NVIDIA Linux | ✅ | ✅ (faster) | ✅✅ |
| NVIDIA Windows | ✅ | ✅ (faster) | ✅✅ |
| AMD | ✅ | ❌ | ✅ |
| Intel | ✅ | ❌ | ✅ |
| WebGPU | ✅ | ❌ | ✅ |
| Maintenance | Low | Medium | Medium |

---

## Architecture Design

### Module Structure

```
redisearch-platform-core/
├── src/
│   ├── lib.rs
│   ├── vector_index.rs          # Existing HNSW
│   ├── simd_metrics.rs          # Existing SIMD
│   ├── storage.rs               # Existing persistence
│   │
│   ├── gpu/
│   │   ├── mod.rs               # GPU module entry point
│   │   ├── traits.rs            # GpuVectorIndex trait
│   │   ├── config.rs            # GPU configuration
│   │   ├── memory.rs            # GPU memory management
│   │   │
│   │   ├── wgpu/
│   │   │   ├── mod.rs           # wgpu backend
│   │   │   ├── index.rs         # WgpuVectorIndex
│   │   │   ├── shaders/
│   │   │   │   ├── l2_distance.wgsl
│   │   │   │   ├── cosine.wgsl
│   │   │   │   ├── inner_product.wgsl
│   │   │   │   └── top_k.wgsl
│   │   │   └── pipelines.rs     # Compute pipelines
│   │   │
│   │   ├── cuda/                # Optional CUDA backend
│   │   │   ├── mod.rs
│   │   │   ├── index.rs         # CudaVectorIndex
│   │   │   ├── kernels/
│   │   │   │   ├── distance.cu  # CUDA C kernels
│   │   │   │   └── topk.cu
│   │   │   └── blas.rs          # cuBLAS wrapper
│   │   │
│   │   └── fallback.rs          # CPU fallback
│   │
│   ├── quantization/
│   │   ├── mod.rs               # Quantization module
│   │   ├── scalar.rs            # Scalar quantization (int8, float16)
│   │   ├── product.rs           # Product Quantization (PQ)
│   │   └── training.rs          # PQ training utilities
│   │
│   └── ivf/
│       ├── mod.rs               # IVF module
│       ├── clustering.rs        # K-means clustering
│       ├── index.rs             # IVFIndex
│       └── gpu_ivf.rs           # GPU-accelerated IVF
```

### Dependency Graph

```
┌─────────────────────────────────────────────────────────────────┐
│                        Cargo Features                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  default = ["wgpu-backend"]                                     │
│                                                                  │
│  wgpu-backend = ["wgpu", "pollster", "bytemuck"]                │
│  gpu-cuda = ["cudarc"]                                          │
│  quantization = []                                               │
│  ivf = ["quantization"]                                          │
│                                                                  │
│  full-gpu = ["wgpu-backend", "gpu-cuda", "ivf"]                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Query                    GPU Memory Layout
    │                         ┌─────────────────────┐
    ▼                         │   Database Vectors  │
┌─────────┐                   │  [n_vectors × dim]  │
│ Search  │                   │   (persistent)      │
│ Request │                   ├─────────────────────┤
└────┬────┘                   │   Query Buffer      │
     │                        │  [batch × dim]      │
     ▼                        │   (temporary)       │
┌─────────────┐               ├─────────────────────┤
│ GPU Upload  │──────────────▶│  Distance Matrix    │
│ (if needed) │               │  [batch × n_vectors]│
└─────────────┘               │   (temporary)       │
     │                        ├─────────────────────┤
     ▼                        │   Top-K Results     │
┌─────────────┐               │  [batch × k]        │
│  Distance   │               │   (output)          │
│  Kernel     │               └─────────────────────┘
└─────────────┘
     │
     ▼
┌─────────────┐
│   Top-K     │
│  Selection  │
└─────────────┘
     │
     ▼
┌─────────────┐
│  Download   │
│  Results    │
└─────────────┘
     │
     ▼
  Results
```

---

## Implementation Plan

### Phase 0: Infrastructure (Week 1)

| Task | Description | Effort |
|------|-------------|--------|
| 0.1 | Create GPU module structure | 2h |
| 0.2 | Define `GpuVectorIndex` trait | 2h |
| 0.3 | Add feature flags to Cargo.toml | 1h |
| 0.4 | Set up CI for GPU testing | 4h |
| 0.5 | Create benchmark harness | 3h |

**Deliverable:** Compile-time feature selection, trait definitions

---

### Phase 1: wgpu Backend (Week 2-3)

| Task | Description | Effort |
|------|-------------|--------|
| 1.1 | wgpu device initialization | 4h |
| 1.2 | L2 distance compute shader (WGSL) | 4h |
| 1.3 | Cosine similarity shader | 3h |
| 1.4 | Inner product shader | 2h |
| 1.5 | GPU memory management | 6h |
| 1.6 | Flat index implementation | 8h |
| 1.7 | Batch search support | 4h |
| 1.8 | Top-K selection on GPU | 6h |
| 1.9 | Integration with existing API | 4h |
| 1.10 | Unit tests and benchmarks | 6h |

**Deliverable:** Working GPU search on all platforms

#### WGSL Shader Example (L2 Distance)

```wgsl
// l2_distance.wgsl
@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> database: array<f32>;
@group(0) @binding(2) var<storage, read_write> distances: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    num_vectors: u32,
    dimension: u32,
    query_offset: u32,
    padding: u32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_vectors) {
        return;
    }
    
    var dist: f32 = 0.0;
    let base = idx * params.dimension;
    
    for (var d: u32 = 0u; d < params.dimension; d++) {
        let diff = query[params.query_offset + d] - database[base + d];
        dist += diff * diff;
    }
    
    distances[idx] = sqrt(dist);
}
```

---

### Phase 2: CUDA Backend (Week 4-5)

| Task | Description | Effort |
|------|-------------|--------|
| 2.1 | cudarc initialization | 4h |
| 2.2 | CUDA kernel for L2 distance | 4h |
| 2.3 | cuBLAS integration for GEMM | 6h |
| 2.4 | Memory pool management | 6h |
| 2.5 | Async kernel execution | 4h |
| 2.6 | Stream management | 4h |
| 2.7 | Integration with trait | 4h |
| 2.8 | Benchmarks vs wgpu | 4h |

**Deliverable:** High-performance NVIDIA backend

#### CUDA Kernel Example

```cuda
// distance.cu
extern "C" __global__ void l2_distance(
    const float* __restrict__ query,
    const float* __restrict__ database,
    float* __restrict__ distances,
    int num_vectors,
    int dimension
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vectors) return;
    
    float dist = 0.0f;
    const float* vec = database + idx * dimension;
    
    #pragma unroll 4
    for (int d = 0; d < dimension; d++) {
        float diff = query[d] - vec[d];
        dist = fmaf(diff, diff, dist);  // Fused multiply-add
    }
    
    distances[idx] = sqrtf(dist);
}
```

---

### Phase 3: IVF Index + Scalar Quantization (Week 6-8) ⭐ CRITICAL

**This is the "production sweet spot" - IVF-Flat and IVF-SQ8 are the most commonly used 
GPU indices in real deployments (FAISS, Milvus, OpenSearch).**

| Task | Description | Effort |
|------|-------------|--------|
| 3.1 | K-means clustering (CPU-based training) | 8h |
| 3.2 | IVF inverted lists structure | 6h |
| 3.3 | Coarse quantizer search | 4h |
| 3.4 | GpuIVFFlat implementation | 10h |
| 3.5 | nprobe parameter handling | 4h |
| 3.6 | **Scalar Quantizer (SQ8)** - int8 compression | 8h |
| 3.7 | **GpuIVFScalarQuantizer** implementation | 10h |
| 3.8 | IVF persistence (save/load) | 6h |
| 3.9 | Benchmarks vs Flat | 4h |

**Deliverable:** Production-ready IVF-Flat and IVF-SQ8 indices

#### Why IVF-SQ8 is a Priority

| Comparison | IVF-Flat | IVF-SQ8 | IVF-PQ |
|------------|----------|---------|--------|
| Memory per vector (768d) | 3KB | 768B (4x less) | 48-96B (32x less) |
| Recall@10 | 99% | 97-98% | 90-95% |
| Complexity | Low | Low | High (training) |
| Training required | Centroids only | Centroids only | Centroids + codebook |

**IVF-SQ8 is often the "sweet spot" before jumping to PQ complexity.**

---

### Phase 4: Product Quantization (Week 9-10)

| Task | Description | Effort |
|------|-------------|--------|
| 4.1 | PQ encoder/decoder | 8h |
| 4.2 | Codebook training (CPU) | 6h |
| 4.3 | PQ distance computation (asymmetric) | 6h |
| 4.4 | GPU PQ search | 8h |
| 4.5 | GpuIVFPQ combination | 10h |
| 4.6 | Serialization/persistence | 4h |

**Deliverable:** 32x memory reduction with IVF-PQ for billion-scale

---

### Phase 5: Integration & Polish (Week 11-12)

| Task | Description | Effort |
|------|-------------|--------|
| 5.1 | Automatic backend selection | 4h |
| 5.2 | Redis command integration | 6h |
| 5.3 | Documentation | 8h |
| 5.4 | Performance tuning | 8h |
| 5.5 | Release preparation | 4h |

**Deliverable:** Production-ready GPU acceleration

---

### Phase 6: DiskANN (Future - Beyond v1.0)

| Task | Description | Effort |
|------|-------------|--------|
| 6.1 | Graph-based index with SSD layout | 20h |
| 6.2 | Vamana algorithm implementation | 16h |
| 6.3 | Disk-resident navigation | 12h |
| 6.4 | Memory-mapped I/O | 8h |
| 6.5 | Compression integration | 8h |

**Deliverable:** Billion+ scale with datasets >> RAM

**Note:** DiskANN is a fundamentally different architecture (disk-based graph) 
and should be considered a v2.0 feature after GPU IVF is production-ready.

---

### Implementation Priority Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Implementation Roadmap                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  v1.0 - GPU Foundation (Week 1-12)                                          │
│  ═══════════════════════════════════                                         │
│                                                                              │
│   Week 1        Week 2-3       Week 4-5       Week 6-8       Week 9-12      │
│  ┌───────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐     │
│  │Infra  │───▶│ wgpu    │───▶│ CUDA    │───▶│ IVF +   │───▶│ PQ +    │     │
│  │Traits │    │ Flat    │    │ Flat    │    │ SQ8 ⭐  │    │ Polish  │     │
│  └───────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘     │
│                                                                              │
│  v2.0 - Extended Scale (Future)                                             │
│  ═══════════════════════════════                                             │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  DiskANN - For datasets that don't fit in RAM/VRAM                  │    │
│  │  Multi-GPU - Distributed index across multiple GPUs                 │    │
│  │  Hybrid CPU/GPU - Automatic data tiering                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## API Design

### Public API

```rust
use redisearch_platform_core::gpu::{
    GpuVectorIndex, GpuConfig, GpuBackend, GpuDistanceMetric
};

// Automatic backend selection
let index = GpuVectorIndex::new(dimension, GpuConfig::default())?;

// Explicit backend selection
let config = GpuConfig {
    backend: GpuBackend::Wgpu,  // or GpuBackend::Cuda, GpuBackend::Cpu
    device_id: 0,
    metric: GpuDistanceMetric::Cosine,
    ..Default::default()
};
let index = GpuVectorIndex::with_config(dimension, config)?;

// Add vectors
index.add_vectors(&doc_ids, &vectors)?;

// Single query
let results = index.search(&query_vector, k)?;  // Vec<(doc_id, distance)>

// Batch query (more efficient on GPU)
let batch_results = index.batch_search(&query_vectors, k)?;  // Vec<Vec<(doc_id, distance)>>

// With IVF-Flat (most common production index)
let ivf_index = IvfFlatGpuIndex::new(dimension, n_clusters, config)?;
ivf_index.train(&training_vectors)?;  // K-means clustering
ivf_index.add_vectors(&doc_ids, &vectors)?;
let results = ivf_index.search(&query, k, nprobe)?;  // nprobe = clusters to search

// With IVF-SQ8 (4x memory savings, minimal recall loss) ⭐ RECOMMENDED
let ivf_sq_index = IvfSQGpuIndex::new(dimension, n_clusters, QuantizerType::SQ8, config)?;
ivf_sq_index.train(&training_vectors)?;
ivf_sq_index.add_vectors(&doc_ids, &vectors)?;  // Vectors compressed to int8
let results = ivf_sq_index.search(&query, k, nprobe)?;

// With IVF-PQ (32x memory savings for billion-scale)
let ivfpq_index = IvfPQGpuIndex::new(dimension, n_clusters, n_subquantizers, n_bits, config)?;
ivfpq_index.train(&training_vectors)?;  // Learns codebook from data
ivfpq_index.add_vectors(&doc_ids, &vectors)?;  // Heavily compressed
let results = ivfpq_index.search(&query, k, nprobe)?;
```

### Index Selection Guide

```rust
/// Choose the right index for your use case
fn select_index(num_vectors: usize, available_ram_gb: f32, dimension: usize) -> &'static str {
    let vector_size_gb = (num_vectors * dimension * 4) as f32 / 1e9;
    
    match (num_vectors, vector_size_gb < available_ram_gb * 0.5) {
        (n, _) if n < 100_000 => "GpuFlat (exact search)",
        (n, true) if n < 10_000_000 => "GpuIVFFlat (fast, full precision)",
        (n, false) if n < 10_000_000 => "GpuIVF-SQ8 (4x compression) ⭐",
        (_, true) => "GpuIVFPQ (32x compression)",
        (_, false) => "Consider DiskANN (bigger than RAM)",
    }
}
```

### Redis Commands Extension

```redis
# Create GPU Flat index (exact search, <1M vectors)
FT.CREATE myindex 
    ON HASH PREFIX 1 doc:
    SCHEMA 
        embedding VECTOR FLAT 6 
            TYPE FLOAT32 
            DIM 768 
            DISTANCE_METRIC COSINE
            GPU ON                    # NEW: Enable GPU
            GPU_DEVICE 0              # NEW: Select device

# Create GPU IVF-Flat index (approximate, 1M-100M vectors)
FT.CREATE myindex_ivf
    ON HASH PREFIX 1 doc:
    SCHEMA 
        embedding VECTOR IVF 10       # NEW: IVF index type
            TYPE FLOAT32 
            DIM 768 
            DISTANCE_METRIC L2
            NLIST 4096                # Number of clusters
            NPROBE 32                 # Clusters to search (recall/speed tradeoff)
            GPU ON

# Create GPU IVF-SQ8 index (compressed, memory efficient) ⭐
FT.CREATE myindex_ivfsq
    ON HASH PREFIX 1 doc:
    SCHEMA 
        embedding VECTOR IVFSQ 12     # NEW: IVF + Scalar Quantization
            TYPE FLOAT32 
            DIM 768 
            DISTANCE_METRIC L2
            NLIST 4096
            NPROBE 32
            QUANTIZER SQ8             # int8 quantization (4x memory savings)
            GPU ON

# GPU-accelerated search (works with any index type)
FT.SEARCH myindex "*=>[KNN 10 @embedding $vec]" 
    PARAMS 2 vec "\x00\x00..." 
    DIALECT 2

# Check GPU index info
FT.INFO myindex_ivfsq
# Returns: 
#   gpu_enabled: true
#   gpu_device: "Apple M2 Pro" | "NVIDIA RTX 4090"
#   gpu_memory_mb: 512
#   index_type: "IVF-SQ8"
#   compression_ratio: 4x
#   nlist: 4096
#   nprobe: 32
#   num_vectors: 10000000
```

---

## Performance Targets

### Benchmark Configurations

| Dataset | Vectors | Dimension | Index Type | GPU Memory |
|---------|---------|-----------|------------|------------|
| Small | 100K | 768 | Flat | ~300 MB |
| Medium | 1M | 768 | IVF-Flat | ~3 GB |
| Medium-Compressed | 1M | 768 | IVF-SQ8 | ~750 MB |
| Large | 10M | 768 | IVF-SQ8 | ~7.5 GB |
| Billion-scale | 100M | 768 | IVF-PQ | ~10 GB |

### Target Performance (vs CPU baseline)

| Operation | CPU (SIMD) | wgpu (Apple M2) | wgpu (RTX 4090) | CUDA (RTX 4090) |
|-----------|------------|-----------------|-----------------|-----------------|
| Flat 100K, k=10 | 50ms | 5ms (10x) | 2ms (25x) | 1ms (50x) |
| Flat 1M, k=10 | 500ms | 50ms (10x) | 15ms (33x) | 8ms (62x) |
| IVF-Flat 1M, nprobe=32 | 100ms | 10ms (10x) | 3ms (33x) | 2ms (50x) |
| **IVF-SQ8 1M, nprobe=32** | 80ms | 8ms (10x) | 2.5ms (32x) | 1.5ms (53x) |
| **IVF-SQ8 10M, nprobe=32** | 800ms | 80ms (10x) | 25ms (32x) | 15ms (53x) |
| IVF-PQ 10M, nprobe=32 | 500ms | 50ms (10x) | 15ms (33x) | 10ms (50x) |
| Batch 1000 queries | 50s | 5s (10x) | 1.5s (33x) | 0.8s (62x) |

### Memory Efficiency Targets

| Index Type | Memory per Vector (768d) | Total for 10M | Recall@10 |
|------------|-------------------------|---------------|-----------|
| Flat (float32) | 3,072 bytes | 30.7 GB | 100% |
| IVF-Flat | 3,072 bytes | 30.7 GB | 98-99% |
| **IVF-SQ8 ⭐** | 768 bytes | **7.7 GB** | **97-98%** |
| IVF-SQ4 | 384 bytes | 3.8 GB | 94-96% |
| IVF-PQ (m=8) | 96 bytes | 960 MB | 90-95% |
| IVF-PQ (m=16) | 48 bytes | 480 MB | 85-92% |

### Why IVF-SQ8 is the Sweet Spot

```
Memory vs Recall Tradeoff (10M vectors, 768d)
═══════════════════════════════════════════════

           30 GB ┤ ████████████████████████████████████████  IVF-Flat (99%)
                 │
           20 GB ┤
                 │
           10 GB ┤ ████████████████████  IVF-SQ8 (97%) ⭐ SWEET SPOT
                 │
            5 GB ┤ ██████████  IVF-SQ4 (95%)
                 │
            1 GB ┤ ████  IVF-PQ (92%)
                 │
                 └──────────────────────────────────────────
                   85%    90%    95%    97%    99%   100%
                                 Recall@10

IVF-SQ8 provides the best balance:
• 4x memory reduction vs float32
• Only 1-2% recall loss
• No complex codebook training
• Simple int8 quantization
```

---

## Risks and Mitigations

### Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| wgpu performance below expectations | High | Medium | Benchmark early; CUDA fallback for NVIDIA |
| Metal shader compilation issues | Medium | Low | Test on multiple macOS versions |
| GPU memory management complexity | High | Medium | Use existing GPU memory allocators |
| Cross-platform testing difficulty | Medium | High | CI with macOS, Linux, Windows runners |
| cudarc API changes | Low | Low | Pin version; abstract behind trait |

### Operational Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| GPU driver issues | Medium | Clear documentation on supported drivers |
| No GPU available | Low | Automatic CPU fallback |
| OOM on GPU | Medium | Memory estimation; graceful spilling |

---

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_gpu_l2_distance() {
    let config = GpuConfig::default();
    let index = GpuVectorIndex::new(3, config).unwrap();
    
    index.add_vectors(&[1, 2], &[
        1.0, 0.0, 0.0,  // vec 1
        0.0, 1.0, 0.0,  // vec 2
    ]).unwrap();
    
    let results = index.search(&[1.0, 0.0, 0.0], 1).unwrap();
    assert_eq!(results[0].0, 1);  // Should find vec 1
    assert!(results[0].1 < 0.01);  // Distance ~0
}
```

### Integration Tests

```rust
#[test]
fn test_gpu_vs_cpu_consistency() {
    let vectors = generate_random_vectors(1000, 128);
    let query = generate_random_vectors(1, 128);
    
    let cpu_results = cpu_index.search(&query, 10);
    let gpu_results = gpu_index.search(&query, 10);
    
    // Results should be identical (or very close for approximate)
    for i in 0..10 {
        assert_eq!(cpu_results[i].0, gpu_results[i].0);
        assert!((cpu_results[i].1 - gpu_results[i].1).abs() < 1e-5);
    }
}
```

### Benchmark Tests

```rust
#[bench]
fn bench_gpu_flat_100k(b: &mut Bencher) {
    let index = setup_gpu_index(100_000, 768);
    let query = random_vector(768);
    
    b.iter(|| {
        index.search(&query, 10).unwrap()
    });
}
```

---

## Alternatives Considered but Rejected

### 1. rust-gpu (GPU shaders in Rust)
- **Rejected because:** Still experimental, limited compute support

### 2. OpenCL
- **Rejected because:** Declining ecosystem, wgpu is better abstraction

### 3. Vulkan Compute directly
- **Rejected because:** Too low-level, wgpu provides sufficient abstraction

### 4. ONNX Runtime for GPU ops
- **Rejected because:** Overkill for distance calculations

---

## Dependencies

### Required (wgpu backend)

```toml
[dependencies]
wgpu = "0.20"
pollster = "0.3"        # Async runtime for wgpu
bytemuck = "1.14"       # Safe transmutes for GPU buffers
```

### Optional (CUDA backend)

```toml
[dependencies]
cudarc = { version = "0.12", optional = true }
```

### Development

```toml
[dev-dependencies]
criterion = "0.5"       # Benchmarking
proptest = "1.4"        # Property-based testing
approx = "0.5"          # Floating-point comparisons
```

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| GPU speedup | ≥10x over CPU | Benchmark suite |
| Recall@10 | ≥99% for Flat, ≥95% for IVF | Recall tests |
| Memory efficiency | ≤4 bytes/dim with PQ | Memory profiling |
| API compatibility | 100% backward compatible | Integration tests |
| Platform coverage | Linux, Windows, macOS | CI pipeline |

---

## References

### Core Technologies
1. [FAISS Documentation](https://faiss.ai/)
2. [FAISS GPU Wiki](https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU)
3. [wgpu Documentation](https://wgpu.rs/)
4. [cudarc Crate](https://crates.io/crates/cudarc)
5. [metal-rs Crate](https://crates.io/crates/metal)
6. [WebGPU Specification](https://www.w3.org/TR/webgpu/)

### Academic Papers
7. [Product Quantization Paper](https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf)
8. [HNSW Paper](https://arxiv.org/abs/1603.09320)
9. [DiskANN Paper](https://proceedings.neurips.cc/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf)
10. [ScaNN Paper](https://arxiv.org/abs/1908.10396)

### Production Vector Database Implementations
11. [Qdrant - HNSW as dense vector index](https://qdrant.tech/articles/vector-search-resource-optimization/)
12. [Weaviate - "HNSW is the default"](https://docs.weaviate.io/weaviate/tutorials/vector-indexing-deep-dive)
13. [Elasticsearch - kNN is HNSW-based](https://discuss.elastic.co/t/is-indexing-with-hnsw-required-for-knn-to-work/349598)
14. [Vespa - HNSW for ANN](https://docs.vespa.ai/en/querying/approximate-nn-hnsw.html)
15. [pgvector - HNSW and IVFFlat](https://supabase.com/docs/guides/ai/vector-indexes/ivf-indexes)
16. [Milvus - IVF family + DiskANN](https://milvus.io/docs/index.md)
17. [OpenSearch - HNSW + IVF via FAISS](https://docs.opensearch.org/latest/mappings/supported-field-types/knn-methods-engines/)
18. [Milvus DiskANN](https://milvus.io/docs/diskann.md)

---

## Appendix A: GPU Shader Specifications

### A.1 WGSL Compute Shader Requirements

```wgsl
// Minimum WebGPU limits we rely on
// - maxStorageBuffersPerShaderStage: 8
// - maxStorageBufferBindingSize: 128MB (134217728 bytes)
// - maxComputeWorkgroupSizeX: 256
// - maxComputeInvocationsPerWorkgroup: 256
```

### A.2 CUDA Kernel Requirements

```cuda
// Minimum compute capability: 6.0 (Pascal)
// Recommended: 7.0+ (Volta) for Tensor Cores
// Required CUDA version: 11.0+
```

---

## Appendix B: Cargo.toml Changes

```toml
[package]
name = "redisearch-platform-core"
version = "0.2.0"

[features]
default = ["wgpu-backend"]

# GPU backends
wgpu-backend = ["wgpu", "pollster", "bytemuck"]
gpu-cuda = ["cudarc"]

# Index types
quantization = []
ivf = ["quantization"]
ivf-gpu = ["ivf", "wgpu-backend"]

# Full GPU support
full-gpu = ["wgpu-backend", "gpu-cuda", "ivf-gpu"]

[dependencies]
# Core
anyhow = "1.0"
thiserror = "1.0"

# Existing
hnsw_rs = "0.1"
redb = "1.0"
bincode = "1.3"

# GPU - wgpu (cross-platform)
wgpu = { version = "0.20", optional = true }
pollster = { version = "0.3", optional = true }
bytemuck = { version = "1.14", features = ["derive"], optional = true }

# GPU - CUDA (NVIDIA only)
cudarc = { version = "0.12", optional = true }

[target.'cfg(target_os = "macos")'.dependencies]
# Metal is accessed through wgpu, no additional deps needed

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
proptest = "1.4"
approx = "0.5"

[[bench]]
name = "gpu_benchmarks"
harness = false
required-features = ["wgpu-backend"]
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2024-12-23 | Rafael Escrich | Initial draft |

---

**Next Steps:**

1. Review and approve this ADR
2. Begin Phase 0 implementation
3. Set up GPU CI/CD pipeline
4. Create detailed technical specs for each phase

