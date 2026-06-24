# 🚀 RedVector

[![CI/CD](https://github.com/rafaelescrich/redvector/actions/workflows/ci.yml/badge.svg)](https://github.com/rafaelescrich/redvector/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**High-Performance In-Memory Vector Database Built in Rust**

RedVector is an in-memory vector database that combines Redis compatibility with advanced vector search capabilities. Built on a Redis-compatible server (rsedis-style) and the Redisearch platform, it targets predictable low-latency performance for AI applications, semantic search, RAG pipelines, and recommendation systems.

**Project status (v0.1.0):** Early development. Redis wire protocol and a large command subset are implemented; vector search (`FT.*` with `--features vector-search`) and optional REST/gRPC (`--features full`) work in-tree. Some README items below are **roadmap** or **platform-crate** capabilities—see [Roadmap](#roadmap) and [Current limitations](#current-limitations).

**Built on:**
- **Redis-Compatible Server**: RESP-based server in Rust derived from the [rsedis](https://github.com/seppo0010/rsedis) lineage—about **150 commands** advertised via `COMMAND`, not full Redis parity with every edge case
- **Redisearch Platform Core**: In-repo vector/HNSW stack; optional **GPU**, **RVF2**, and **S3** exist as feature-gated modules in that crate and are **not** enabled by the main `redvector` crate’s `full` feature today

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   RedVector = In-Memory Vector DB + Redis Protocol + REST/gRPC APIs         │
│                                                                              │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐    │
│   │   Strings   │   │   Vectors   │   │   REST API  │   │  gRPC API   │    │
│   │   Lists     │   │   HNSW      │   │   Port 8888 │   │  Port 50051 │    │
│   │   Sets      │   │   Cosine    │   │   JSON      │   │  Protobuf   │    │
│   │   Hashes    │   │   Euclidean │   │ Qdrant-like  │   │ Qdrant-like  │    │
│   └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘    │
│                                                                              │
│   ONE SERVER • 50+ CLIENT LANGUAGES • THREE PROTOCOLS • IN-MEMORY          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### 🎯 Core Vector Database Features

- **In-Memory Storage**: All data and vectors stored in RAM for ultra-low latency
- **HNSW Index**: Hierarchical Navigable Small World graph for fast approximate nearest neighbor search
- **Multiple Distance Metrics**: 
  - Cosine Similarity
  - Euclidean Distance
  - Inner Product
- **High-Dimensional Vectors**: Support for vectors of any dimension
- **Real-Time Updates**: Add, update, and delete vectors with instant index updates
- **Batch Operations**: Efficient bulk insert and update operations

### 🔴 Redis Protocol Compatibility

- **~150 commands**: Broad rsedis-style subset on port **6379** (see `COMMAND` output); behavior may differ from Redis on persistence, replication, and admin commands
- **Data structures** (representative):
  - **Strings**: GET, SET, MGET, MSET, INCR, DECR, APPEND, GETSET, STRLEN
  - **Lists**: LPUSH, RPUSH, LPOP, RPOP, LRANGE, LINDEX, LLEN, LTRIM
  - **Sets**: SADD, SREM, SMEMBERS, SINTER, SUNION, SDIFF, SCARD, SISMEMBER
  - **Hashes**: HSET, HGET, HMSET, HGETALL, HDEL, HKEYS, HVALS, HLEN
  - **Sorted Sets**: ZADD, ZRANGE, ZRANK, ZSCORE, ZREM, ZCARD, ZCOUNT
- **Pub/Sub**: SUBSCRIBE, PUBLISH, PSUBSCRIBE, UNSUBSCRIBE
- **Transactions**: MULTI, EXEC, DISCARD, WATCH for atomic operations
- **Persistence**: RDB/AOF-related commands exist; **durable RDB save / AOF rewrite to disk is still incomplete** (see [Current limitations](#current-limitations))

### 🔍 Vector Search Features (RediSearch Compatible)

Built on **Redisearch Platform Core** with HNSW indexing:

- **FT.CREATE**: Create vector indexes; schema parsing is simplified (dimension from `VECTOR(dim)` in `SCHEMA`)
- **FT.ADD**: Add documents with vector embeddings to the index
- **FT.SEARCH**: KNN-style search; query vector as comma-separated floats (see tests / handler for options)
- **FT.INFO**: Get detailed index information and statistics
- **FT.DROP**: Delete indexes and collections
- **FT.DEL**: Delete individual documents from indexes
- **HNSW Backend**: Hierarchical Navigable Small World graph for fast approximate search
- **Multi-vector (RVF2)**: Optional in **redisearch-platform-core** (`rvf2` feature); not wired through the default `redvector` binary features yet

### 🌐 REST API (Qdrant-inspired)

Implemented in `src/api.rs` when built with `--features full` (or `api-server`). JSON bodies/responses; not a full Qdrant clone.

- **Meta**: `GET /health`, `GET /api/info`, `GET /` (HTML API docs)
- **Collections**:
  - `POST /api/collections/:name` — create collection (JSON body)
  - `GET /api/collections` — list collections
  - `GET /api/collections/:name` — collection info
  - `DELETE /api/collections/:name` — delete collection
- **Points**:
  - `POST /api/collections/:name/points` — upsert points (JSON body)
  - `GET /api/collections/:name/search?vector=0.1,0.2,...&limit=10` — similarity search (**query string**, comma-separated floats)

Per-point GET/delete routes are **not** exposed yet.

### 🟣 Qdrant-compatible REST API

A separate, drop-in Qdrant-compatible server (verified with the official Python
`qdrant-client==1.17.0`, e.g. Open WebUI) runs on **port 6333** by default
(override with `QDRANT_COMPAT_PORT`) when built with `--features full`. Unlike
real Qdrant it accepts **arbitrary string point IDs** (not just uint/UUID).
Implemented in `src/qdrant_api.rs`. See [docs/QDRANT_COMPAT.md](docs/QDRANT_COMPAT.md)
for endpoints, scoring semantics, filtering, and caveats.

### 🔌 gRPC API

- **VectorService**: High-performance gRPC interface (port 50051)
- **Methods**:
  - `CreateCollection` - Create new vector collection
  - `Upsert` - Insert or update vectors
  - `Search` - Perform similarity search
  - `GetCollectionInfo` - Retrieve collection metadata
  - `DeleteCollection` - Remove collection
- **Protobuf**: Efficient binary protocol for maximum throughput

### ⚡ Performance Features

- **Zero GC Pauses**: Pure Rust implementation eliminates garbage collection pauses in the server process
- **Concurrent Operations**: Multi-threaded architecture for parallel processing
- **Memory Efficiency**: In-memory structures tuned for embedding workloads
- **HNSW**: Approximate nearest-neighbor search when `hnsw-backend` is enabled
- **GPU (roadmap / platform crate)**: Optional `gpu-wgpu` / `gpu-cuda` in **redisearch-platform-core**—not compiled into the default `redvector` `full` feature set
- **SIMD**: Additional SIMD distance kernels are planned (see platform crate / ADRs); not the primary story for the main binary yet
- **LRU**: Available in platform storage paths; integration depth depends on configuration and features

### 💾 Persistence & Durability

- **Redis-side RDB/AOF**: Work in progress; do not rely on `SAVE`/`BGSAVE`/`BGREWRITEAOF` for production durability yet
- **redisearch-platform-core + redb**: The platform crate includes **redb**-backed storage and related design; wiring and defaults for the top-level server are still evolving
- **S3 / object store**: Optional **`s3`** feature in the platform crate (AWS SDK), not enabled by `redvector`’s default `full` feature

### 🔧 Developer Experience

- **Multi-Protocol Support**: Use Redis clients, REST, or gRPC
- **Language Agnostic**: Works with any language that has a Redis client
- **Docker Support**: Easy deployment with containerization
- **Self-Hosted**: Full control over your data and infrastructure
- **Open Source**: Apache 2.0 licensed

---

## 🚀 Quick Start

### Using Docker

```bash
docker build -t redvector:latest .
docker run -d -p 6379:6379 -p 8888:8888 -p 50051:50051 redvector:latest
```

### Building from Source

```bash
# Clone
git clone https://github.com/rafaelescrich/redvector.git
cd redvector

# Build with all features (Redis + Vector Search + REST + gRPC)
cargo build --release --features full

# Run
./target/release/redvector
```

Output (with `--features full`):
```
🚀 RedVector v0.1.0 - Redis-Compatible Vector Database
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔴 Redis Protocol: localhost:6379
📊 REST API:       http://localhost:8888
🔌 gRPC API:       http://localhost:50051
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

With **default features** (no `api-server`), only the Redis port is started; build with `--features full` for REST and gRPC.

### Connect with Any Redis Client

```python
import redis
r = redis.Redis()

# Standard Redis commands work
r.set("hello", "world")
print(r.get("hello"))  # b'world'

# Vector search with FT.* commands
r.execute_command("FT.CREATE", "myindex", "SCHEMA", "embedding", "VECTOR(384)")
r.execute_command("FT.ADD", "myindex", "doc1", "1.0", "FIELDS", "vector", "0.1,0.2,...")
results = r.execute_command("FT.SEARCH", "myindex", "0.1,0.2,...")
```

### Using REST API

```bash
# Create a collection
curl -X POST http://localhost:8888/api/collections/my_vectors \
  -H "Content-Type: application/json" \
  -d '{"vector_size": 384, "distance": "Cosine"}'

# Add vectors
curl -X POST http://localhost:8888/api/collections/my_vectors/points \
  -H "Content-Type: application/json" \
  -d '{
    "points": [
      {"id": 1, "vector": [0.1, 0.2, 0.3, ...]}
    ]
  }'

# Search (GET + comma-separated floats)
curl -G "http://localhost:8888/api/collections/my_vectors/search" \
  --data-urlencode "vector=0.1,0.2,0.3" \
  --data-urlencode "limit=10"
```

### Using gRPC API

```bash
# Using grpcurl
grpcurl -plaintext localhost:50051 redvector.VectorService/CreateCollection \
  -d '{"name": "my_vectors", "vector_size": 384, "distance": "Cosine"}'
```

---

## 🏗️ Architecture

RedVector is built on two core components:

1. **Redis-Compatible Server** (rsedis-style): RESP, core data structures, and a large command subset; persistence and replication are not complete
2. **Redisearch Platform Core**: Library crate in this repo for vector/HNSW and optional GPU, RVF2, S3, redb-backed storage—used by `FT.*` when `vector-search` is enabled

**Important:** With `--features full`, the REST/gRPC servers currently use a **separate** `Database` instance from the Redis acceptor path (see `src/main.rs`). Data ingested over Redis is not visible to REST/gRPC and vice versa until that wiring is unified.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RedVector In-Memory Architecture                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                   │
│   │ Redis Proto  │   │   REST API   │   │   gRPC API   │                   │
│   │  Port 6379   │   │  Port 8888   │   │  Port 50051  │                   │
│   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                   │
│          │                  │                  │                            │
│          └──────────────────┼──────────────────┘                            │
│                             ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │              Redisearch Platform Core                                │   │
│   │   • HNSW (CPU) • Optional GPU / RVF2 / redb / S3 (platform features)   │   │
│   │   • Cosine / Euclidean / inner product (metric support varies by path) │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                             │                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │         Redis-Compatible Key-Value Store (In-Memory)                 │   │
│   │    Strings • Lists • Sets • Hashes • Sorted Sets • Pub/Sub          │   │
│   │    • ~150 commands • Transactions • RDB/AOF (in progress)             │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                             │                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                       Persistence Layer                              │   │
│   │    • RDB/AOF (WIP) • redb / platform persistence (see crate features)   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🆚 Comparison with Other Vector Databases

| Feature | RedVector | Qdrant | Milvus | Pinecone | pgvector |
|---------|-----------|--------|--------|----------|----------|
| **Type** | In-Memory | Disk-based | Hybrid | Cloud | PostgreSQL Extension |
| **Language** | 🦀 Rust | 🦀 Rust | Go | ? | C |
| **Redis Protocol** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **REST API** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **gRPC API** | ✅ | ✅ | ✅ | ❌ | ❌ |
| **No GC Pauses** | ✅ | ✅ | ❌ | ? | ✅ |
| **Built-in Cache** | Partial (platform) | ❌ | ❌ | ❌ | ❌ |
| **Pub/Sub** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Transactions** | ✅ | ❌ | ❌ | ❌ | ✅ |
| **Self-Hosted** | ✅ | ✅ | ✅ | ❌ | ✅ |
| **Open Source** | ✅ | ✅ | ✅ | ❌ | ✅ |
| **In-Memory** | ✅ | ❌ | Partial | ❌ | ❌ |

---

## 🗺️ Roadmap

### v0.1.0 — current release (**in progress**, not “done”)

Shipped in source today:

- [x] Redis-compatible server (rsedis-style) with ~150 commands in `COMMAND`
- [x] `FT.CREATE` / `FT.ADD` / `FT.SEARCH` / `FT.INFO` / `FT.DROP` / `FT.DEL` behind `--features vector-search` (with `hnsw-backend` for HNSW in the main path)
- [x] Optional REST + gRPC in the same binary (`--features full` / `api-server`)
- [x] Docker image builds with `--features full`

Still open for v0.1.x (was incorrectly implied as finished before):

- [ ] **Single shared database** between Redis and REST/gRPC (today: separate `Database` for APIs—see `src/main.rs`)
- [ ] **Production persistence**: complete RDB serialization/deserialization, real `SAVE`/`BGSAVE` to disk, AOF rewrite (several `TODO`s in `command/src/command.rs`)
- [ ] **Integration tests** in CI for `FT.*`, REST, and gRPC (root crate currently runs **0** unit tests; add coverage over time)
- [ ] **REST parity**: per-point get/delete, JSON `search` body if desired, distance metric plumbing (REST currently reports Cosine in listing)
- [ ] **Wire platform features** into the default binary when ready: optional `gpu-*`, `rvf2`, `s3` from **redisearch-platform-core**

### v0.2.0 — GPU acceleration (platform + binary)

- [ ] Enable and document `gpu-wgpu` / `gpu-cuda` (or `gpu-all`) from the platform crate in the main `redvector` feature set
- [ ] GPU distance metrics and benchmarks
- [ ] Flat / IVF on GPU (as feasible)

### v1.0.0 — production ready

- [ ] IVF-SQ8 / IVF-PQ and related compression paths where applicable
- [ ] RVF2 / memory-mapped multi-vector workflows productized
- [ ] Full-text search (RediSearch-style) if scoped
- [ ] Hardening, fuzzing, and benchmark suite

### v2.0.0 — enterprise (closed source) *(vision)*

- [ ] Distributed clustering, replication, managed console

---

## Current limitations

- **Split datastore** when using `full`: Redis and HTTP/gRPC do not share one `Database` yet.
- **Persistence**: treat the server as **in-memory** for production until RDB/AOF work is finished.
- **Compatibility**: Redis clients work for many commands; do not assume identical semantics to Redis 7.x for every command.
- **Tests**: run `cargo test --all-features` (and add crate/integration tests as they land); today the binary crate contributes no tests.

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [GPU Acceleration Plan](docs/adr/ADR-001-GPU-ACCELERATION.md) | GPU implementation roadmap |
| [Architecture Advantages](docs/adr/ADR-002-ARCHITECTURE-ADVANTAGES.md) | Why RedVector's design is unique |
| [Docker Guide](DOCKER.md) | Container deployment |

---

## 🤝 Contributing

Contributions are welcome! See our [Architecture Decision Records](docs/adr/) for design context.

```bash
# Run tests (expand coverage over time)
cargo test --all-features

# Build with all features (Redis + FT.* + REST + gRPC)
cargo build --release --features full

# Run (optional config file as first arg)
./target/release/redvector
```

---

## 🙏 Acknowledgments

RedVector is inspired by and built upon the excellent work of the open-source community:

- **[rsedis](https://github.com/seppo0010/rsedis)**: Redis re-implemented in Rust by [Sebastian Waisbrot](https://github.com/seppo0010). The rsedis project provided significant inspiration for the Redis-compatible server implementation.

- **[RediSearch](https://github.com/RediSearch/RediSearch)**: A query and indexing engine for Redis, providing secondary indexing, full-text search, and vector similarity search. RediSearch's design and feature set inspired the vector search capabilities in RedVector.

We are grateful to the maintainers and contributors of these projects for their valuable work in the open-source ecosystem.

---

## 📄 License

Copyright (c) 2025–2026, Rafael Escrich

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with 🦀 Rust • Powered by Redis-Compatible Server + Redisearch Platform • In-Memory Vector Database**

[Documentation](docs/) • [Issues](https://github.com/rafaelescrich/redvector/issues) • [Discussions](https://github.com/rafaelescrich/redvector/discussions)

</div>
