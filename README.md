# 🚀 RedVector

[![CI/CD](https://github.com/rafaelescrich/redvector/actions/workflows/ci.yml/badge.svg)](https://github.com/rafaelescrich/redvector/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**High-Performance In-Memory Vector Database Built in Rust**

RedVector is an in-memory vector database that combines Redis compatibility with advanced vector search capabilities. Built on a Redis-compatible server (rsedis-style) and the Redisearch platform, it delivers predictable low-latency performance for AI applications, semantic search, RAG pipelines, and recommendation systems.

**Built on:**
- **Redis-Compatible Server**: Full Redis protocol implementation in Rust (150+ commands)
- **Redisearch Platform**: Vector search engine with HNSW indexing, GPU acceleration, and multi-vector support

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   RedVector = In-Memory Vector DB + Redis Protocol + REST/gRPC APIs         │
│                                                                              │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐    │
│   │   Strings   │   │   Vectors   │   │   REST API  │   │  gRPC API   │    │
│   │   Lists     │   │   HNSW      │   │   Port 8888 │   │  Port 50051 │    │
│   │   Sets      │   │   Cosine    │   │   JSON      │   │  Protobuf   │    │
│   │   Hashes    │   │   Euclidean │   │   Qdrant-   │   │  Qdrant-    │    │
│   └─────────────┘   └─────────────┘   └──compatible─┘   └──compatible─┘    │
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

- **150+ Redis Commands**: Full compatibility with Redis protocol (port 6379)
- **Data Structures**:
  - **Strings**: GET, SET, MGET, MSET, INCR, DECR, APPEND, GETSET, STRLEN
  - **Lists**: LPUSH, RPUSH, LPOP, RPOP, LRANGE, LINDEX, LLEN, LTRIM
  - **Sets**: SADD, SREM, SMEMBERS, SINTER, SUNION, SDIFF, SCARD, SISMEMBER
  - **Hashes**: HSET, HGET, HMSET, HGETALL, HDEL, HKEYS, HVALS, HLEN
  - **Sorted Sets**: ZADD, ZRANGE, ZRANK, ZSCORE, ZREM, ZCARD, ZCOUNT
- **Pub/Sub**: SUBSCRIBE, PUBLISH, PSUBSCRIBE, UNSUBSCRIBE
- **Transactions**: MULTI, EXEC, DISCARD, WATCH for atomic operations
- **Persistence**: RDB snapshots and AOF (Append-Only File) for data durability

### 🔍 Vector Search Features (RediSearch Compatible)

Built on **Redisearch Platform Core** with HNSW indexing:

- **FT.CREATE**: Create vector indexes with configurable HNSW parameters (m, ef_construction)
- **FT.ADD**: Add documents with vector embeddings to the index
- **FT.SEARCH**: K-nearest neighbor (KNN) similarity search with configurable ef_search
- **FT.INFO**: Get detailed index information and statistics
- **FT.DROP**: Delete indexes and collections
- **FT.DEL**: Delete individual documents from indexes
- **HNSW Backend**: Hierarchical Navigable Small World graph for fast approximate search
- **Automatic Backend Selection**: Linear scan for small datasets, HNSW for large datasets
- **Multi-Vector Support**: RVF2 format for ColPali/ColBERT-style multi-vector retrieval (optional)

### 🌐 REST API (Qdrant-Compatible)

- **Collection Management**:
  - `POST /api/collections/:name` - Create collection
  - `GET /api/collections` - List all collections
  - `GET /api/collections/:name` - Get collection info
  - `DELETE /api/collections/:name` - Delete collection
- **Vector Operations**:
  - `POST /api/collections/:name/points` - Upsert vectors
  - `GET /api/collections/:name/points/:id` - Get vector by ID
  - `POST /api/collections/:name/points/delete` - Delete vectors
  - `GET /api/collections/:name/search` - Search vectors
- **JSON Format**: All requests and responses use JSON

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

- **Zero GC Pauses**: Pure Rust implementation eliminates garbage collection
- **Predictable Latency**: Consistent P99 performance under load
- **Concurrent Operations**: Multi-threaded architecture for parallel processing
- **Memory Efficiency**: Optimized data structures for minimal memory footprint
- **Fast Index Updates**: Real-time index modifications without blocking
- **GPU Acceleration**: Optional wgpu (Vulkan/Metal/DX12) and CUDA backends for vector operations
- **SIMD Optimizations**: CPU-optimized distance metrics for faster similarity calculations
- **LRU Caching**: Hot vector cache for frequently accessed embeddings

### 💾 Persistence & Durability

- **RDB Snapshots**: Point-in-time snapshots of the Redis-compatible database
- **AOF (Append-Only File)**: Durable write-ahead logging for Redis data
- **redb Vector Storage**: Persistent storage for vectors and metadata using redb
- **HNSW Snapshots**: Periodic snapshots of HNSW index structure
- **Background Persistence**: Non-blocking save operations
- **Data Recovery**: Automatic recovery on server restart
- **S3 Storage**: Optional S3/GCS/MinIO integration for vector storage (feature flag)

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

Output:
```
🚀 RedVector v0.1.0 - In-Memory Vector Database
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔴 Redis Protocol: localhost:6379
📊 REST API:       http://localhost:8888
🔌 gRPC API:       http://localhost:50051
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

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

# Search
curl -X POST http://localhost:8888/api/collections/my_vectors/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, 0.3, ...], "limit": 10}'
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

1. **Redis-Compatible Server** (rsedis-style): Handles all Redis protocol commands, data structures, and persistence
2. **Redisearch Platform Core**: Provides vector search capabilities with HNSW indexing, GPU acceleration, and advanced features

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
│   │   • HNSW Vector Index (CPU)  • GPU Acceleration (wgpu/CUDA)         │   │
│   │   • RVF2 Multi-Vector Storage  • Quantization (SQ8, PQ)              │   │
│   │   • Cosine/Euclidean/Inner Product  • Persistent Index (redb)        │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                             │                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │         Redis-Compatible Key-Value Store (In-Memory)                 │   │
│   │    Strings • Lists • Sets • Hashes • Sorted Sets • Pub/Sub          │   │
│   │    • 150+ Redis Commands  • Transactions  • Persistence (RDB/AOF)    │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                             │                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                       Persistence Layer                              │   │
│   │              • RDB Snapshots  • AOF Append-Only File                 │   │
│   │              • redb Vector Storage  • HNSW Snapshots                  │   │
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
| **Built-in Cache** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Pub/Sub** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Transactions** | ✅ | ❌ | ❌ | ❌ | ✅ |
| **Self-Hosted** | ✅ | ✅ | ✅ | ❌ | ✅ |
| **Open Source** | ✅ | ✅ | ✅ | ❌ | ✅ |
| **In-Memory** | ✅ | ❌ | Partial | ❌ | ❌ |

---

## 🗺️ Roadmap

### v0.1.0 - Current ✅
- Redis-compatible server (rsedis-style) with 150+ commands
- Redisearch Platform Core integration
- In-memory vector storage with HNSW indexing
- RediSearch FT.* commands (FT.CREATE, FT.ADD, FT.SEARCH, FT.INFO, FT.DROP, FT.DEL)
- Integrated REST API (Qdrant-compatible)
- Integrated gRPC API
- Persistence (RDB, AOF, redb for vectors)

### v0.2.0 - GPU Acceleration
- wgpu backend (Vulkan/Metal/DX12) via Redisearch Platform Core
- CUDA backend for NVIDIA GPUs
- Apple Silicon Metal optimization
- GPU-accelerated distance metrics
- Flat and IVF indexes on GPU

### v1.0.0 - Production Ready
- IVF-SQ8 index (4x compression) via Redisearch Platform Core
- IVF-PQ index (32x compression) for maximum memory efficiency
- Memory-mapped vector storage with RVF2 format
- Full-text search integration (Redisearch rust-port)
- Production hardening and comprehensive benchmarks

### v2.0.0 - Enterprise (Closed Source)
- Distributed clustering
- Multi-node replication
- Cloud management console

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
# Run tests
cargo test --all-features

# Build with all features
cargo build --release --features full

# Run
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

Copyright (c) 2025, Rafael Escrich

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with 🦀 Rust • Powered by Redis-Compatible Server + Redisearch Platform • In-Memory Vector Database**

[Documentation](docs/) • [Issues](https://github.com/rafaelescrich/redvector/issues) • [Discussions](https://github.com/rafaelescrich/redvector/discussions)

</div>
