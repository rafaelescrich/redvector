# 🚀 RedVector

[![CI/CD](https://github.com/rafaelescrich/redvector/actions/workflows/ci.yml/badge.svg)](https://github.com/rafaelescrich/redvector/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**The Redis-Compatible Vector Database Built in Rust**

RedVector combines the power of Redis with vector search capabilities. Use your existing Redis clients to build AI applications with semantic search, RAG pipelines, and recommendation systems.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   RedVector = Redis Protocol + Vector Search + REST/gRPC APIs               │
│                                                                              │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐    │
│   │   Strings   │   │   Vectors   │   │   REST API  │   │  gRPC API   │    │
│   │   Lists     │   │   HNSW      │   │   Port 8888 │   │  Port 50051 │    │
│   │   Sets      │   │   Cosine    │   │   JSON      │   │  Protobuf   │    │
│   │   Hashes    │   │   Euclidean │   │   Qdrant-   │   │  Qdrant-    │    │
│   └─────────────┘   └─────────────┘   └──compatible─┘   └──compatible─┘    │
│                                                                              │
│   ONE SERVER • 50+ CLIENT LANGUAGES • THREE PROTOCOLS                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Why RedVector?

### 🔌 Three APIs, One Server
```bash
# Redis Protocol (port 6379) - Works with any Redis client
redis-cli SET hello world

# REST API (port 8888) - Qdrant-compatible
curl -X POST http://localhost:8888/api/collections/my_vectors \
  -d '{"vector_size": 384, "distance": "Cosine"}'

# gRPC API (port 50051) - High-performance
grpcurl -plaintext localhost:50051 redvector.VectorService/Search
```

### 🦀 Pure Rust = No GC Pauses
Unlike Go-based vector databases (Milvus, Weaviate), RedVector has **zero garbage collection**:
- ✅ Predictable P99 latency
- ✅ No stop-the-world pauses
- ✅ Consistent performance under load

### 📦 One Server, Not Three
Other setups require: `Vector DB + Redis Cache + Message Queue`

RedVector gives you: **Everything in one place**

```redis
# Atomic operation: Update product + embedding + invalidate cache + notify
MULTI
HSET product:123 name "New Name" price 99.99
FT.ADD products product:123 1.0 FIELDS vector "0.1,0.2,..."
DEL cache:product:123
PUBLISH product_updates "123"
EXEC
```

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
🚀 RedVector v0.1.0 - Redis-Compatible Vector Database
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

---

## 📊 Features

### ✅ Redis Compatibility (150+ Commands)
| Category | Commands |
|----------|----------|
| **Strings** | GET, SET, MGET, MSET, INCR, APPEND, ... |
| **Lists** | LPUSH, RPUSH, LPOP, RPOP, LRANGE, ... |
| **Sets** | SADD, SREM, SMEMBERS, SINTER, SUNION, ... |
| **Hashes** | HSET, HGET, HMSET, HGETALL, ... |
| **Sorted Sets** | ZADD, ZRANGE, ZRANK, ZSCORE, ... |
| **Pub/Sub** | SUBSCRIBE, PUBLISH, PSUBSCRIBE, ... |
| **Transactions** | MULTI, EXEC, DISCARD, WATCH |
| **Persistence** | SAVE, BGSAVE, AOF |

### ✅ Vector Search (RediSearch Compatible)
| Command | Description |
|---------|-------------|
| `FT.CREATE` | Create a vector index |
| `FT.ADD` | Add document with vector |
| `FT.SEARCH` | Similarity search (KNN) |
| `FT.INFO` | Index information |
| `FT.DROP` | Delete index |

### ✅ REST API (Qdrant-Compatible)
| Endpoint | Description |
|----------|-------------|
| `POST /api/collections/:name` | Create collection |
| `GET /api/collections` | List collections |
| `POST /api/collections/:name/points` | Upsert vectors |
| `GET /api/collections/:name/search` | Search vectors |
| `DELETE /api/collections/:name` | Delete collection |

### ✅ gRPC API
| Service | Methods |
|---------|---------|
| `VectorService` | CreateCollection, Upsert, Search, GetCollectionInfo, DeleteCollection |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RedVector Architecture                               │
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
│   │                    HNSW Vector Index Engine                          │   │
│   │         • Cosine Similarity  • Euclidean  • Inner Product           │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                             │                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │              Redis-Compatible Key-Value Store                        │   │
│   │    Strings • Lists • Sets • Hashes • Sorted Sets • Pub/Sub          │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                             │                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                       Persistence Layer                              │   │
│   │              • RDB Snapshots  • AOF Append-Only File                 │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🆚 Comparison with Other Vector Databases

| Feature | RedVector | Qdrant | Milvus | Pinecone | pgvector |
|---------|-----------|--------|--------|----------|----------|
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

---

## 🗺️ Roadmap

### v0.1.0 - Current ✅
- Redis protocol compatibility (150+ commands)
- HNSW vector index (CPU)
- RediSearch FT.* commands
- Integrated REST API
- Integrated gRPC API
- Persistence (RDB, AOF)

### v0.2.0 - GPU Acceleration
- wgpu backend (Vulkan/Metal/DX12)
- CUDA backend for NVIDIA
- Apple Silicon native support

### v1.0.0 - Production Ready
- IVF-SQ8 (4x compression)
- IVF-PQ (32x compression)
- Memory-mapped vector storage
- Production hardening

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

## 📄 License

Copyright (c) 2024-2025, Rafael Escrich

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with 🦀 Rust • Compatible with 🔴 Redis • APIs like 🟣 Qdrant**

[Documentation](docs/) • [Issues](https://github.com/rafaelescrich/redvector/issues) • [Discussions](https://github.com/rafaelescrich/redvector/discussions)

</div>
