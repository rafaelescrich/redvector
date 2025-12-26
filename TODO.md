# RedVector Roadmap

## ✅ Completed Features

### Redis Compatibility
- **150+ commands** implemented (GET, SET, MGET, MSET, LPUSH, RPUSH, SADD, ZADD, HSET, PUBLISH, SUBSCRIBE, MULTI, EXEC, etc.)
- **30+ configuration options** with CONFIG GET/SET support
- **Full INFO command** with all major sections
- **Persistence**: RDB snapshots, AOF append-only file

### Vector Search
- **FT.CREATE** - Create vector indexes with HNSW backend
- **FT.ADD** - Add documents with vector embeddings
- **FT.SEARCH** - KNN similarity search
- **FT.INFO** - Index information
- **FT.DROP** - Delete indexes

### APIs
- **Redis Protocol** (port 6379) - Compatible with 50+ client libraries
- **REST API** (port 8888) - Qdrant-compatible endpoints
- **gRPC API** (port 50051) - High-performance Protobuf API

---

## 🔨 In Progress

### v0.2.0 - GPU Acceleration
- [ ] wgpu backend (Vulkan/Metal/DX12)
- [ ] CUDA backend for NVIDIA
- [ ] Apple Silicon Metal optimization
- [ ] GPU-accelerated distance metrics

---

## 📋 Planned

### v1.0.0 - Production Ready
- [ ] IVF-Flat index
- [ ] IVF-SQ8 index (4x compression)
- [ ] IVF-PQ index (32x compression)
- [ ] Memory-mapped vector storage
- [ ] Production hardening and benchmarks

### v2.0.0 - Enterprise (Closed Source)
- [ ] Distributed clustering
- [ ] Multi-node replication
- [ ] Sharding and partitioning
- [ ] Cloud management console
- [ ] Enterprise support

---

## ⚠️ Known Limitations

### Not Yet Implemented
- Lua scripting (EVAL, EVALSHA, SCRIPT)
- Cluster mode commands
- Full replication (SYNC, PSYNC, REPLCONF)
- Some advanced replication configs

These features are planned for future versions.
