# RedVector Launch Kit

Repo: https://github.com/rafaelescrich/redvector
Local source: graph-agentic-rag/redvector
Language policy: UnblockTheChain = en-US for all communications.

## One-line pitch

RedVector is a Redis-compatible, in-memory vector database written in Rust — one binary, three protocols (RESP/REST/gRPC), HNSW vector search, and a drop-in Qdrant-compatible API.

## The honest 30-second pitch (use everywhere, consistency matters)

RedVector = Redis you already know + vector search you need, in a single Rust binary. Point any Redis client at it and you get ~150 commands plus FT.* vector search (HNSW, cosine/euclidean/inner product). It also speaks a Qdrant-compatible REST API (verified with the official qdrant-client), so tools like Open WebUI work out of the box. Apache 2.0, self-hosted, Docker or cargo build. v0.1.0: early stage, honest about limitations — RDB/AOF durability is still incomplete.

## Key facts to reuse (from README + benchmark)

- ~150 Redis commands on port 6379 (rsedis lineage)
- FT.CREATE / FT.ADD / FT.SEARCH / FT.INFO / FT.DROP / FT.DEL vector commands
- REST API on 8888, gRPC on 50051, Qdrant-compatible on 6333
- Qdrant compat accepts arbitrary string point IDs (unlike real Qdrant)
- Benchmark (i7-8750H, 32GB, GTX 1060, Nvidia 10-Q RAG test):
  - ~18.5ms avg query latency (incl. local ONNX embedding + search)
  - 100% retrieval accuracy on financial queries
  - SIMD cosine 9.42x speedup over scalar (768-dim), euclidean 4.08x
  - ~2.2MB disk footprint for the test index
- Honest limitations: RDB/AOF persistence WIP, GPU/RVF2/S3 exist in the platform crate but aren't in the default feature set

## Target communities and angles

### Hacker News (Show HN)
Title options:
- "Show HN: RedVector – Redis-compatible vector database in Rust"
- "Show HN: A single-binary Rust vector DB that speaks Redis, REST, gRPC, and Qdrant's API"
Post as text post with 3 short paragraphs: what it is, why (one binary instead of Redis+vector DB), honest status. HN rewards candor — lead with v0.1.0 limitations. Best posting window: Tue-Thu, 8-10am US Eastern. Reply to every comment in the first 2 hours.

### r/rust
Angle: systems engineering. SIMD distance kernels (9.4x cosine speedup), zero-GC latency, redb storage, feature-gated architecture. Rust devs love honest "what's hard" details. Mention rsedis lineage. Flair: "Project".

### r/Database, r/dataengineering
Angle: consolidation story — replace Redis + a dedicated vector DB with one binary. Cost/ops simplicity. Benchmark numbers from RAG_BENCHMARK_REPORT.md.

### r/LocalLLaMA, r/MachineLearning, r/vectordatabase
Angle: RAG tooling. Qdrant-compatible API means Open WebUI works out of the box. Local-first, no cloud dependency, runs on a laptop. The Nvidia 10-Q benchmark story is the hook (real SEC filing, 100% retrieval accuracy).

### AI/ML Discords + forums
- Latent Space, MLOps Community, Qdrant/Weaviate adjacent Discords: short intro + link, offer to answer questions.
- Hugging Face: post in relevant Spaces/discussions about local RAG stacks.

### Dev.to / Medium
Full launch post (draft below).

### X/Twitter
Thread draft below. Tag @rustlang-adjacent accounts sparingly; one quote-tweet from a known Rust account is worth 10 posts.

## X/Twitter thread draft (6 posts)

1/ I built RedVector: a Redis-compatible vector database in Rust. One binary — RESP, REST, gRPC, and a drop-in Qdrant-compatible API. Apache 2.0, self-hosted.

2/ Why? Most RAG stacks run Redis for cache + a separate vector DB for embeddings. RedVector merges them: ~150 Redis commands + FT.* vector search (HNSW, cosine/euclidean) on the same port your Redis client already speaks.

3/ Verified with the official qdrant-client — tools like Open WebUI connect out of the box. Bonus: unlike real Qdrant, it accepts arbitrary string point IDs.

4/ Benchmark on a consumer laptop (i7-8750H): ~18.5ms avg query latency on a real Nvidia 10-Q RAG workload, 100% retrieval accuracy, SIMD cosine distance 9.4x faster than scalar.

5/ Honest status: v0.1.0, early stage. RDB/AOF durability is incomplete — don't trust it for production durability yet. Roadmap: persistence, GPU kernels, multi-vector.

6/ Try it: github.com/rafaelescrich/redvector — Docker one-liner or cargo build --features full. Feedback and issues welcome.

## Reddit post draft (r/rust + r/Database variant)

Title: RedVector: Redis-compatible in-memory vector database in Rust (single binary, three protocols)

Body:

I built RedVector to collapse a common stack — Redis for hot data plus a dedicated vector DB for embeddings — into one Rust binary.

What it does today:
- RESP protocol with ~150 Redis commands (rsedis lineage), port 6379
- Vector search via FT.* commands (HNSW index, cosine/euclidean/inner product)
- REST API (port 8888), gRPC (port 50051), and a Qdrant-compatible REST API (port 6333) verified against the official Python qdrant-client — Open WebUI works out of the box
- SIMD distance kernels: 9.4x speedup on cosine (768-dim) vs scalar on an i7-8750H
- Docker image or cargo build --release --features full

Honest limitations (v0.1.0): RDB/AOF durability is still incomplete; GPU, multi-vector (RVF2), and S3 exist in the platform crate but aren't wired into the default feature set yet.

Benchmark report with a real SEC 10-Q RAG workload is in the repo. I'd especially love feedback on the Qdrant-compat layer and the FT.* command surface.

github.com/rafaelescrich/redvector

## Show HN draft (plain text, 3 paragraphs)

RedVector is an in-memory vector database written in Rust that speaks the Redis wire protocol. Point any Redis client at it and you get ~150 familiar commands plus FT.* vector search (HNSW) — the idea is to collapse the common "Redis + separate vector DB" RAG stack into one binary.

It also exposes REST (8888), gRPC (50051), and a Qdrant-compatible REST API (6333) verified with the official Python qdrant-client, so existing tooling (e.g. Open WebUI) connects without changes. There's a benchmark report in the repo: on a consumer laptop it sustains ~18.5ms end-to-end RAG queries over an Nvidia 10-Q filing with 100% retrieval accuracy on the test set, and its SIMD cosine kernel is ~9.4x faster than scalar at 768 dimensions.

Status is early (v0.1.0): RDB/AOF durability is incomplete, and GPU/multi-vector support exists in the platform crate but isn't enabled by default features. I'd love feedback on the API surfaces and what would make you actually run this. Docker one-liner and cargo build instructions in the README.

## Launch checklist (in order)

1. Polish GitHub repo: topics/tags (vector-database, rust, redis, qdrant, rag, hnsw, semantic-search), good social preview image, crates.io publish if planned
2. Make sure the quick-start actually works from a clean machine (docker build + cargo build) — first comment on HN is always "didn't compile"
3. Post Show HN Tue-Thu morning ET; be online for 2 hours
4. Same day: r/rust post, r/Database post
5. Day 2: r/LocalLLaMA + dev.to article (expand the Show HN text into a full post with the benchmark story)
6. Day 3: X thread + Discord communities
7. Week 2: follow-up blog post with whatever feedback/feature requests the launch generated — second wave of attention

## Suggested cron job for unblockthechain-technical-research

Add a weekly job that monitors RedVector GitHub issues/stars, HN/Reddit mentions, and drafts responses + next-step release notes into ops-hub/companies/unblockthechain/releases/.
