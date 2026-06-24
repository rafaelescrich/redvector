# Qdrant-compatible REST API

RedVector ships a Qdrant-compatible REST layer so the official Python
`qdrant-client` (verified against **v1.17.0**, as used by Open WebUI) works
against it unmodified.

It runs as a **separate HTTP server** alongside the Redis protocol, the native
`/api` REST server, and the gRPC server. It is in-memory only and reuses
`HnswVectorIndex` for approximate nearest-neighbour search.

## Running

```bash
# build with the API server + HNSW backend
cargo build --features full

# run (also starts Redis, native REST :8888, gRPC :50051)
cargo run --features full
# or: ./target/debug/redvector redvector.conf
```

### Port

- Default: **6333** (Qdrant's REST default).
- Override with the `QDRANT_COMPAT_PORT` environment variable:

```bash
QDRANT_COMPAT_PORT=7777 cargo run --features full
```

Point a client at it:

```python
from qdrant_client import QdrantClient
client = QdrantClient(url="http://localhost:6333")
```

## Why use this instead of real Qdrant

Real Qdrant only accepts point IDs that are unsigned integers or UUIDs.
Open WebUI sends `item["id"]` which is often a non-UUID hash string, so real
Qdrant rejects it. **RedVector accepts ANY string id** (stored verbatim) and
maps it internally to a `u64` for the HNSW index.

## Supported endpoints

| Method | Path | Notes |
|--------|------|-------|
| GET  | `/` | Returns `{"title":...,"version":"1.17.0","commit":"redvector"}` (semver for client version check). |
| GET  | `/healthz`, `/readyz`, `/livez` | 200 `healthz check passed`. |
| GET  | `/collections` | List collection names. |
| GET  | `/collections/{name}` | Collection info (404 with Qdrant error JSON if missing). |
| GET  | `/collections/{name}/exists` | `{"exists": bool}`. |
| PUT  | `/collections/{name}` | Create. Body: `{"vectors":{"size":N,"distance":"Cosine"},"hnsw_config":{"m":N}?}`. |
| DELETE | `/collections/{name}` | Delete (idempotent). |
| PUT  | `/collections/{name}/index` | Create payload index. **No-op** (in-memory scan), returns completed. |
| PUT  | `/collections/{name}/points` | Upsert. Body: `{"points":[{"id":<str\|int>,"vector":[...],"payload":{...}}]}`. |
| POST | `/collections/{name}/points/query` | KNN. Body: `{"query":[...]\|{"nearest":[...]}, "limit":N, "filter":{...}?}`. |
| POST | `/collections/{name}/points/scroll` | List points (optionally filtered), no vector search. |
| POST | `/collections/{name}/points/delete` | Delete by `{"points":[id,...]}` or `{"filter":{...}}`. |

All responses use Qdrant's envelope `{"result": R, "status": "ok", "time": 0.0}`
(except `/` and the health checks). Errors use `{"status":{"error":"..."},"time":0.0}`.

## Distance / scoring

`distance` may be `Cosine` (default), `Euclid`, or `Dot`.

The underlying HNSW index computes cosine **distance** (0 = identical,
2 = opposite). The query endpoint converts this to the score qdrant-client
expects:

- **Cosine / Dot**: cosine **similarity** in `[-1, 1]` = `1 - distance`
  (higher = more similar). Open WebUI relies on this (`(score+1)/2`).
- **Euclid**: raw distance (lower = better).

> Note: the HNSW backend is cosine-only internally; `Euclid`/`Dot` are accepted
> and reported in collection config, but ranking is computed from cosine
> distance. Cosine is fully correct and is what Open WebUI uses.

## Filtering

Supported `Filter` subset:

```json
{"must":[{"key":"metadata.hash","match":{"value":"x"}}],
 "should":[...], "must_not":[...]}
```

- A point matches if it satisfies **all** `must`, **none** of `must_not`, and
  (if `should` is non-empty) **at least one** `should`.
- `key` is resolved as a dotted path against the payload, e.g.
  `metadata.hash` -> `payload["metadata"]["hash"]`.
- Match values may be strings, numbers, or bools (numeric comparison is
  type-loose, so `1` matches `1.0`).

Applied to `query`, `scroll`, and `delete`.

## Persistence

Collections are persisted to disk so they survive restarts.

- Data dir: `QDRANT_COMPAT_DATA_DIR` (default `/data/qdrant`); created if missing.
- One JSON file per collection (`<data_dir>/<collection>.json`) holding the
  dimension, distance, and every point `{id, vector, payload}`. Writes are
  atomic (temp file + rename) and happen on every mutation (create / upsert /
  delete points / delete collection).
- On startup each file is loaded and the **HNSW index is rebuilt from the stored
  vectors** (the graph itself is not serialized), so load time scales with the
  number of vectors.

## Caveats / not implemented

- HNSW removals tombstone the mapping (the vector stays in the graph but is not
  returned). This matches the existing `HnswVectorIndex::remove` behaviour.
- Payload field indexes are a no-op (full in-memory scan is used).
- Only the dense single-vector config is supported (no named/sparse/multi
  vectors, no quantization, no on-disk).
- `with_payload`/`with_vector` projection options are accepted but payload is
  always returned and vectors are returned as `null`.
- `offset`-based scroll pagination is accepted but ignored; `next_page_offset`
  is always `null`.
- Only the `match.value` condition is implemented (no range/geo/`any`/text).

## Smoke test

With the server running on 6333:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue

c = QdrantClient(url="http://localhost:6333")
c.create_collection("t", vectors_config=VectorParams(size=4, distance=Distance.COSINE))
c.upsert("t", points=[
    PointStruct(id="hash-not-uuid", vector=[1,0,0,0], payload={"text":"a","metadata":{"hash":"x"}}),
    PointStruct(id="other", vector=[0.9,0.1,0,0], payload={"text":"b","metadata":{"hash":"x"}}),
])
print(c.query_points("t", query=[1,0,0,0], limit=2).points)
flt = Filter(must=[FieldCondition(key="metadata.hash", match=MatchValue(value="x"))])
print(c.scroll("t", scroll_filter=flt)[0])
c.delete("t", points_selector=["other"])
```
