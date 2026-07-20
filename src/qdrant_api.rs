//! Qdrant-compatible REST API for RedVector.
//!
//! Implements the subset of the Qdrant REST API that the official Python
//! `qdrant-client` v1.17.0 (as used by Open WebUI) exercises, so that it works
//! against RedVector unmodified.
//!
//! Runs on a dedicated port (default 6333, configurable via `QDRANT_COMPAT_PORT`).
//! In-memory only. Reuses `HnswVectorIndex` for ANN.
//!
//! Key difference vs. real Qdrant: point IDs may be ARBITRARY strings (non-UUID
//! hashes), not just uint/UUID. We store the original id and map it to an
//! internal u64 for the HNSW index.

#![cfg(all(feature = "api-server", feature = "hnsw-backend"))]

use std::collections::HashMap;
use std::path::{Path as FsPath, PathBuf};
use std::sync::{Arc, Mutex};

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{delete, get, post, put},
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use database::vector_index::{HnswVectorIndex, VectorMetric};

// ============================================================================
// Point id (string or integer)
// ============================================================================

/// A Qdrant point id. Real Qdrant only allows uint/UUID; we accept any string
/// or integer and preserve it verbatim.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PointId {
    Int(u64),
    Str(String),
}

impl PointId {
    /// Convert the original id into a `serde_json::Value` for responses,
    /// preserving int-ness.
    fn to_value(&self) -> Value {
        match self {
            PointId::Int(i) => json!(i),
            PointId::Str(s) => json!(s),
        }
    }
}

// ============================================================================
// Stored data model
// ============================================================================

struct StoredPoint {
    #[allow(dead_code)]
    vector: Vec<f32>,
    payload: Value,
}

/// One Qdrant collection backed by an `HnswVectorIndex`.
struct Collection {
    index: HnswVectorIndex,
    dimension: usize,
    distance: VectorMetric,
    points: HashMap<PointId, StoredPoint>,
    id_to_uid: HashMap<PointId, u64>,
    uid_to_id: HashMap<u64, PointId>,
    next_uid: u64,
}

impl Collection {
    fn new(dimension: usize, distance: VectorMetric, m: Option<usize>) -> Self {
        Self {
            index: HnswVectorIndex::new(dimension, distance, m, Some(200)),
            dimension,
            distance,
            points: HashMap::new(),
            id_to_uid: HashMap::new(),
            uid_to_id: HashMap::new(),
            next_uid: 0,
        }
    }

    fn distance_name(&self) -> &'static str {
        match self.distance {
            VectorMetric::Cosine => "Cosine",
            VectorMetric::Euclidean => "Euclid",
            VectorMetric::InnerProduct => "Dot",
        }
    }

    /// Convert the raw distance returned by `HnswVectorIndex::search` into the
    /// Qdrant-style score that qdrant-client expects.
    ///
    /// The underlying index always uses cosine distance (0 = identical,
    /// 1 = orthogonal, 2 = opposite). Qdrant returns a *similarity* for Cosine
    /// and Dot (higher = better), and the raw distance for Euclid (lower = better).
    fn score_from_distance(&self, distance: f32) -> f32 {
        match self.distance {
            // cosine similarity in [-1, 1] = 1 - cosine_distance
            VectorMetric::Cosine | VectorMetric::InnerProduct => 1.0 - distance,
            VectorMetric::Euclidean => distance,
        }
    }

    /// Insert or replace a point (upsert semantics).
    fn upsert(&mut self, id: PointId, vector: Vec<f32>, payload: Value) -> Result<(), String> {
        if vector.len() != self.dimension {
            return Err(format!(
                "Wrong input: Vector dimension error: expected dim: {}, got {}",
                self.dimension,
                vector.len()
            ));
        }

        // If the point already exists, drop its old index mapping first.
        if let Some(old_uid) = self.id_to_uid.get(&id).copied() {
            let _ = self.index.remove(old_uid);
            self.uid_to_id.remove(&old_uid);
        }

        let uid = self.next_uid;
        self.next_uid += 1;
        self.index.add(uid, vector.clone())?;
        self.id_to_uid.insert(id.clone(), uid);
        self.uid_to_id.insert(uid, id.clone());
        self.points.insert(id, StoredPoint { vector, payload });
        Ok(())
    }

    fn delete_by_id(&mut self, id: &PointId) {
        if let Some(uid) = self.id_to_uid.remove(id) {
            let _ = self.index.remove(uid);
            self.uid_to_id.remove(&uid);
        }
        self.points.remove(id);
    }

    /// Build the serializable on-disk snapshot of this collection.
    fn to_persisted(&self) -> PersistedCollection {
        let points = self
            .points
            .iter()
            .map(|(id, stored)| PersistedPoint {
                id: id.clone(),
                vector: stored.vector.clone(),
                payload: stored.payload.clone(),
            })
            .collect();
        PersistedCollection {
            dimension: self.dimension,
            distance: self.distance_name().to_string(),
            points,
        }
    }

    /// Rebuild an in-memory collection from a persisted snapshot. The HNSW
    /// graph is rebuilt from scratch by re-adding every stored vector.
    fn from_persisted(p: PersistedCollection) -> Self {
        let distance = parse_distance(&p.distance);
        let mut col = Collection::new(p.dimension, distance, None);
        for pt in p.points {
            // Reuse upsert so id/uid maps and the index stay consistent.
            // Ignore dimension mismatches defensively (shouldn't happen).
            let _ = col.upsert(pt.id, pt.vector, pt.payload);
        }
        col
    }
}

// ============================================================================
// On-disk persistence
// ============================================================================

/// Serializable snapshot of a single point.
#[derive(Serialize, Deserialize)]
struct PersistedPoint {
    id: PointId,
    vector: Vec<f32>,
    payload: Value,
}

/// Serializable snapshot of a whole collection. The HNSW graph itself is NOT
/// serialized; it is rebuilt from `vector`s on load.
#[derive(Serialize, Deserialize)]
struct PersistedCollection {
    dimension: usize,
    distance: String,
    points: Vec<PersistedPoint>,
}

/// Default data directory when `QDRANT_COMPAT_DATA_DIR` is unset.
pub const DEFAULT_QDRANT_DATA_DIR: &str = "/data/qdrant";

/// Resolve the data directory from `QDRANT_COMPAT_DATA_DIR`.
pub fn qdrant_data_dir() -> PathBuf {
    std::env::var("QDRANT_COMPAT_DATA_DIR")
        .unwrap_or_else(|_| DEFAULT_QDRANT_DATA_DIR.to_string())
        .into()
}

/// Sanitize a collection name into a safe single-segment filename.
/// Collection names from Open WebUI are safe (`open_webui_<uuid>`), but we
/// still guard against path separators and other surprises.
fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn collection_path(dir: &FsPath, name: &str) -> PathBuf {
    dir.join(format!("{}.json", sanitize_name(name)))
}

/// Persist a single collection to disk. Errors are logged but not fatal.
fn persist_collection(dir: &FsPath, name: &str, col: &Collection) {
    if let Err(e) = std::fs::create_dir_all(dir) {
        eprintln!("⚠️  Qdrant persist: cannot create {}: {}", dir.display(), e);
        return;
    }
    let path = collection_path(dir, name);
    let snapshot = col.to_persisted();
    match serde_json::to_vec(&snapshot) {
        Ok(bytes) => {
            // Write to a temp file then rename for atomicity.
            let tmp = path.with_extension("json.tmp");
            if let Err(e) = std::fs::write(&tmp, &bytes) {
                eprintln!("⚠️  Qdrant persist: write {} failed: {}", tmp.display(), e);
                return;
            }
            if let Err(e) = std::fs::rename(&tmp, &path) {
                eprintln!("⚠️  Qdrant persist: rename to {} failed: {}", path.display(), e);
            }
        }
        Err(e) => eprintln!("⚠️  Qdrant persist: serialize `{}` failed: {}", name, e),
    }
}

/// Remove a collection's file from disk (on delete-collection).
fn remove_collection_file(dir: &FsPath, name: &str) {
    let path = collection_path(dir, name);
    if path.exists() {
        if let Err(e) = std::fs::remove_file(&path) {
            eprintln!("⚠️  Qdrant persist: remove {} failed: {}", path.display(), e);
        }
    }
}

/// Scan the data dir and load all `*.json` collections into memory.
/// Safe if the dir is missing/empty (returns an empty map).
fn load_all_collections(dir: &FsPath) -> HashMap<String, Collection> {
    let mut out = HashMap::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return out, // missing dir => fresh start
    };
    for entry in entries.flatten() {
        let path = entry.path();
        // Only top-level *.json files; skip *.json.tmp and others.
        let is_json = path.extension().map(|e| e == "json").unwrap_or(false);
        if !is_json || !path.is_file() {
            continue;
        }
        let name = match path.file_stem().and_then(|s| s.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("⚠️  Qdrant load: read {} failed: {}", path.display(), e);
                continue;
            }
        };
        match serde_json::from_slice::<PersistedCollection>(&bytes) {
            Ok(snapshot) => {
                let n_points = snapshot.points.len();
                let col = Collection::from_persisted(snapshot);
                println!(
                    "🟣 Qdrant: restored collection `{}` ({} points, HNSW rebuilt)",
                    name, n_points
                );
                out.insert(name, col);
            }
            Err(e) => eprintln!("⚠️  Qdrant load: parse {} failed: {}", path.display(), e),
        }
    }
    out
}

// ============================================================================
// Shared state
// ============================================================================

#[derive(Clone)]
pub struct QdrantState {
    collections: Arc<Mutex<HashMap<String, Collection>>>,
    data_dir: Arc<PathBuf>,
}

impl QdrantState {
    pub fn new() -> Self {
        Self::with_data_dir(qdrant_data_dir())
    }

    /// Create state backed by `data_dir`, loading any collections already on
    /// disk. Safe if the dir is empty or missing.
    pub fn with_data_dir(data_dir: PathBuf) -> Self {
        if let Err(e) = std::fs::create_dir_all(&data_dir) {
            eprintln!(
                "⚠️  Qdrant persist: cannot create data dir {}: {}",
                data_dir.display(),
                e
            );
        }
        let collections = load_all_collections(&data_dir);
        Self {
            collections: Arc::new(Mutex::new(collections)),
            data_dir: Arc::new(data_dir),
        }
    }
}

impl Default for QdrantState {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Response envelope helpers
// ============================================================================

fn envelope(result: Value) -> Json<Value> {
    Json(json!({
        "result": result,
        "status": "ok",
        "time": 0.0
    }))
}

/// Build a Qdrant error response (matches the shape qdrant-client expects:
/// `{"status": {"error": "..."}, "time": 0.0}`).
fn error_response(code: StatusCode, message: impl Into<String>) -> axum::response::Response {
    let body = Json(json!({
        "status": { "error": message.into() },
        "time": 0.0
    }));
    (code, body).into_response()
}

fn not_found(name: &str) -> axum::response::Response {
    error_response(
        StatusCode::NOT_FOUND,
        format!("Collection `{}` doesn't exist!", name),
    )
}

fn update_result() -> Value {
    json!({ "operation_id": 0, "status": "completed" })
}

// ============================================================================
// Filtering
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
struct MatchValue {
    value: Value,
}

#[derive(Debug, Clone, Deserialize)]
struct FieldCondition {
    key: String,
    #[serde(default)]
    #[serde(rename = "match")]
    r#match: Option<MatchValue>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct Filter {
    #[serde(default)]
    must: Option<Vec<FieldCondition>>,
    #[serde(default)]
    should: Option<Vec<FieldCondition>>,
    #[serde(default)]
    must_not: Option<Vec<FieldCondition>>,
}

/// Resolve a dotted key like `metadata.hash` against the payload.
fn resolve_key<'a>(payload: &'a Value, key: &str) -> Option<&'a Value> {
    let mut cur = payload;
    for part in key.split('.') {
        cur = cur.get(part)?;
    }
    Some(cur)
}

/// Loose equality between a JSON value and a match value (handles numeric types).
fn values_eq(actual: &Value, expected: &Value) -> bool {
    if actual == expected {
        return true;
    }
    match (actual.as_f64(), expected.as_f64()) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}

fn condition_matches(payload: &Value, cond: &FieldCondition) -> bool {
    let actual = match resolve_key(payload, &cond.key) {
        Some(v) => v,
        None => return false,
    };
    match &cond.r#match {
        Some(m) => values_eq(actual, &m.value),
        // No match clause: treat presence of the key as a match.
        None => true,
    }
}

fn passes_filter(payload: &Value, filter: &Filter) -> bool {
    if let Some(must) = &filter.must {
        if !must.iter().all(|c| condition_matches(payload, c)) {
            return false;
        }
    }
    if let Some(must_not) = &filter.must_not {
        if must_not.iter().any(|c| condition_matches(payload, c)) {
            return false;
        }
    }
    if let Some(should) = &filter.should {
        if !should.is_empty() && !should.iter().any(|c| condition_matches(payload, c)) {
            return false;
        }
    }
    true
}

// ============================================================================
// Distance parsing
// ============================================================================

fn parse_distance(s: &str) -> VectorMetric {
    match s.to_lowercase().as_str() {
        "euclid" | "euclidean" | "l2" => VectorMetric::Euclidean,
        "dot" | "innerproduct" | "ip" => VectorMetric::InnerProduct,
        _ => VectorMetric::Cosine,
    }
}

// ============================================================================
// Root / health
// ============================================================================

async fn root() -> Json<Value> {
    Json(json!({
        "title": "qdrant - vector search engine",
        "version": "1.17.0",
        "commit": "redvector"
    }))
}

async fn healthz() -> impl IntoResponse {
    (StatusCode::OK, "healthz check passed")
}

// ============================================================================
// Collections
// ============================================================================

async fn list_collections(State(state): State<QdrantState>) -> Json<Value> {
    let cols = state.collections.lock().unwrap();
    let names: Vec<Value> = cols.keys().map(|n| json!({ "name": n })).collect();
    envelope(json!({ "collections": names }))
}

async fn collection_exists(
    State(state): State<QdrantState>,
    Path(name): Path<String>,
) -> Json<Value> {
    let cols = state.collections.lock().unwrap();
    envelope(json!({ "exists": cols.contains_key(&name) }))
}

async fn get_collection(
    State(state): State<QdrantState>,
    Path(name): Path<String>,
) -> axum::response::Response {
    let cols = state.collections.lock().unwrap();
    match cols.get(&name) {
        Some(c) => {
            let count = c.points.len();
            envelope(json!({
                "status": "green",
                "optimizer_status": "ok",
                "vectors_count": count,
                "indexed_vectors_count": count,
                "points_count": count,
                "segments_count": 1,
                "config": {
                    "params": {
                        "vectors": {
                            "size": c.dimension,
                            "distance": c.distance_name()
                        },
                        "shard_number": 1,
                        "replication_factor": 1,
                        "write_consistency_factor": 1,
                        "on_disk_payload": false
                    },
                    "hnsw_config": {
                        "m": 16,
                        "ef_construct": 200,
                        "full_scan_threshold": 10000,
                        "max_indexing_threads": 0,
                        "on_disk": false
                    },
                    "optimizer_config": {
                        "deleted_threshold": 0.2,
                        "vacuum_min_vector_number": 1000,
                        "default_segment_number": 0,
                        "max_segment_size": null,
                        "memmap_threshold": null,
                        "indexing_threshold": 20000,
                        "flush_interval_sec": 5,
                        "max_optimization_threads": null
                    },
                    "wal_config": {
                        "wal_capacity_mb": 32,
                        "wal_segments_ahead": 0
                    }
                },
                "payload_schema": {}
            }))
            .into_response()
        }
        None => not_found(&name),
    }
}

#[derive(Deserialize)]
struct VectorsConfig {
    size: usize,
    #[serde(default = "default_distance_str")]
    distance: String,
    #[serde(default)]
    #[allow(dead_code)]
    on_disk: Option<bool>,
}

fn default_distance_str() -> String {
    "Cosine".to_string()
}

#[derive(Deserialize)]
struct HnswConfigReq {
    #[serde(default)]
    m: Option<usize>,
}

#[derive(Deserialize)]
struct CreateCollectionRequest {
    vectors: VectorsConfig,
    #[serde(default)]
    hnsw_config: Option<HnswConfigReq>,
}

async fn create_collection(
    State(state): State<QdrantState>,
    Path(name): Path<String>,
    Json(req): Json<CreateCollectionRequest>,
) -> Json<Value> {
    let distance = parse_distance(&req.vectors.distance);
    let m = req.hnsw_config.as_ref().and_then(|h| h.m);
    let mut cols = state.collections.lock().unwrap();
    let col = Collection::new(req.vectors.size, distance, m);
    persist_collection(&state.data_dir, &name, &col);
    cols.insert(name, col);
    envelope(json!(true))
}

async fn delete_collection(
    State(state): State<QdrantState>,
    Path(name): Path<String>,
) -> Json<Value> {
    let mut cols = state.collections.lock().unwrap();
    cols.remove(&name);
    remove_collection_file(&state.data_dir, &name);
    envelope(json!(true))
}

// Payload index create — NO-OP since we scan in-memory.
async fn create_field_index(
    State(_state): State<QdrantState>,
    Path(_name): Path<String>,
    Query(_q): Query<HashMap<String, String>>,
    _body: Option<Json<Value>>,
) -> Json<Value> {
    envelope(update_result())
}

// ============================================================================
// Points: upsert
// ============================================================================

#[derive(Deserialize)]
struct PointStruct {
    id: PointId,
    vector: Vec<f32>,
    #[serde(default)]
    payload: Option<Value>,
}

#[derive(Deserialize)]
struct UpsertRequest {
    points: Vec<PointStruct>,
}

async fn upsert_points(
    State(state): State<QdrantState>,
    Path(name): Path<String>,
    Query(_q): Query<HashMap<String, String>>,
    Json(req): Json<UpsertRequest>,
) -> axum::response::Response {
    let mut cols = state.collections.lock().unwrap();
    let col = match cols.get_mut(&name) {
        Some(c) => c,
        None => return not_found(&name),
    };
    for p in req.points {
        let payload = p.payload.unwrap_or_else(|| json!({}));
        if let Err(e) = col.upsert(p.id, p.vector, payload) {
            return error_response(StatusCode::BAD_REQUEST, e);
        }
    }
    persist_collection(&state.data_dir, &name, col);
    envelope(update_result()).into_response()
}

// ============================================================================
// Points: query (KNN)
// ============================================================================

/// qdrant-client sends the query as either a bare vector `[..]` or wrapped in a
/// nearest-neighbour object `{"nearest": [..]}`. Accept both.
#[derive(Deserialize)]
#[serde(untagged)]
enum QueryInput {
    Vector(Vec<f32>),
    Nearest { nearest: Vec<f32> },
}

impl QueryInput {
    fn into_vec(self) -> Vec<f32> {
        match self {
            QueryInput::Vector(v) => v,
            QueryInput::Nearest { nearest } => nearest,
        }
    }
}

#[derive(Deserialize)]
struct QueryRequest {
    #[serde(default)]
    query: Option<QueryInput>,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    filter: Option<Filter>,
    #[serde(default)]
    #[allow(dead_code)]
    with_payload: Option<Value>,
}

fn default_limit() -> usize {
    10
}

async fn query_points(
    State(state): State<QdrantState>,
    Path(name): Path<String>,
    Json(req): Json<QueryRequest>,
) -> axum::response::Response {
    let cols = state.collections.lock().unwrap();
    let col = match cols.get(&name) {
        Some(c) => c,
        None => return not_found(&name),
    };

    let query_vec = match req.query {
        Some(v) => v.into_vec(),
        None => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "query vector is required".to_string(),
            )
        }
    };

    let filter = req.filter.unwrap_or_default();
    let has_filter = filter.must.is_some() || filter.should.is_some() || filter.must_not.is_some();

    // Over-fetch when filtering so we still have `limit` results after filtering.
    let fetch_k = if has_filter {
        (req.limit * 4).max(req.limit + 16)
    } else {
        req.limit
    };

    let raw = match col.index.search(&query_vec, fetch_k.max(1), None) {
        Ok(r) => r,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, e),
    };

    let mut points = Vec::new();
    for (uid, distance) in raw {
        let id = match col.uid_to_id.get(&uid) {
            Some(id) => id,
            None => continue, // deleted
        };
        let stored = match col.points.get(id) {
            Some(s) => s,
            None => continue,
        };
        if has_filter && !passes_filter(&stored.payload, &filter) {
            continue;
        }
        points.push(json!({
            "id": id.to_value(),
            "version": 0,
            "score": col.score_from_distance(distance),
            "payload": stored.payload,
            "vector": null
        }));
        if points.len() >= req.limit {
            break;
        }
    }

    envelope(json!({ "points": points })).into_response()
}

// ============================================================================
// Points: scroll
// ============================================================================

#[derive(Deserialize)]
struct ScrollRequest {
    #[serde(default)]
    filter: Option<Filter>,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    #[allow(dead_code)]
    offset: Option<Value>,
    #[serde(default)]
    #[allow(dead_code)]
    with_payload: Option<Value>,
}

async fn scroll_points(
    State(state): State<QdrantState>,
    Path(name): Path<String>,
    Json(req): Json<ScrollRequest>,
) -> axum::response::Response {
    let cols = state.collections.lock().unwrap();
    let col = match cols.get(&name) {
        Some(c) => c,
        None => return not_found(&name),
    };

    let filter = req.filter.unwrap_or_default();
    let has_filter = filter.must.is_some() || filter.should.is_some() || filter.must_not.is_some();

    let mut points = Vec::new();
    for (id, stored) in col.points.iter() {
        if has_filter && !passes_filter(&stored.payload, &filter) {
            continue;
        }
        points.push(json!({
            "id": id.to_value(),
            "payload": stored.payload,
            "vector": null
        }));
        if points.len() >= req.limit {
            break;
        }
    }

    envelope(json!({ "points": points, "next_page_offset": null })).into_response()
}

// ============================================================================
// Points: delete
// ============================================================================

#[derive(Deserialize)]
#[serde(untagged)]
enum DeleteRequest {
    ByIds { points: Vec<PointId> },
    ByFilter { filter: Filter },
}

async fn delete_points(
    State(state): State<QdrantState>,
    Path(name): Path<String>,
    Query(_q): Query<HashMap<String, String>>,
    Json(req): Json<DeleteRequest>,
) -> axum::response::Response {
    let mut cols = state.collections.lock().unwrap();
    let col = match cols.get_mut(&name) {
        Some(c) => c,
        None => return not_found(&name),
    };

    match req {
        DeleteRequest::ByIds { points } => {
            for id in points {
                col.delete_by_id(&id);
            }
        }
        DeleteRequest::ByFilter { filter } => {
            let to_delete: Vec<PointId> = col
                .points
                .iter()
                .filter(|(_, s)| passes_filter(&s.payload, &filter))
                .map(|(id, _)| id.clone())
                .collect();
            for id in to_delete {
                col.delete_by_id(&id);
            }
        }
    }

    persist_collection(&state.data_dir, &name, col);
    envelope(update_result()).into_response()
}

// ============================================================================
// Router + server
// ============================================================================

pub fn create_qdrant_router(state: QdrantState) -> Router {
    use tower_http::cors::CorsLayer;
    Router::new()
        .route("/", get(root))
        .route("/healthz", get(healthz))
        .route("/readyz", get(healthz))
        .route("/livez", get(healthz))
        .route("/collections", get(list_collections))
        .route("/collections/:name", get(get_collection))
        .route("/collections/:name", put(create_collection))
        .route("/collections/:name", delete(delete_collection))
        .route("/collections/:name/exists", get(collection_exists))
        .route("/collections/:name/index", put(create_field_index))
        .route("/collections/:name/points", put(upsert_points))
        .route("/collections/:name/points/query", post(query_points))
        .route("/collections/:name/points/scroll", post(scroll_points))
        .route("/collections/:name/points/delete", post(delete_points))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Default Qdrant REST port.
pub const DEFAULT_QDRANT_PORT: u16 = 6333;

/// Resolve the port from `QDRANT_COMPAT_PORT`, falling back to 6333.
pub fn qdrant_port() -> u16 {
    std::env::var("QDRANT_COMPAT_PORT")
        .ok()
        .and_then(|v| v.parse::<u16>().ok())
        .unwrap_or(DEFAULT_QDRANT_PORT)
}

/// Spawn the Qdrant-compatible server in the background.
pub async fn start_qdrant_server() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let data_dir = qdrant_data_dir();
    println!("🟣 Qdrant data dir: {}", data_dir.display());
    let state = QdrantState::with_data_dir(data_dir);
    let port = qdrant_port();
    tokio::spawn(async move {
        let app = create_qdrant_router(state);
        let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port))
            .await
            .expect("Failed to bind Qdrant-compat server");
        println!("🟣 Qdrant API: http://localhost:{}", port);
        axum::serve(listener, app)
            .await
            .expect("Qdrant-compat server failed");
    });
    Ok(())
}
