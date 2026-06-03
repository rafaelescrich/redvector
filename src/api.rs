//! Integrated REST and gRPC API server for RedVector
//! 
//! This module provides REST and gRPC APIs that run in the same process as the Redis server,
//! sharing the same database directly without going through the Redis protocol.

#![cfg(feature = "api-server")]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{Html, Json},
    routing::{get, post, delete},
    Router,
};
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;
use database::Database;
use parser::parse_vector_csv;

#[cfg(feature = "hnsw-backend")]
use database::vector_index::{HnswVectorIndex, VectorMetric};

// ============================================================================
// Shared State
// ============================================================================

/// Shared application state for API handlers
#[derive(Clone)]
pub struct ApiState {
    /// Direct reference to the database (shared with Redis server)
    pub db: Arc<Mutex<Database>>,
    /// Vector indexes (shared with FT.* commands)
    pub indexes: Arc<Mutex<HashMap<String, Arc<Mutex<HnswVectorIndex>>>>>,
}

impl ApiState {
    pub fn new(db: Arc<Mutex<Database>>) -> Self {
        Self {
            db,
            indexes: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

// ============================================================================
// REST API Types
// ============================================================================

#[derive(Deserialize)]
pub struct CreateCollectionRequest {
    vector_size: usize,
    #[serde(default = "default_distance")]
    distance: String,
}

fn default_distance() -> String {
    "Cosine".to_string()
}

#[derive(Deserialize)]
pub struct UpsertRequest {
    points: Vec<PointInput>,
}

#[derive(Deserialize)]
pub struct PointInput {
    id: u64,
    vector: Vec<f32>,
    #[serde(default)]
    #[allow(dead_code)]
    payload: HashMap<String, String>,
}

#[derive(Deserialize)]
pub struct SearchQuery {
    vector: String,  // Comma-separated floats
    #[serde(default = "default_limit")]
    limit: usize,
}

fn default_limit() -> usize {
    10
}

#[derive(Serialize)]
pub struct ApiResponse<T> {
    success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Serialize)]
pub struct CollectionInfo {
    name: String,
    vector_size: usize,
    points_count: usize,
    distance: String,
}

#[derive(Serialize)]
pub struct SearchResult {
    id: u64,
    score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    payload: Option<HashMap<String, String>>,
}

#[derive(Serialize)]
pub struct SearchResponse {
    results: Vec<SearchResult>,
    took_ms: u64,
}

// ============================================================================
// REST API Handlers
// ============================================================================

/// Health check endpoint
async fn health() -> Json<ApiResponse<String>> {
    Json(ApiResponse {
        success: true,
        data: Some("RedVector API is running".to_string()),
        error: None,
    })
}

/// Server info endpoint
async fn info(State(state): State<ApiState>) -> Json<ApiResponse<serde_json::Value>> {
    let db = state.db.lock().unwrap();
    let indexes = state.indexes.lock().unwrap();
    
    Json(ApiResponse {
        success: true,
        data: Some(serde_json::json!({
            "server": "RedVector",
            "version": env!("CARGO_PKG_VERSION"),
            "redis_port": db.config.port,
            "collections": indexes.len(),
            "features": {
                "redis_protocol": true,
                "rest_api": true,
                "grpc_api": true,
                "vector_search": true,
                "hnsw": cfg!(feature = "hnsw-backend"),
            }
        })),
        error: None,
    })
}

/// Create a new collection/index
async fn create_collection(
    State(state): State<ApiState>,
    Path(collection_name): Path<String>,
    Json(req): Json<CreateCollectionRequest>,
) -> Result<Json<ApiResponse<CollectionInfo>>, StatusCode> {
    let mut indexes = state.indexes.lock().unwrap();
    
    if indexes.contains_key(&collection_name) {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some(format!("Collection '{}' already exists", collection_name)),
        }));
    }
    
    // Create new HNSW index
    #[cfg(feature = "hnsw-backend")]
    {
        let metric = match req.distance.to_lowercase().as_str() {
            "euclidean" | "l2" => VectorMetric::Euclidean,
            "innerproduct" | "ip" | "dot" => VectorMetric::InnerProduct,
            _ => VectorMetric::Cosine,
        };
        
        let index = HnswVectorIndex::new(req.vector_size, metric, Some(16), Some(200));
        indexes.insert(collection_name.clone(), Arc::new(Mutex::new(index)));
        
        Ok(Json(ApiResponse {
            success: true,
            data: Some(CollectionInfo {
                name: collection_name,
                vector_size: req.vector_size,
                points_count: 0,
                distance: req.distance,
            }),
            error: None,
        }))
    }
    
    #[cfg(not(feature = "hnsw-backend"))]
    {
        Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some("HNSW backend not enabled".to_string()),
        }))
    }
}

/// Get collection info
async fn get_collection(
    State(state): State<ApiState>,
    Path(collection_name): Path<String>,
) -> Result<Json<ApiResponse<CollectionInfo>>, StatusCode> {
    let indexes = state.indexes.lock().unwrap();
    
    match indexes.get(&collection_name) {
        Some(index_arc) => {
            let index = index_arc.lock().unwrap();
            Ok(Json(ApiResponse {
                success: true,
                data: Some(CollectionInfo {
                    name: collection_name,
                    vector_size: index.dimension(),
                    points_count: index.len(),
                    distance: "Cosine".to_string(),
                }),
                error: None,
            }))
        }
        None => Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some(format!("Collection '{}' not found", collection_name)),
        })),
    }
}

/// Delete a collection
async fn delete_collection(
    State(state): State<ApiState>,
    Path(collection_name): Path<String>,
) -> Result<Json<ApiResponse<String>>, StatusCode> {
    let mut indexes = state.indexes.lock().unwrap();
    
    match indexes.remove(&collection_name) {
        Some(_) => Ok(Json(ApiResponse {
            success: true,
            data: Some(format!("Collection '{}' deleted", collection_name)),
            error: None,
        })),
        None => Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some(format!("Collection '{}' not found", collection_name)),
        })),
    }
}

/// List all collections
async fn list_collections(
    State(state): State<ApiState>,
) -> Result<Json<ApiResponse<Vec<CollectionInfo>>>, StatusCode> {
    let indexes = state.indexes.lock().unwrap();
    
    let collections: Vec<CollectionInfo> = indexes.iter().map(|(name, index_arc)| {
        let index = index_arc.lock().unwrap();
        CollectionInfo {
            name: name.clone(),
            vector_size: index.dimension(),
            points_count: index.len(),
            distance: "Cosine".to_string(),
        }
    }).collect();
    
    Ok(Json(ApiResponse {
        success: true,
        data: Some(collections),
        error: None,
    }))
}

/// Upsert points into a collection
async fn upsert_points(
    State(state): State<ApiState>,
    Path(collection_name): Path<String>,
    Json(req): Json<UpsertRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let indexes = state.indexes.lock().unwrap();
    
    match indexes.get(&collection_name) {
        Some(index_arc) => {
            let mut index = index_arc.lock().unwrap();
            let mut upserted = 0;
            let mut errors = Vec::new();
            
            for point in req.points {
                match index.add(point.id, point.vector) {
                    Ok(_) => upserted += 1,
                    Err(e) => errors.push(format!("Point {}: {}", point.id, e)),
                }
            }
            
            Ok(Json(ApiResponse {
                success: errors.is_empty(),
                data: Some(serde_json::json!({
                    "upserted": upserted,
                    "errors": errors,
                })),
                error: None,
            }))
        }
        None => Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some(format!("Collection '{}' not found", collection_name)),
        })),
    }
}

/// Search for similar vectors
async fn search_points(
    State(state): State<ApiState>,
    Path(collection_name): Path<String>,
    Query(params): Query<SearchQuery>,
) -> Result<Json<ApiResponse<SearchResponse>>, StatusCode> {
    let start = std::time::Instant::now();
    let indexes = state.indexes.lock().unwrap();
    
    match indexes.get(&collection_name) {
        Some(index_arc) => {
            // Parse query vector
            let query_vector = parse_vector_csv(&params.vector);
            
            match query_vector {
                Ok(qv) => {
                    let index = index_arc.lock().unwrap();
                    match index.search(&qv, params.limit, None) {
                        Ok(results) => {
                            let search_results: Vec<SearchResult> = results
                                .into_iter()
                                .map(|(id, score)| SearchResult {
                                    id,
                                    score,
                                    payload: None,
                                })
                                .collect();
                            
                            Ok(Json(ApiResponse {
                                success: true,
                                data: Some(SearchResponse {
                                    results: search_results,
                                    took_ms: start.elapsed().as_millis() as u64,
                                }),
                                error: None,
                            }))
                        }
                        Err(e) => Ok(Json(ApiResponse {
                            success: false,
                            data: None,
                            error: Some(format!("Search failed: {}", e)),
                        })),
                    }
                }
                Err(e) => Ok(Json(ApiResponse {
                    success: false,
                    data: None,
                    error: Some(format!("Invalid vector format: {}", e)),
                })),
            }
        }
        None => Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some(format!("Collection '{}' not found", collection_name)),
        })),
    }
}

/// Serve static HTML for API documentation
async fn api_docs() -> Html<&'static str> {
    Html(r#"
<!DOCTYPE html>
<html>
<head>
    <title>RedVector API</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               max-width: 900px; margin: 50px auto; padding: 20px; background: #0a0a0a; color: #e0e0e0; }
        h1 { color: #ff6b6b; }
        h2 { color: #4ecdc4; border-bottom: 1px solid #333; padding-bottom: 10px; }
        code { background: #1a1a2e; padding: 2px 6px; border-radius: 4px; color: #feca57; }
        pre { background: #1a1a2e; padding: 15px; border-radius: 8px; overflow-x: auto; }
        .endpoint { margin: 20px 0; padding: 15px; background: #16213e; border-radius: 8px; border-left: 4px solid #4ecdc4; }
        .method { display: inline-block; padding: 4px 10px; border-radius: 4px; font-weight: bold; margin-right: 10px; }
        .get { background: #27ae60; }
        .post { background: #3498db; }
        .delete { background: #e74c3c; }
    </style>
</head>
<body>
    <h1>🚀 RedVector API</h1>
    <p>Redis-Compatible Vector Database with REST & gRPC APIs</p>
    
    <h2>REST Endpoints</h2>
    
    <div class="endpoint">
        <span class="method get">GET</span><code>/health</code>
        <p>Health check endpoint</p>
    </div>
    
    <div class="endpoint">
        <span class="method get">GET</span><code>/api/info</code>
        <p>Server information</p>
    </div>
    
    <div class="endpoint">
        <span class="method get">GET</span><code>/api/collections</code>
        <p>List all collections</p>
    </div>
    
    <div class="endpoint">
        <span class="method post">POST</span><code>/api/collections/:name</code>
        <p>Create a new collection</p>
        <pre>{ "vector_size": 384, "distance": "Cosine" }</pre>
    </div>
    
    <div class="endpoint">
        <span class="method get">GET</span><code>/api/collections/:name</code>
        <p>Get collection info</p>
    </div>
    
    <div class="endpoint">
        <span class="method delete">DELETE</span><code>/api/collections/:name</code>
        <p>Delete a collection</p>
    </div>
    
    <div class="endpoint">
        <span class="method post">POST</span><code>/api/collections/:name/points</code>
        <p>Upsert points</p>
        <pre>{ "points": [{ "id": 1, "vector": [0.1, 0.2, ...], "payload": {} }] }</pre>
    </div>
    
    <div class="endpoint">
        <span class="method get">GET</span><code>/api/collections/:name/search?vector=0.1,0.2,...&limit=10</code>
        <p>Search for similar vectors</p>
    </div>
    
    <h2>gRPC Endpoint</h2>
    <p>gRPC service available at <code>:50051</code></p>
    <pre>
service VectorService {
    rpc CreateCollection(CreateCollectionRequest) returns (CreateCollectionResponse);
    rpc Upsert(UpsertRequest) returns (UpsertResponse);
    rpc Search(SearchRequest) returns (SearchResponse);
    rpc GetCollectionInfo(GetCollectionInfoRequest) returns (GetCollectionInfoResponse);
    rpc DeleteCollection(DeleteCollectionRequest) returns (DeleteCollectionResponse);
}</pre>
    
    <h2>Redis Protocol</h2>
    <p>Connect via Redis client on port <code>:6379</code></p>
    <pre>
FT.CREATE myindex SCHEMA vector VECTOR(384)
FT.ADD myindex doc1 1.0 FIELDS vector "0.1,0.2,0.3,..."
FT.SEARCH myindex "0.1,0.2,0.3,..."
    </pre>
</body>
</html>
    "#)
}

// ============================================================================
// Router Creation
// ============================================================================

/// Create the REST API router
pub fn create_rest_router(state: ApiState) -> Router {
    Router::new()
        .route("/", get(api_docs))
        .route("/health", get(health))
        .route("/api/info", get(info))
        .route("/api/collections", get(list_collections))
        .route("/api/collections/:name", post(create_collection))
        .route("/api/collections/:name", get(get_collection))
        .route("/api/collections/:name", delete(delete_collection))
        .route("/api/collections/:name/points", post(upsert_points))
        .route("/api/collections/:name/search", get(search_points))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

// ============================================================================
// gRPC Service
// ============================================================================

pub mod grpc {
    tonic::include_proto!("redvector");
}

use grpc::vector_service_server::{VectorService, VectorServiceServer};
use tonic::{Request, Response, Status};

/// gRPC service implementation
pub struct VectorServiceImpl {
    state: ApiState,
}

impl VectorServiceImpl {
    pub fn new(state: ApiState) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl VectorService for VectorServiceImpl {
    async fn create_collection(
        &self,
        request: Request<grpc::CreateCollectionRequest>,
    ) -> Result<Response<grpc::CreateCollectionResponse>, Status> {
        let req = request.into_inner();
        let mut indexes = self.state.indexes.lock().unwrap();
        
        if indexes.contains_key(&req.collection_name) {
            return Ok(Response::new(grpc::CreateCollectionResponse {
                success: true,
                message: format!("Collection '{}' already exists", req.collection_name),
            }));
        }
        
        #[cfg(feature = "hnsw-backend")]
        {
            let metric = match req.distance.to_lowercase().as_str() {
                "euclidean" | "l2" => VectorMetric::Euclidean,
                "innerproduct" | "ip" | "dot" => VectorMetric::InnerProduct,
                _ => VectorMetric::Cosine,
            };
            
            let index = HnswVectorIndex::new(req.vector_size as usize, metric, Some(16), Some(200));
            indexes.insert(req.collection_name.clone(), Arc::new(Mutex::new(index)));
        }
        
        Ok(Response::new(grpc::CreateCollectionResponse {
            success: true,
            message: format!("Collection '{}' created", req.collection_name),
        }))
    }

    async fn upsert(
        &self,
        request: Request<grpc::UpsertRequest>,
    ) -> Result<Response<grpc::UpsertResponse>, Status> {
        let req = request.into_inner();
        let indexes = self.state.indexes.lock().unwrap();
        
        match indexes.get(&req.collection_name) {
            Some(index_arc) => {
                let mut index = index_arc.lock().unwrap();
                let mut upserted = 0u32;
                
                for point in req.points {
                    if index.add(point.id, point.vector).is_ok() {
                        upserted += 1;
                    }
                }
                
                Ok(Response::new(grpc::UpsertResponse {
                    success: true,
                    upserted_count: upserted,
                }))
            }
            None => Err(Status::not_found(format!(
                "Collection '{}' not found",
                req.collection_name
            ))),
        }
    }

    async fn search(
        &self,
        request: Request<grpc::SearchRequest>,
    ) -> Result<Response<grpc::SearchResponse>, Status> {
        let req = request.into_inner();
        let indexes = self.state.indexes.lock().unwrap();
        
        match indexes.get(&req.collection_name) {
            Some(index_arc) => {
                let index = index_arc.lock().unwrap();
                match index.search(&req.vector, req.limit as usize, None) {
                    Ok(results) => {
                        let scored_points: Vec<grpc::ScoredPoint> = results
                            .into_iter()
                            .map(|(id, score)| grpc::ScoredPoint {
                                id,
                                score,
                                payload: HashMap::new(),
                            })
                            .collect();
                        
                        Ok(Response::new(grpc::SearchResponse {
                            results: scored_points,
                        }))
                    }
                    Err(e) => Err(Status::internal(format!("Search failed: {}", e))),
                }
            }
            None => Err(Status::not_found(format!(
                "Collection '{}' not found",
                req.collection_name
            ))),
        }
    }

    async fn get_collection_info(
        &self,
        request: Request<grpc::GetCollectionInfoRequest>,
    ) -> Result<Response<grpc::GetCollectionInfoResponse>, Status> {
        let req = request.into_inner();
        let indexes = self.state.indexes.lock().unwrap();
        
        match indexes.get(&req.collection_name) {
            Some(index_arc) => {
                let index = index_arc.lock().unwrap();
                Ok(Response::new(grpc::GetCollectionInfoResponse {
                    collection_name: req.collection_name,
                    vector_size: index.dimension() as u32,
                    points_count: index.len() as u64,
                    distance: "Cosine".to_string(),
                }))
            }
            None => Err(Status::not_found(format!(
                "Collection '{}' not found",
                req.collection_name
            ))),
        }
    }

    async fn delete_collection(
        &self,
        request: Request<grpc::DeleteCollectionRequest>,
    ) -> Result<Response<grpc::DeleteCollectionResponse>, Status> {
        let req = request.into_inner();
        let mut indexes = self.state.indexes.lock().unwrap();
        
        match indexes.remove(&req.collection_name) {
            Some(_) => Ok(Response::new(grpc::DeleteCollectionResponse { success: true })),
            None => Err(Status::not_found(format!(
                "Collection '{}' not found",
                req.collection_name
            ))),
        }
    }
}

/// Create gRPC server
pub fn create_grpc_service(state: ApiState) -> VectorServiceServer<VectorServiceImpl> {
    VectorServiceServer::new(VectorServiceImpl::new(state))
}

// ============================================================================
// Server Startup
// ============================================================================

/// Configuration for API servers
pub struct ApiConfig {
    pub rest_port: u16,
    pub grpc_port: u16,
    pub rest_enabled: bool,
    pub grpc_enabled: bool,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            rest_port: 8888,
            grpc_port: 50051,
            rest_enabled: true,
            grpc_enabled: true,
        }
    }
}

/// Start the API servers (REST and gRPC)
/// This function is called from main.rs and runs the API servers in background threads
pub async fn start_api_servers(
    db: Arc<Mutex<Database>>,
    config: ApiConfig,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let state = ApiState::new(db);
    
    // Start REST server
    if config.rest_enabled {
        let rest_state = state.clone();
        let rest_port = config.rest_port;
        tokio::spawn(async move {
            let app = create_rest_router(rest_state);
            let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", rest_port))
                .await
                .expect("Failed to bind REST server");
            println!("📊 REST API: http://localhost:{}", rest_port);
            axum::serve(listener, app).await.expect("REST server failed");
        });
    }
    
    // Start gRPC server
    if config.grpc_enabled {
        let grpc_state = state.clone();
        let grpc_port = config.grpc_port;
        tokio::spawn(async move {
            let addr = format!("0.0.0.0:{}", grpc_port).parse().unwrap();
            let svc = create_grpc_service(grpc_state);
            println!("🔌 gRPC API: http://localhost:{}", grpc_port);
            tonic::transport::Server::builder()
                .add_service(svc)
                .serve(addr)
                .await
                .expect("gRPC server failed");
        });
    }
    
    Ok(())
}

