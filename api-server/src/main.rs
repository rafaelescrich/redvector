use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{Html, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

mod grpc;
mod sql;

use grpc::VectorServiceImpl;
use sql::SqlExecutor;

#[derive(Clone)]
struct AppState {
    redis_client: Arc<redis::Client>,
    sql_executor: Arc<SqlExecutor>,
}

#[derive(Deserialize)]
struct SearchRequest {
    query: String,
    limit: Option<usize>,
}

#[derive(Deserialize)]
struct AddDocumentRequest {
    id: String,
    text: String,
    metadata: Option<HashMap<String, String>>,
}

#[derive(Serialize)]
struct SearchResult {
    id: String,
    text: String,
    score: f32,
    metadata: Option<HashMap<String, String>>,
}

#[derive(Serialize)]
struct SearchResponse {
    results: Vec<SearchResult>,
    query: String,
}

#[derive(Serialize)]
struct ApiResponse<T> {
    success: bool,
    data: Option<T>,
    error: Option<String>,
}

/// Generate embeddings for text (simplified - in production, use a real embedding model)
fn generate_embedding(text: &str) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    text.to_lowercase().hash(&mut hasher);
    let hash = hasher.finish();
    
    // Generate 384-dimensional vector (common for sentence embeddings)
    let mut embedding = vec![0.0f32; 384];
    let mut seed = hash;
    
    for i in 0..384 {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        embedding[i] = ((seed % 2000) as f32 / 1000.0) - 1.0;
    }
    
    // Normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in &mut embedding {
            *val /= norm;
        }
    }
    
    embedding
}

/// Create index for semantic search
async fn create_index(
    State(state): State<AppState>,
    Path(index_name): Path<String>,
) -> Result<Json<ApiResponse<String>>, StatusCode> {
    tracing::info!("Creating index: {}", index_name);
    
    let mut conn = state.redis_client.get_connection()
        .map_err(|e| {
            tracing::error!("Failed to get Redis connection: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    
    // Check if index already exists - don't drop it if it has documents
    tracing::debug!("Checking if index exists: {}", index_name);
    let index_exists = redis::cmd("FT.INFO")
        .arg(&index_name)
        .query::<redis::Value>(&mut conn)
        .is_ok();
    
    if index_exists {
        tracing::warn!("Index '{}' already exists! Not recreating to preserve documents.", index_name);
        return Ok(Json(ApiResponse {
            success: true,
            data: Some(format!("Index '{}' already exists (preserving documents)", index_name)),
            error: None,
        }));
    }
    
    // Drop existing index if it exists (shouldn't happen now, but keep for safety)
    tracing::debug!("Dropping existing index if it exists: {}", index_name);
    let _ = redis::cmd("FT.DROP")
        .arg(&index_name)
        .arg("DD")
        .query::<String>(&mut conn);
    
    // Create index with vector field (must use "vector" not "embedding" for rsedis)
    tracing::info!("Creating index with schema: text TEXT, vector VECTOR(384)");
    match redis::cmd("FT.CREATE")
        .arg(&index_name)
        .arg("SCHEMA")
        .arg("text")
        .arg("TEXT")
        .arg("vector")
        .arg("VECTOR(384)")
        .query::<String>(&mut conn)
    {
        Ok(msg) => {
            tracing::info!("Index '{}' created successfully: {}", index_name, msg);
            Ok(Json(ApiResponse {
                success: true,
                data: Some(format!("Index '{}' created successfully", index_name)),
                error: None,
            }))
        }
        Err(e) => {
            tracing::error!("Failed to create index '{}': {}", index_name, e);
            Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some(format!("Failed to create index: {}", e)),
            }))
        }
    }
}

/// Add document to index
async fn add_document(
    State(state): State<AppState>,
    Path(index_name): Path<String>,
    Json(payload): Json<AddDocumentRequest>,
) -> Result<Json<ApiResponse<String>>, StatusCode> {
    tracing::info!("Adding document to index '{}': id={}, text_len={}", 
        index_name, payload.id, payload.text.len());
    
    let mut conn = state.redis_client.get_connection()
        .map_err(|e| {
            tracing::error!("Failed to get Redis connection: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    
    tracing::debug!("Generating embedding for text: {}", &payload.text[..payload.text.len().min(50)]);
    let embedding = generate_embedding(&payload.text);
    tracing::debug!("Generated embedding: {} dimensions, norm={:.4}", 
        embedding.len(), 
        embedding.iter().map(|x| x * x).sum::<f32>().sqrt());
    
    let embedding_str = embedding.iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",");
    
    tracing::debug!("Building FT.ADD command for index: {}", index_name);
    let mut cmd = redis::cmd("FT.ADD");
    cmd.arg(&index_name)
        .arg(&payload.id)
        .arg("1.0")
        .arg("FIELDS")
        .arg("text")
        .arg(&payload.text)
        .arg("vector")
        .arg(&embedding_str);
    
    if let Some(metadata) = &payload.metadata {
        tracing::debug!("Adding {} metadata fields", metadata.len());
        for (key, value) in metadata {
            cmd.arg(key).arg(value);
        }
    }
    
    // Calculate the hash that rsedis will use (same algorithm as ft_commands.rs)
    let doc_hash: u64 = payload.id.as_bytes().iter()
        .fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64));
    
    // Store document metadata in Redis for later retrieval
    let doc_key = format!("ft:doc:{}:{}", index_name, doc_hash);
    let mut doc_data: HashMap<String, String> = HashMap::new();
    doc_data.insert("id".to_string(), payload.id.clone());
    doc_data.insert("text".to_string(), payload.text.clone());
    if let Some(metadata) = &payload.metadata {
        for (k, v) in metadata {
            doc_data.insert(k.clone(), v.clone());
        }
    }
    
    // Store as JSON in Redis
    let doc_json = serde_json::to_string(&doc_data).unwrap_or_default();
    tracing::debug!("Storing document metadata at key: {}", doc_key);
    let _: Result<(), _> = redis::cmd("SET")
        .arg(&doc_key)
        .arg(&doc_json)
        .query(&mut conn);
    
    tracing::info!("Executing FT.ADD command");
    match cmd.query::<String>(&mut conn) {
        Ok(msg) => {
            tracing::info!("Document '{}' (hash: {}) added successfully to index '{}': {}", 
                payload.id, doc_hash, index_name, msg);
            Ok(Json(ApiResponse {
                success: true,
                data: Some(format!("Document '{}' added successfully", payload.id)),
                error: None,
            }))
        }
        Err(e) => {
            tracing::error!("Failed to add document '{}' to index '{}': {}", 
                payload.id, index_name, e);
            Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some(format!("Failed to add document: {}", e)),
            }))
        }
    }
}

/// Search for similar documents
async fn search(
    State(state): State<AppState>,
    Path(index_name): Path<String>,
    Query(params): Query<SearchRequest>,
) -> Result<Json<ApiResponse<SearchResponse>>, StatusCode> {
    tracing::info!("Search request: index='{}', query='{}', limit={:?}", 
        index_name, params.query, params.limit);
    
    let mut conn = state.redis_client.get_connection()
        .map_err(|e| {
            tracing::error!("Failed to get Redis connection: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    
    tracing::debug!("Generating embedding for query: {}", params.query);
    let query_embedding = generate_embedding(&params.query);
    tracing::debug!("Generated query embedding: {} dimensions, norm={:.4}", 
        query_embedding.len(),
        query_embedding.iter().map(|x| x * x).sum::<f32>().sqrt());
    
    let query_str = query_embedding.iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",");
    
    let limit = params.limit.unwrap_or(10);
    tracing::info!("Executing FT.SEARCH: index={}, limit={}", index_name, limit);
    
    // Search using vector similarity
    let result: Result<redis::Value, _> = redis::cmd("FT.SEARCH")
        .arg(&index_name)
        .arg(&query_str)
        .arg("LIMIT")
        .arg("0")
        .arg(limit.to_string())
        .query(&mut conn);
    
    match result {
        Ok(redis::Value::Bulk(results)) => {
            tracing::info!("FT.SEARCH returned bulk response with {} items", results.len());
            tracing::debug!("Raw Redis response: {:?}", results);
            
            let count = if let Some(redis::Value::Int(c)) = results.get(0) {
                tracing::info!("Search found {} total results", c);
                *c as usize
            } else {
                tracing::warn!("Could not parse result count from first element: {:?}", results.get(0));
                0
            };
            
            if count == 0 {
                tracing::warn!("No results found. Checking if documents exist in index...");
                // Try to get document count
                let _ = redis::cmd("FT.INFO")
                    .arg(&index_name)
                    .query::<redis::Value>(&mut conn)
                    .map(|info| {
                        tracing::debug!("Index info: {:?}", info);
                    });
            }
            
            let mut search_results = Vec::new();
            
            // Parse results - rsedis returns: [count, [doc_id, "score", score], [doc_id, "score", score], ...]
            tracing::debug!("Parsing {} result items (starting from index 1)", results.len() - 1);
            for i in 1..results.len() {
                tracing::debug!("Processing result at index {}: {:?}", i, results.get(i));
                
                if let Some(redis::Value::Bulk(doc_data)) = results.get(i) {
                    tracing::debug!("Found document array with {} elements", doc_data.len());
                    
                    if doc_data.len() >= 3 {
                        // Format: [doc_id, "score", score_value]
                        let id = match &doc_data[0] {
                            redis::Value::Data(bytes) => String::from_utf8_lossy(bytes).to_string(),
                            redis::Value::Int(n) => n.to_string(),
                            other => {
                                tracing::warn!("Unexpected doc_id format: {:?}", other);
                                continue;
                            }
                        };
                        
                        let score = match &doc_data[2] {
                            redis::Value::Data(bytes) => {
                                String::from_utf8_lossy(bytes).parse().unwrap_or(0.0)
                            }
                            redis::Value::Int(n) => *n as f32,
                            redis::Value::Bulk(_) => 0.0,
                            other => {
                                tracing::warn!("Unexpected score format: {:?}", other);
                                0.0
                            }
                        };
                        
                        tracing::debug!("Found document: id={}, score={:.4}", id, score);
                        
                        // Fetch document metadata from Redis using the hash ID
                        let doc_key = format!("ft:doc:{}:{}", index_name, id);
                        tracing::debug!("Fetching document from key: {}", doc_key);
                        
                        let (text, original_id) = match redis::cmd("GET")
                            .arg(&doc_key)
                            .query::<Option<String>>(&mut conn)
                        {
                            Ok(Some(doc_json)) => {
                                tracing::debug!("Retrieved document JSON: {} chars", doc_json.len());
                                match serde_json::from_str::<HashMap<String, String>>(&doc_json) {
                                    Ok(doc_data) => {
                                        let text = doc_data.get("text")
                                            .cloned()
                                            .unwrap_or_else(|| String::new());
                                        let original_id = doc_data.get("id")
                                            .cloned()
                                            .unwrap_or_else(|| id.clone());
                                        tracing::debug!("Extracted text: {} chars, original_id: {}", text.len(), original_id);
                                        (text, original_id)
                                    }
                                    Err(e) => {
                                        tracing::warn!("Failed to parse document JSON for {}: {}", id, e);
                                        (String::new(), id.clone())
                                    }
                                }
                            }
                            Ok(None) => {
                                tracing::warn!("Document not found in Redis for hash {}", id);
                                (String::new(), id.clone())
                            }
                            Err(e) => {
                                tracing::warn!("Failed to fetch document for {}: {}", id, e);
                                (String::new(), id.clone())
                            }
                        };
                        
                        tracing::info!("Parsed result: id={}, original_id={}, text_len={}, score={:.4}", 
                            id, original_id, text.len(), score);
                        
                        search_results.push(SearchResult {
                            id: original_id,
                            text,
                            score,
                            metadata: None,
                        });
                    } else {
                        tracing::warn!("Document array has unexpected length: {}", doc_data.len());
                    }
                } else {
                    tracing::warn!("Unexpected result format at index {}: not Bulk", i);
                }
            }
            
            tracing::info!("Successfully parsed {} search results", search_results.len());
            
            Ok(Json(ApiResponse {
                success: true,
                data: Some(SearchResponse {
                    results: search_results,
                    query: params.query,
                }),
                error: None,
            }))
        }
        Ok(other) => {
            tracing::error!("Unexpected response format from FT.SEARCH: {:?}", other);
            Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some(format!("Unexpected response format: {:?}", other)),
            }))
        }
        Err(e) => {
            tracing::error!("FT.SEARCH failed: {}", e);
            Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some(format!("Search failed: {}", e)),
            }))
        }
    }
}

/// Health check endpoint
async fn health() -> Json<ApiResponse<String>> {
    Json(ApiResponse {
        success: true,
        data: Some("API is running".to_string()),
        error: None,
    })
}

/// Serve HTML frontend
async fn index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

/// SQL query endpoint
/// POST /api/sql
/// Body: { "query": "SELECT * FROM collection WHERE vector = '[0.1, 0.2, ...]' LIMIT 10" }
#[derive(Deserialize)]
struct SqlRequest {
    query: String,
}

async fn execute_sql(
    State(state): State<AppState>,
    Json(req): Json<SqlRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    match state.sql_executor.execute(&req.query) {
        Ok(result) => {
            let json_result = serde_json::json!({
                "columns": result.columns,
                "rows": result.rows,
            });
            Ok(Json(ApiResponse {
                success: true,
                data: Some(json_result),
                error: None,
            }))
        }
        Err(e) => {
            Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some(e.to_string()),
            }))
        }
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();
    
    tracing::info!("Starting RedVector API Server...");
    
    // Connect to RedVector (rsedis)
    let redis_client = Arc::new(redis::Client::open("redis://127.0.0.1:6379/")
        .expect("Failed to connect to RedVector"));
    
    let sql_executor = Arc::new(SqlExecutor::new(Arc::clone(&redis_client)));
    
    let app_state = AppState {
        redis_client: Arc::clone(&redis_client),
        sql_executor: Arc::clone(&sql_executor),
    };
    
    // REST API routes
    let app = Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        .route("/api/index/:index_name", post(create_index))
        .route("/api/index/:index_name/document", post(add_document))
        .route("/api/index/:index_name/search", get(search))
        .route("/api/sql", post(execute_sql))
        .nest_service("/static", ServeDir::new("static"))
        .layer(CorsLayer::permissive())
        .with_state(app_state);
    
    // Start REST server
    let rest_listener = tokio::net::TcpListener::bind("0.0.0.0:8888")
        .await
        .expect("Failed to bind to port 8888");
    
    // Start gRPC server
    let grpc_service = VectorServiceImpl::new(redis_client.clone());
    let grpc_addr = "0.0.0.0:50051".parse().expect("Invalid gRPC address");
    
    println!("🚀 RedVector API Server starting...");
    println!("📊 REST API: http://localhost:8888");
    println!("🔌 gRPC API: http://localhost:50051");
    println!("🔍 REST endpoints:");
    println!("   GET  /health");
    println!("   POST /api/index/:index_name");
    println!("   POST /api/index/:index_name/document");
    println!("   GET  /api/index/:index_name/search?query=...&limit=10");
    println!("   POST /api/sql");
    println!("🔍 gRPC endpoints:");
    println!("   CreateCollection, Upsert, Search, GetCollectionInfo, DeleteCollection");
    
    // Run both servers concurrently
    tokio::select! {
        rest_result = axum::serve(rest_listener, app) => {
            rest_result.expect("REST server failed");
        }
        grpc_result = {
            let svc = grpc::vector_service::vector_service_server::VectorServiceServer::new(grpc_service);
            
            // Enable reflection for grpcurl
            // The file descriptor is generated during build in OUT_DIR
            let descriptor = include_bytes!(concat!(env!("OUT_DIR"), "/proto/vector_descriptor.bin"));
            let reflection_service = tonic_reflection::server::Builder::configure()
                .register_encoded_file_descriptor_set(descriptor)
                .build()
                .unwrap();
            
            tonic::transport::Server::builder()
                .add_service(reflection_service)
                .add_service(svc)
                .serve(grpc_addr)
        } => {
            grpc_result.expect("gRPC server failed");
        }
    }
}

