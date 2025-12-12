//! gRPC server implementation for RedVector
//! Similar to Qdrant's gRPC API

use redis::Commands;
use std::sync::Arc;
use tonic::{Request, Response, Status};

pub mod vector_service {
    tonic::include_proto!("redvector");
}

use vector_service::vector_service_server::VectorService;
use vector_service::*;

pub struct VectorServiceImpl {
    redis_client: Arc<redis::Client>,
}

impl VectorServiceImpl {
    pub fn new(redis_client: Arc<redis::Client>) -> Self {
        Self { redis_client }
    }
}

#[tonic::async_trait]
impl VectorService for VectorServiceImpl {
    async fn create_collection(
        &self,
        request: Request<CreateCollectionRequest>,
    ) -> Result<Response<CreateCollectionResponse>, Status> {
        let req = request.into_inner();
        let collection_name = req.collection_name;
        let vector_size = req.vector_size;
        let _distance = req.distance.as_str();

        let mut conn = self.redis_client.get_connection()
            .map_err(|e| Status::internal(format!("Redis connection error: {}", e)))?;

        // Create index using FT.CREATE
        let schema = format!("SCHEMA vector_field VECTOR({})", vector_size);
        let result: Result<String, redis::RedisError> = redis::cmd("FT.CREATE")
            .arg(&collection_name)
            .arg(&schema)
            .query(&mut conn);

        match result {
            Ok(_) => Ok(Response::new(CreateCollectionResponse {
                success: true,
                message: format!("Collection '{}' created successfully", collection_name),
            })),
            Err(e) => {
                if e.to_string().contains("already exists") {
                    Ok(Response::new(CreateCollectionResponse {
                        success: true,
                        message: format!("Collection '{}' already exists", collection_name),
                    }))
                } else {
                    Err(Status::internal(format!("Failed to create collection: {}", e)))
                }
            }
        }
    }

    async fn upsert(
        &self,
        request: Request<UpsertRequest>,
    ) -> Result<Response<UpsertResponse>, Status> {
        let req = request.into_inner();
        let collection_name = req.collection_name;
        let points = req.points;

        let mut conn = self.redis_client.get_connection()
            .map_err(|e| Status::internal(format!("Redis connection error: {}", e)))?;

        let mut upserted = 0;

        for point in points {
            let doc_id = format!("{}", point.id);
            let vector_str = point.vector.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",");

            // Build FIELDS argument
            let mut args = vec![
                "FT.ADD".to_string(),
                collection_name.clone(),
                doc_id,
                "1.0".to_string(),
                "FIELDS".to_string(),
                "vector_field".to_string(),
                vector_str,
            ];

            // Add payload as metadata
            for (key, value) in point.payload {
                args.push(key);
                args.push(value);
            }

            let result: Result<String, redis::RedisError> = redis::cmd("FT.ADD")
                .arg(&args[1..])
                .query(&mut conn);

            if result.is_ok() {
                upserted += 1;
            }
        }

        Ok(Response::new(UpsertResponse {
            success: true,
            upserted_count: upserted,
        }))
    }

    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let req = request.into_inner();
        let collection_name = req.collection_name;
        let query_vector = req.query_vector;
        let top = req.top.max(1).min(100); // Limit to 1-100

        let mut conn = self.redis_client.get_connection()
            .map_err(|e| Status::internal(format!("Redis connection error: {}", e)))?;

        // Convert vector to comma-separated string
        let query_str = query_vector.iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",");

        // Search using FT.SEARCH
        let result: Result<redis::Value, redis::RedisError> = redis::cmd("FT.SEARCH")
            .arg(&collection_name)
            .arg(&query_str)
            .arg("LIMIT")
            .arg("0")
            .arg(top.to_string())
            .query(&mut conn);

        match result {
            Ok(redis::Value::Bulk(mut arr)) => {
                if arr.is_empty() {
                    return Ok(Response::new(SearchResponse { result: vec![] }));
                }

                // First element is count
                let _count = arr.remove(0);
                let mut scored_points = Vec::new();

                // Process results: [doc_id, "score", score_value, ...]
                let mut i = 0;
                while i < arr.len() {
                    if let (Some(redis::Value::Data(doc_id_bytes)), Some(redis::Value::Data(score_bytes))) = 
                        (arr.get(i), arr.get(i + 2)) {
                        if let (Ok(doc_id_str), Ok(score_str)) = 
                            (String::from_utf8(doc_id_bytes.clone()), String::from_utf8(score_bytes.clone())) {
                            if let Ok(doc_id) = doc_id_str.parse::<u64>() {
                                if let Ok(score) = score_str.parse::<f32>() {
                                    scored_points.push(ScoredPoint {
                                        id: doc_id,
                                        score,
                                        payload: std::collections::HashMap::new(),
                                    });
                                }
                            }
                        }
                    }
                    i += 3; // Skip to next result
                }

                Ok(Response::new(SearchResponse {
                    result: scored_points,
                }))
            }
            Ok(_) => Ok(Response::new(SearchResponse { result: vec![] })),
            Err(e) => Err(Status::internal(format!("Search failed: {}", e))),
        }
    }

    async fn get_collection_info(
        &self,
        request: Request<GetCollectionInfoRequest>,
    ) -> Result<Response<GetCollectionInfoResponse>, Status> {
        let req = request.into_inner();
        let collection_name = req.collection_name;

        let mut conn = self.redis_client.get_connection()
            .map_err(|e| Status::internal(format!("Redis connection error: {}", e)))?;

        // Get info using FT.INFO
        let result: Result<redis::Value, redis::RedisError> = redis::cmd("FT.INFO")
            .arg(&collection_name)
            .query(&mut conn);

        match result {
            Ok(redis::Value::Bulk(info)) => {
                let mut vector_size = 0;
                let mut points_count = 0;
                let mut distance = "Cosine".to_string();

                // Parse info array
                let mut i = 0;
                while i < info.len() - 1 {
                    if let (Some(redis::Value::Data(key)), Some(redis::Value::Data(value))) = 
                        (info.get(i), info.get(i + 1)) {
                        if let (Ok(key_str), Ok(value_str)) = 
                            (String::from_utf8(key.clone()), String::from_utf8(value.clone())) {
                            match key_str.as_str() {
                                "num_docs" => {
                                    points_count = value_str.parse().unwrap_or(0);
                                }
                                _ => {}
                            }
                        }
                    }
                    i += 2;
                }

                Ok(Response::new(GetCollectionInfoResponse {
                    collection_name,
                    vector_size,
                    points_count,
                    distance,
                }))
            }
            Err(e) => Err(Status::not_found(format!("Collection not found: {}", e))),
            _ => Err(Status::internal("Invalid response format")),
        }
    }

    async fn delete_collection(
        &self,
        request: Request<DeleteCollectionRequest>,
    ) -> Result<Response<DeleteCollectionResponse>, Status> {
        let req = request.into_inner();
        let collection_name = req.collection_name;

        let mut conn = self.redis_client.get_connection()
            .map_err(|e| Status::internal(format!("Redis connection error: {}", e)))?;

        let result: Result<String, redis::RedisError> = redis::cmd("FT.DROP")
            .arg(&collection_name)
            .query(&mut conn);

        match result {
            Ok(_) => Ok(Response::new(DeleteCollectionResponse { success: true })),
            Err(e) => Err(Status::internal(format!("Failed to delete collection: {}", e))),
        }
    }
}

