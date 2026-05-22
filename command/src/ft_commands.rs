//! RediSearch (FT.*) command handlers
//! 
//! This module provides FT.* command implementations using HNSW for vector search.

#[cfg(feature = "vector-search")]
use parser::ParsedCommand;
#[cfg(feature = "vector-search")]
use response::Response;
#[cfg(feature = "vector-search")]
use database::Database;
#[cfg(feature = "vector-search")]
use redisearch_platform_core::vector_index::VectorMetric;
#[cfg(feature = "vector-search")]
use redisearch_platform_core::persistent_index::PersistentVectorIndex;
#[cfg(feature = "vector-search")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "vector-search")]
use std::collections::HashMap;

// Global index storage (indexed by database index and index name)
// Format: (dbindex, index_name) -> PersistentVectorIndex
#[cfg(feature = "vector-search")]
lazy_static::lazy_static! {
    static ref INDEX_STORAGE: Arc<Mutex<HashMap<(usize, String), Arc<Mutex<PersistentVectorIndex>>>>> = 
        Arc::new(Mutex::new(HashMap::new()));
}

// Index metadata stored in Redis with key: "ft:meta:{index_name}"
const INDEX_META_PREFIX: &[u8] = b"ft:meta:";

// Import validation macros (same pattern as command.rs)
macro_rules! try_validate {
    ($expr: expr, $err: expr) => {
        match $expr {
            Ok(r) => r,
            Err(_) => return Response::Error($err.to_string()),
        }
    };
}

macro_rules! validate_arguments_gte {
    ($parser: expr, $expected: expr) => {
        if $parser.argv.len() < $expected {
            return Response::Error(format!(
                "ERR wrong number of arguments for '{}' command",
                $parser.get_str(0).unwrap()
            ));
        }
    };
}

macro_rules! validate_arguments_exact {
    ($parser: expr, $expected: expr) => {
        if $parser.argv.len() != $expected {
            return Response::Error(format!(
                "ERR wrong number of arguments for '{}' command",
                $parser.get_str(0).unwrap()
            ));
        }
    };
}

/// Get or create index metadata key
fn index_meta_key(index_name: &str) -> Vec<u8> {
    let mut key = INDEX_META_PREFIX.to_vec();
    key.extend_from_slice(index_name.as_bytes());
    key
}

/// Get index from storage
#[cfg(feature = "vector-search")]
fn get_index(dbindex: usize, index_name: &str) -> Option<Arc<Mutex<PersistentVectorIndex>>> {
    let storage = INDEX_STORAGE.lock().unwrap();
    storage.get(&(dbindex, index_name.to_string())).cloned()
}

/// Create and store index
#[cfg(feature = "vector-search")]
fn create_index(dbindex: usize, index_name: String, dimension: usize, metric: VectorMetric, db: &Database) -> Arc<Mutex<PersistentVectorIndex>> {
    let storage_path = if db.config.storage_mode == "disk" {
        let path = std::path::Path::new(&db.config.dir).join(format!("{}_{}.redb", index_name, dbindex));
        Some(path)
    } else {
        None
    };
    
    let index = Arc::new(Mutex::new(PersistentVectorIndex::new(
        index_name.clone(),
        dimension,
        metric,
        storage_path.as_deref(),
        1000, // Snapshot every 1000 vectors
    ).expect("Failed to create persistent index")));
    
    let mut storage = INDEX_STORAGE.lock().unwrap();
    storage.insert((dbindex, index_name), index.clone());
    index
}

/// Handle FT.CREATE command
/// 
/// Syntax: FT.CREATE index [OPTIONS] schema
/// 
/// Creates a new search index with the specified schema.
pub fn ft_create(parser: &mut ParsedCommand, db: &mut Database, dbindex: usize) -> Response {
    // Debug: Always return a test message first to verify handler is called
    eprintln!("FT_CREATE handler called! argc={}", parser.argv.len());
    
    validate_arguments_gte!(parser, 2);
    
    let index_name = try_validate!(parser.get_str(1), "ERR invalid index name");
    
    // Basic validation
    if index_name.is_empty() {
        return Response::Error("ERR index name cannot be empty".to_owned());
    }
    
    // Check if index already exists
    let meta_key = index_meta_key(index_name);
    if db.get(dbindex, &meta_key).is_some() {
        return Response::Error("ERR index already exists".to_owned());
    }
    
    // Parse schema (simplified - look for VECTOR fields)
    // For now, create a default vector index with dimension 128
    let mut dimension = 128; // Default dimension
    let mut has_vector = false;
    
    // Simple schema parsing: look for "SCHEMA" keyword and VECTOR fields
    let mut i = 2;
    while i < parser.argv.len() {
        if let Ok(keyword) = parser.get_str(i) {
            if keyword.to_uppercase() == "SCHEMA" && i + 1 < parser.argv.len() {
                // Parse schema fields
                let mut j = i + 1;
                while j + 1 < parser.argv.len() {
                    if let (Ok(_field_name), Ok(field_type)) = (parser.get_str(j), parser.get_str(j + 1)) {
                        if field_type.to_uppercase().starts_with("VECTOR") {
                            has_vector = true;
                            // Try to parse dimension from VECTOR(dim)
                            if let Some(start) = field_type.find('(') {
                                if let Some(end) = field_type.find(')') {
                                    if let Ok(dim) = field_type[start+1..end].parse::<usize>() {
                                        dimension = dim;
                                    }
                                }
                            }
                        }
                        j += 2;
                    } else {
                        break;
                    }
                }
                break;
            }
        }
        i += 1;
    }
    
    // Create vector index if vector field found, otherwise just store metadata
    if has_vector {
        let _index = create_index(dbindex, index_name.to_string(), dimension, VectorMetric::Cosine, db);
    }
    
    // Store index metadata in Redis
    let meta_value = format!("dimension:{}", dimension);
    let meta_key = index_meta_key(index_name);
    let el = db.get_or_create(dbindex, &meta_key);
    if let Err(_) = el.set(meta_value.as_bytes().to_vec()) {
        return Response::Error("ERR failed to create index metadata".to_owned());
    }
    
    Response::Status("OK".to_owned())
}

/// Handle FT.SEARCH command
/// 
/// Syntax: FT.SEARCH index query [OPTIONS]
/// 
/// Searches the index with the given query.
pub fn ft_search(parser: &mut ParsedCommand, db: &Database, dbindex: usize) -> Response {
    validate_arguments_gte!(parser, 3);
    
    let index_name = try_validate!(parser.get_str(1), "ERR invalid index name");
    let query = try_validate!(parser.get_str(2), "ERR invalid query");
    
    // Check if index exists
    let meta_key = index_meta_key(index_name);
    if db.get(dbindex, &meta_key).is_none() {
        return Response::Error("ERR no such index".to_owned());
    }
    
    // For vector search: parse query as vector
    // Format: * or a vector query
    if query == "*" {
        // Return all documents (simplified)
        let mut result = Vec::new();
        result.push(Response::Integer(0)); // Count
        return Response::Array(result);
    }
    
    // Try to get vector index
    if let Some(index_arc) = get_index(dbindex, index_name) {
        // Parse query as vector (simplified - expect comma-separated floats)
        let vector: Result<Vec<f32>, _> = query
            .split(',')
            .map(|s| s.trim().parse::<f32>())
            .collect();
        
        if let Ok(query_vector) = vector {
            let index = index_arc.lock().unwrap();
            if let Ok(results) = index.search(&query_vector, 10) {
                let mut response = Vec::new();
                response.push(Response::Integer(results.len() as i64));
                
                for (doc_id, score) in results {
                    let mut doc = Vec::new();
                    doc.push(Response::Data(format!("{}", doc_id).into_bytes()));
                    doc.push(Response::Data(b"score".to_vec()));
                    doc.push(Response::Data(format!("{:.6}", score).into_bytes()));
                    response.push(Response::Array(doc));
                }
                
                return Response::Array(response);
            }
        }
    }
    
    // For text search (not yet implemented), return empty results
    let mut result = Vec::new();
    result.push(Response::Integer(0)); // Count of results
    Response::Array(result)
}

/// Handle FT.INFO command
/// 
/// Syntax: FT.INFO index
/// 
/// Returns information about the index.
pub fn ft_info(parser: &mut ParsedCommand, db: &Database, dbindex: usize) -> Response {
    validate_arguments_exact!(parser, 2);
    
    let index_name = try_validate!(parser.get_str(1), "ERR invalid index name");
    
    // Check if index exists
    let meta_key = index_meta_key(index_name);
    let _meta_value = match db.get(dbindex, &meta_key) {
        Some(v) => v,
        None => return Response::Error("ERR no such index".to_owned()),
    };
    
    // Get index stats if available
    let num_docs = if let Some(index_arc) = get_index(dbindex, index_name) {
        let index = index_arc.lock().unwrap();
        index.len()
    } else {
        0
    };
    
    // Build info response
    let mut info = Vec::new();
    info.push(Response::Data(b"index_name".to_vec()));
    info.push(Response::Data(index_name.as_bytes().to_vec()));
    info.push(Response::Data(b"num_docs".to_vec()));
    info.push(Response::Data(format!("{}", num_docs).into_bytes()));
    info.push(Response::Data(b"index_definition".to_vec()));
    info.push(Response::Array(vec![]));
    
    Response::Array(info)
}

/// Handle FT.DROP command
/// 
/// Syntax: FT.DROP index [DD]
/// 
/// Drops the index and optionally deletes associated keys.
pub fn ft_drop(parser: &mut ParsedCommand, db: &mut Database, dbindex: usize) -> Response {
    validate_arguments_gte!(parser, 2);
    
    let index_name = try_validate!(parser.get_str(1), "ERR invalid index name");
    let delete_docs = parser.argv.len() > 2 && 
        parser.get_str(2).map(|s| s.to_uppercase() == "DD").unwrap_or(false);
    
    // Check if index exists
    let meta_key = index_meta_key(index_name);
    if db.get(dbindex, &meta_key).is_none() {
        return Response::Error("ERR no such index".to_owned());
    }
    
    // Remove from storage
    let mut storage = INDEX_STORAGE.lock().unwrap();
    storage.remove(&(dbindex, index_name.to_string()));
    
    // Delete metadata
    db.remove(dbindex, &meta_key);
    
    // If DD option, delete associated document keys (simplified)
    if delete_docs {
        // TODO: Track and delete document keys
    }
    
    Response::Status("OK".to_owned())
}

/// Handle FT.ADD command
/// 
/// Syntax: FT.ADD index docId score [FIELDS field value ...] [REPLACE] [PARTIAL] [LANGUAGE lang] [PAYLOAD payload]
/// 
/// Adds a document to the index.
pub fn ft_add(parser: &mut ParsedCommand, db: &mut Database, dbindex: usize) -> Response {
    validate_arguments_gte!(parser, 4);
    
    let index_name = try_validate!(parser.get_str(1), "ERR invalid index name");
    let doc_id_str = try_validate!(parser.get_str(2), "ERR invalid document ID");
    let _score = try_validate!(parser.get_f64(3), "ERR invalid score");
    
    // Check if index exists
    let meta_key = index_meta_key(index_name);
    if db.get(dbindex, &meta_key).is_none() {
        return Response::Error("ERR no such index".to_owned());
    }
    
    // Parse FIELDS to extract vector
    let mut vector: Option<Vec<f32>> = None;
    let mut i = 4;
    while i + 1 < parser.argv.len() {
        if let (Ok(field), Ok(_value)) = (parser.get_str(i), parser.get_str(i + 1)) {
            if field.to_uppercase() == "FIELDS" {
                // Look for vector field
                let mut j = i + 1;
                while j + 1 < parser.argv.len() {
                    if let (Ok(fname), Ok(fval)) = (parser.get_str(j), parser.get_str(j + 1)) {
                        if fname.to_lowercase().contains("vector") {
                            // Parse vector (comma-separated floats)
                            let vec_result: Result<Vec<f32>, _> = fval
                                .split(',')
                                .map(|s| s.trim().parse::<f32>())
                                .collect();
                            if let Ok(v) = vec_result {
                                vector = Some(v);
                            }
                        }
                        j += 2;
                    } else {
                        break;
                    }
                }
                break;
            }
        }
        i += 1;
    }
    
    // Add vector to index if found
    if let Some(query_vector) = vector {
        // Get or create index
        let index_arc = get_index(dbindex, index_name)
            .unwrap_or_else(|| {
                // Create default index if not found
                create_index(dbindex, index_name.to_string(), query_vector.len(), VectorMetric::Cosine, db)
            });
        
        let mut index = index_arc.lock().unwrap();
        // Use doc_id as numeric ID (simplified - hash the string)
        let doc_id = doc_id_str.as_bytes().iter().fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64));
        
        if let Err(e) = index.add(doc_id, query_vector) {
            return Response::Error(format!("ERR failed to add vector: {}", e));
        }
    }
    
    Response::Status("OK".to_owned())
}

/// Handle FT.DEL command
/// 
/// Syntax: FT.DEL index docId [DD]
/// 
/// Deletes a document from the index.
pub fn ft_del(parser: &mut ParsedCommand, db: &mut Database, dbindex: usize) -> Response {
    validate_arguments_gte!(parser, 3);
    
    let index_name = try_validate!(parser.get_str(1), "ERR invalid index name");
    let doc_id_str = try_validate!(parser.get_str(2), "ERR invalid document ID");
    let delete_key = parser.argv.len() > 3 && 
        parser.get_str(3).map(|s| s.to_uppercase() == "DD").unwrap_or(false);
    
    // Check if index exists
    let meta_key = index_meta_key(index_name);
    if db.get(dbindex, &meta_key).is_none() {
        return Response::Error("ERR no such index".to_owned());
    }
    
    // Delete from vector index if it exists
    let mut deleted = false;
    if let Some(index_arc) = get_index(dbindex, index_name) {
        let mut index = index_arc.lock().unwrap();
        let doc_id = doc_id_str.as_bytes().iter().fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64));
        
        if index.remove(doc_id).is_ok() {
            deleted = true;
        }
    }
    
    // If DD option, delete the associated key
    if delete_key {
        let doc_key = doc_id_str.as_bytes().to_vec();
        db.remove(dbindex, &doc_key);
    }
    
    Response::Integer(if deleted { 1 } else { 0 })
}

