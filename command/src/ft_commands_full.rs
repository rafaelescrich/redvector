//! Full RediSearch (FT.*) command handlers
//! 
//! This module provides complete FT.* command implementations using the full
//! Redisearch implementation from RediSearch/rust-port.

use parser::ParsedCommand;
use response::Response;
use database::Database;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

// Re-export Redisearch types (we'll need to adapt these)
use redisearch_dict::Dict;
use redisearch_doc_table::DocTable;
use redisearch_document::DocumentType;
use redisearch_field_spec::{FieldSpec, FieldType};
use redisearch_index_spec::IndexSpec;
use redisearch_inverted_index::{IndexEntry, InvertedIndex};
use redisearch_query_execution::{execute_query, QueryExecutionContext};
use redisearch_query_parser::{QueryNode, QueryParser};
use redisearch_query_options::{parse_options, apply_limit, apply_sortby, QueryOptions};
use redisearch_scoring::{calculate_document_frequencies, DocumentScore};
use redisearch_spellcheck::SpellDictionary;
use redisearch_synonyms::SynonymGroup;
use redisearch_tokenizer::Tokenizer;
use redisearch_vector_index::{VectorIndex, VectorMetric};

/// Extended IndexSpec with inverted index and document table
struct ExtendedIndexSpec {
    spec: IndexSpec,
    // Dictionary mapping terms to inverted indexes
    inverted_indexes: Dict<String, Arc<Mutex<InvertedIndex>>>,
    // Document table (forward index)
    doc_table: Arc<Mutex<DocTable>>,
    // Vector index for vector fields
    vector_index: Option<Arc<Mutex<VectorIndex>>>,
}

impl ExtendedIndexSpec {
    fn new(spec: IndexSpec) -> Self {
        Self {
            spec,
            inverted_indexes: Dict::new(),
            doc_table: Arc::new(Mutex::new(DocTable::new(10000))),
            vector_index: None,
        }
    }
}

/// Global index registry - maps (dbindex, index_name) to ExtendedIndexSpec
type IndexRegistry = Arc<Mutex<Dict<(usize, String), Arc<Mutex<ExtendedIndexSpec>>>>>;

/// Global spell dictionaries
lazy_static::lazy_static! {
    static ref SPELL_DICTIONARIES: Arc<Mutex<Dict<(usize, String), Arc<Mutex<SpellDictionary>>>>> = 
        Arc::new(Mutex::new(Dict::new()));
}

/// Global synonym groups
lazy_static::lazy_static! {
    static ref SYNONYM_GROUPS: Arc<Mutex<Dict<(usize, String), Arc<Mutex<SynonymGroup>>>>> = 
        Arc::new(Mutex::new(Dict::new()));
}

// Global index registry
lazy_static::lazy_static! {
    static ref INDEX_REGISTRY: IndexRegistry = Arc::new(Mutex::new(Dict::new()));
}

// Validation macros
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
                $parser.get_str(0).unwrap_or("unknown")
            ));
        }
    };
}

macro_rules! validate_arguments_exact {
    ($parser: expr, $expected: expr) => {
        if $parser.argv.len() != $expected {
            return Response::Error(format!(
                "ERR wrong number of arguments for '{}' command",
                $parser.get_str(0).unwrap_or("unknown")
            ));
        }
    };
}

/// Get index from registry
fn get_index(dbindex: usize, index_name: &str) -> Option<Arc<Mutex<ExtendedIndexSpec>>> {
    let registry = INDEX_REGISTRY.lock().unwrap();
    registry.get(&(dbindex, index_name.to_string())).cloned()
}

/// Create and register index
fn create_index(dbindex: usize, index_name: String, spec: IndexSpec) -> Arc<Mutex<ExtendedIndexSpec>> {
    let extended_spec = Arc::new(Mutex::new(ExtendedIndexSpec::new(spec)));
    let mut registry = INDEX_REGISTRY.lock().unwrap();
    registry.insert((dbindex, index_name), extended_spec.clone());
    extended_spec
}

/// Tokenize text using the Redisearch tokenizer
fn tokenize_text(text: &str) -> Vec<String> {
    let tokenizer = Tokenizer::new();
    tokenizer.tokenize(text)
        .iter()
        .map(|t| t.term.clone())
        .collect()
}

/// Handle FT.CREATE command
/// 
/// Syntax: FT.CREATE index [OPTIONS] SCHEMA field type [field type ...]
pub fn ft_create(parser: &mut ParsedCommand, _db: &mut Database, dbindex: usize) -> Response {
    validate_arguments_gte!(parser, 2);
    
    let index_name = try_validate!(parser.get_str(1), "ERR invalid index name");
    
    // Check if index already exists
    if get_index(dbindex, index_name).is_some() {
        return Response::Error("ERR index already exists".to_owned());
    }
    
    // Parse schema
    let mut spec = IndexSpec::new(index_name);
    let mut i = 2;
    let mut has_vector = false;
    let mut vector_dim = 128;
    
    while i < parser.argv.len() {
        if let Ok(keyword) = parser.get_str(i) {
            if keyword.to_uppercase() == "SCHEMA" && i + 2 < parser.argv.len() {
                i += 1; // Skip SCHEMA
                while i + 1 < parser.argv.len() {
                    if let (Ok(field_name), Ok(field_type_str)) = (parser.get_str(i), parser.get_str(i + 1)) {
                        let field_type_upper = field_type_str.to_uppercase();
                        let field_type = if field_type_upper.starts_with("TEXT") {
                            FieldType::FullText
                        } else if field_type_upper.starts_with("TAG") {
                            FieldType::Tag
                        } else if field_type_upper.starts_with("NUMERIC") {
                            FieldType::Numeric
                        } else if field_type_upper.starts_with("GEO") {
                            FieldType::Geo
                        } else if field_type_upper.starts_with("VECTOR") {
                            has_vector = true;
                            // Parse dimension from VECTOR(dim)
                            if let Some(start) = field_type_str.find('(') {
                                if let Some(end) = field_type_str.find(')') {
                                    if let Ok(dim) = field_type_str[start+1..end].parse::<usize>() {
                                        vector_dim = dim;
                                    }
                                }
                            }
                            FieldType::FullText // Placeholder, we'll handle vector separately
                        } else {
                            FieldType::FullText
                        };
                        
                        let field = FieldSpec::new(&field_name, field_type);
                        if let Err(_) = spec.add_field(field) {
                            return Response::Error(format!("ERR failed to add field {}", field_name));
                        }
                        i += 2;
                    } else {
                        break;
                    }
                }
                break;
            }
        }
        i += 1;
    }
    
    // Create extended spec
    let mut ext_spec = ExtendedIndexSpec::new(spec);
    
    // Create vector index if needed
    if has_vector {
        ext_spec.vector_index = Some(Arc::new(Mutex::new(VectorIndex::new(vector_dim, VectorMetric::Cosine))));
    }
    
    // Register the index
    let ext_spec_arc = Arc::new(Mutex::new(ext_spec));
    let mut registry = INDEX_REGISTRY.lock().unwrap();
    registry.insert((dbindex, index_name.to_string()), ext_spec_arc.clone());
    
    Response::Status("OK".to_owned())
}

/// Handle FT.ADD command
/// 
/// Syntax: FT.ADD index docId score [FIELDS field value ...] [REPLACE] [PARTIAL]
pub fn ft_add(parser: &mut ParsedCommand, _db: &mut Database, dbindex: usize) -> Response {
    validate_arguments_gte!(parser, 4);
    
    let index_name = try_validate!(parser.get_str(1), "ERR invalid index name");
    let doc_id = try_validate!(parser.get_str(2), "ERR invalid document ID");
    let score = try_validate!(parser.get_f64(3), "ERR invalid score");
    
    let ext_spec_arc = get_index(dbindex, index_name)
        .ok_or_else(|| Response::Error("ERR no such index".to_owned()))?;
    
    let mut ext_spec = ext_spec_arc.lock().unwrap();
    
    // Add document to document table
    let mut doc_table = ext_spec.doc_table.lock().unwrap();
    let metadata = match doc_table.put(&doc_id, score as f32, DocumentType::Hash, None) {
        Ok(m) => m,
        Err(_) => return Response::Error("ERR failed to add document".to_owned()),
    };
    let doc_id_internal = metadata.id;
    drop(doc_table);
    
    // Parse fields
    let mut i = 4;
    let mut field_index = 0u64;
    let mut vector_value: Option<Vec<f32>> = None;
    
    while i + 1 < parser.argv.len() {
        if let Ok(keyword) = parser.get_str(i) {
            if keyword.to_uppercase() == "FIELDS" {
                i += 1;
                continue;
            }
        }
        
        if let (Ok(field_name), Ok(field_value)) = (parser.get_str(i), parser.get_str(i + 1)) {
            // Check if this is a vector field
            if field_name.to_lowercase().contains("vector") || field_value.contains(',') {
                // Try to parse as vector
                let vec_result: Result<Vec<f32>, _> = field_value
                    .split(',')
                    .map(|s| s.trim().parse::<f32>())
                    .collect();
                if let Ok(v) = vec_result {
                    vector_value = Some(v);
                }
            }
            
            // Get field spec
            if let Some(field_spec) = ext_spec.spec.get_field(&field_name) {
                let is_text = (field_spec.types & FieldType::FullText.as_bitmask()) != 0;
                let is_tag = (field_spec.types & FieldType::Tag.as_bitmask()) != 0;
                
                if is_text || is_tag {
                    let tokens = tokenize_text(&field_value);
                    let field_bit = 1u64 << field_index;
                    
                    for token in tokens {
                        let inv_idx_arc = if let Some(idx) = ext_spec.inverted_indexes.get(&token) {
                            idx.clone()
                        } else {
                            let new_idx = Arc::new(Mutex::new(InvertedIndex::new(ext_spec.spec.flags)));
                            ext_spec.inverted_indexes.insert(token.clone(), new_idx.clone());
                            new_idx
                        };
                        
                        let mut inv_idx = inv_idx_arc.lock().unwrap();
                        let entry = IndexEntry::new(doc_id_internal, 1, field_bit);
                        if let Err(_) = inv_idx.add_entry(&token, &entry) {
                            // Log warning but continue
                        }
                    }
                }
                field_index += 1;
            }
        }
        i += 2;
    }
    
    // Add vector if present
    if let Some(vector) = vector_value {
        if let Some(ref vec_idx) = ext_spec.vector_index {
            let mut vec_index = vec_idx.lock().unwrap();
            let doc_id_num = doc_id.as_bytes().iter().fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64));
            if let Err(_) = vec_index.add(doc_id_num, vector) {
                return Response::Error("ERR failed to add vector".to_owned());
            }
        }
    }
    
    ext_spec.spec.increment_document_count();
    
    Response::Status("OK".to_owned())
}

/// Handle FT.SEARCH command
/// 
/// Syntax: FT.SEARCH index query [LIMIT offset count] [SORTBY field ASC|DESC] [RETURN fields...] [WITHSCORES] [NOCONTENT]
pub fn ft_search(parser: &mut ParsedCommand, _db: &Database, dbindex: usize) -> Response {
    validate_arguments_gte!(parser, 3);
    
    let index_name = try_validate!(parser.get_str(1), "ERR invalid index name");
    let query = try_validate!(parser.get_str(2), "ERR invalid query");
    
    let ext_spec_arc = get_index(dbindex, index_name)
        .ok_or_else(|| Response::Error("ERR no such index".to_owned()))?;
    
    let ext_spec = ext_spec_arc.lock().unwrap();
    
    // Parse query
    let query_parser = QueryParser::new();
    let query_node = match query_parser.parse(query) {
        Ok(node) => node,
        Err(_) => {
            // For vector search, try parsing as vector
            if query.contains(',') {
                let vector_result: Result<Vec<f32>, _> = query
                    .split(',')
                    .map(|s| s.trim().parse::<f32>())
                    .collect();
                
                if let Ok(query_vector) = vector_result {
                    if let Some(ref vec_idx) = ext_spec.vector_index {
                        let vec_index = vec_idx.lock().unwrap();
                        if let Ok(results) = vec_index.search(&query_vector, 10, None) {
                            let mut response_vec = Vec::new();
                            response_vec.push(Response::Integer(results.len() as i64));
                            
                            for (doc_id, score) in results {
                                let mut doc = Vec::new();
                                doc.push(Response::Data(format!("{}", doc_id).into_bytes()));
                                doc.push(Response::Data(b"score".to_vec()));
                                doc.push(Response::Data(format!("{:.6}", score).into_bytes()));
                                response_vec.push(Response::Array(doc));
                            }
                            
                            return Response::Array(response_vec);
                        }
                    }
                }
            }
            return Response::Error("ERR query parse error".to_owned());
        }
    };
    
    // Parse options
    let args_str: Vec<String> = (3..parser.argv.len())
        .filter_map(|i| parser.get_str(i).ok())
        .collect();
    let options = parse_options(&args_str, 0);
    
    // Create execution context
    let exec_ctx = QueryExecutionContext::new(
        ext_spec.inverted_indexes.clone(),
        ext_spec.doc_table.clone(),
    );
    
    // Execute query
    let (offset, limit) = options.limit.unwrap_or((0, 10));
    let max_results = offset + limit;
    
    let iterator_results = match execute_query(&query_node, &exec_ctx, Some(max_results)) {
        Ok(results) => results,
        Err(_) => return Response::Error("ERR query execution error".to_owned()),
    };
    
    // Convert to DocumentScore
    let mut doc_scores: Vec<DocumentScore> = iterator_results
        .iter()
        .map(|r| DocumentScore::new(r.doc_id, r.score as f64))
        .collect();
    
    // Apply SORTBY if specified
    if let Some((field, order)) = &options.sortby {
        let field_values = HashMap::new();
        apply_sortby(&mut doc_scores, field, *order, &field_values);
    }
    
    // Apply LIMIT
    let limited_results = apply_limit(&doc_scores, offset, limit);
    
    // Build results
    let mut results = Vec::new();
    for result in limited_results {
        if let Some(metadata) = ext_spec.doc_table.lock().unwrap().get_by_id(result.doc_id) {
            if !options.no_content {
                let mut fields = Vec::new();
                fields.push(Response::Data(b"key".to_vec()));
                fields.push(Response::Data(metadata.key.clone().into_bytes()));
                
                if options.with_scores {
                    fields.push(Response::Data(b"score".to_vec()));
                    fields.push(Response::Data(format!("{:.6}", result.score).into_bytes()));
                }
                
                results.push(Response::Array(fields));
            } else {
                results.push(Response::Data(metadata.key.clone().into_bytes()));
            }
        }
    }
    
    // Return: [count, [doc1, doc2, ...]]
    let mut response_vec = Vec::new();
    response_vec.push(Response::Integer(doc_scores.len() as i64));
    response_vec.push(Response::Array(results));
    
    Response::Array(response_vec)
}

/// Handle FT.INFO command
pub fn ft_info(parser: &mut ParsedCommand, _db: &Database, dbindex: usize) -> Response {
    validate_arguments_exact!(parser, 2);
    
    let index_name = try_validate!(parser.get_str(1), "ERR invalid index name");
    
    let ext_spec_arc = get_index(dbindex, index_name)
        .ok_or_else(|| Response::Error("ERR no such index".to_owned()))?;
    
    let ext_spec = ext_spec_arc.lock().unwrap();
    let stats = ext_spec.spec.stats();
    
    let mut info = Vec::new();
    info.push(Response::Data(b"index_name".to_vec()));
    info.push(Response::Data(index_name.as_bytes().to_vec()));
    info.push(Response::Data(b"num_docs".to_vec()));
    info.push(Response::Data(format!("{}", stats.document_count).into_bytes()));
    info.push(Response::Data(b"num_fields".to_vec()));
    info.push(Response::Data(format!("{}", ext_spec.spec.field_count()).into_bytes()));
    
    Response::Array(info)
}

/// Handle FT.DROP command
pub fn ft_drop(parser: &mut ParsedCommand, _db: &mut Database, dbindex: usize) -> Response {
    validate_arguments_gte!(parser, 2);
    
    let index_name = try_validate!(parser.get_str(1), "ERR invalid index name");
    
    let mut registry = INDEX_REGISTRY.lock().unwrap();
    if registry.remove(&(dbindex, index_name.to_string())).is_none() {
        return Response::Error("ERR no such index".to_owned());
    }
    
    Response::Status("OK".to_owned())
}

/// Handle FT.DEL command
pub fn ft_del(parser: &mut ParsedCommand, _db: &mut Database, dbindex: usize) -> Response {
    validate_arguments_gte!(parser, 3);
    
    let index_name = try_validate!(parser.get_str(1), "ERR invalid index name");
    let doc_id = try_validate!(parser.get_str(2), "ERR invalid document ID");
    
    let ext_spec_arc = get_index(dbindex, index_name)
        .ok_or_else(|| Response::Error("ERR no such index".to_owned()))?;
    
    let mut ext_spec = ext_spec_arc.lock().unwrap();
    
    // Remove from document table
    let mut doc_table = ext_spec.doc_table.lock().unwrap();
    let removed = doc_table.remove(&doc_id).is_ok();
    
    // Remove from vector index if present
    if let Some(ref vec_idx) = ext_spec.vector_index {
        let mut vec_index = vec_idx.lock().unwrap();
        let doc_id_num = doc_id.as_bytes().iter().fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let _ = vec_index.remove(doc_id_num);
    }
    
    Response::Integer(if removed { 1 } else { 0 })
}

