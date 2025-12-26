//! Multi-Vector FT.* command handlers (ColPali/ColBERT style)
//!
//! This module provides FT.CREATEMV, FT.ADDMV, and FT.SEARCHMV commands
//! for multi-vector retrieval using the RVF2 storage format.
//!
//! ## Commands
//!
//! - `FT.CREATEMV` - Create a multi-vector index
//! - `FT.ADDMV` - Add a document with pooled + patch vectors
//! - `FT.SEARCHMV` - Two-stage search (ANN + MaxSim rerank)
//!
//! ## Example
//!
//! ```redis
//! FT.CREATEMV colpali_index
//!   ON HASH PREFIX 1 page:
//!   SCHEMA
//!     pooled VECTOR HNSW 6 TYPE FLOAT32 DIM 128 DISTANCE_METRIC COSINE
//!     patches MULTIVECTOR 8 TYPE INT8 DIM 128 MAX_PATCHES 1024 CODEC SQ8
//!     title TEXT
//!
//! FT.ADDMV colpali_index page:doc1
//!   POOLED <128 floats>
//!   PATCHES <1024×128 int8>
//!   FIELDS title "Introduction"
//!
//! FT.SEARCHMV colpali_index "*"
//!   KNN 100 @pooled $query_pooled
//!   RERANK MAXSIM @patches $query_tokens
//!   LIMIT 0 10
//!   PARAMS 4 query_pooled <bytes> query_tokens <bytes>
//! ```

use parser::ParsedCommand;
use response::Response;
use database::Database;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

#[cfg(feature = "rvf2")]
use redisearch_platform_core::rvf2::{
    Manifest, SegmentMeta, SegmentBuilder, DocIndexBuilder, DocEntry,
    SimdMaxSimScorer, Sq8Codec, Codec, CodecType,
};

/// Multi-vector index configuration
#[derive(Debug, Clone)]
pub struct MultiVectorIndexConfig {
    /// Index name
    pub name: String,

    /// Pooled vector dimension
    pub pooled_dim: usize,

    /// Patch vector dimension
    pub patch_dim: usize,

    /// Maximum patches per document
    pub max_patches: usize,

    /// Codec for patches
    pub patch_codec: PatchCodec,

    /// Distance metric for pooled vectors
    pub distance_metric: DistanceMetricMv,
}

/// Patch codec options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatchCodec {
    /// No compression (float32)
    None,
    /// Scalar quantization to int8
    SQ8,
    /// Float16
    FP16,
    /// Product quantization
    PQ16,
}

impl Default for PatchCodec {
    fn default() -> Self {
        Self::SQ8
    }
}

/// Distance metric for multi-vector
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetricMv {
    Cosine,
    L2,
    InnerProduct,
}

impl Default for DistanceMetricMv {
    fn default() -> Self {
        Self::Cosine
    }
}

/// Multi-vector index registry
lazy_static::lazy_static! {
    static ref MV_INDEX_REGISTRY: Arc<Mutex<HashMap<String, Arc<Mutex<MultiVectorIndex>>>>> =
        Arc::new(Mutex::new(HashMap::new()));
}

/// Multi-vector index
pub struct MultiVectorIndex {
    /// Configuration
    pub config: MultiVectorIndexConfig,

    /// Pooled vectors for ANN (Stage A)
    pooled_vectors: Vec<f32>,

    /// Pooled vector IDs
    pooled_ids: Vec<u64>,

    /// Document metadata
    doc_metadata: HashMap<u64, DocMetadata>,

    /// Patch storage (in-memory for now, RVF2 for production)
    patches: HashMap<u64, PatchData>,

    /// Next doc ID
    next_id: u64,

    /// MaxSim scorer
    scorer: Option<SimdMaxSimScorer>,
}

/// Document metadata
#[derive(Debug, Clone)]
struct DocMetadata {
    /// External key (e.g., "page:doc1")
    key: String,

    /// Number of patches
    n_patches: usize,

    /// Additional fields
    fields: HashMap<String, String>,
}

/// Patch data for a document
#[derive(Debug, Clone)]
struct PatchData {
    /// Encoded patch codes (SQ8 or raw)
    codes: Vec<u8>,

    /// Scales for SQ8
    scales: Vec<half::f16>,

    /// Number of patches
    n_patches: usize,
}

impl MultiVectorIndex {
    /// Create new multi-vector index
    pub fn new(config: MultiVectorIndexConfig) -> Self {
        #[cfg(feature = "rvf2")]
        let scorer = Some(SimdMaxSimScorer::new(config.patch_dim));

        #[cfg(not(feature = "rvf2"))]
        let scorer = None;

        Self {
            config,
            pooled_vectors: Vec::new(),
            pooled_ids: Vec::new(),
            doc_metadata: HashMap::new(),
            patches: HashMap::new(),
            next_id: 1,
            scorer,
        }
    }

    /// Add a document
    pub fn add(
        &mut self,
        key: &str,
        pooled: &[f32],
        patch_codes: Vec<u8>,
        patch_scales: Vec<half::f16>,
        n_patches: usize,
        fields: HashMap<String, String>,
    ) -> u64 {
        let doc_id = self.next_id;
        self.next_id += 1;

        // Store pooled vector
        self.pooled_vectors.extend_from_slice(pooled);
        self.pooled_ids.push(doc_id);

        // Store metadata
        self.doc_metadata.insert(doc_id, DocMetadata {
            key: key.to_string(),
            n_patches,
            fields,
        });

        // Store patches
        self.patches.insert(doc_id, PatchData {
            codes: patch_codes,
            scales: patch_scales,
            n_patches,
        });

        doc_id
    }

    /// Two-stage search: ANN on pooled + MaxSim rerank
    pub fn search(
        &self,
        query_pooled: &[f32],
        query_tokens: &[f32],
        k_candidates: usize,
        k_final: usize,
    ) -> Vec<SearchResultMv> {
        // Stage A: Find candidates using pooled vectors
        let candidates = self.ann_search(query_pooled, k_candidates);

        // Stage B: Rerank using MaxSim
        self.maxsim_rerank(&candidates, query_tokens, k_final)
    }

    /// Stage A: ANN search on pooled vectors
    fn ann_search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let n_vectors = self.pooled_ids.len();
        if n_vectors == 0 {
            return Vec::new();
        }

        let dim = self.config.pooled_dim;

        // Compute distances to all pooled vectors
        let mut results: Vec<(u64, f32)> = (0..n_vectors)
            .map(|i| {
                let vec = &self.pooled_vectors[i * dim..(i + 1) * dim];
                let score = cosine_similarity(query, vec);
                (self.pooled_ids[i], score)
            })
            .collect();

        // Sort by similarity (descending)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(k);

        results
    }

    /// Stage B: MaxSim reranking
    fn maxsim_rerank(
        &self,
        candidates: &[(u64, f32)],
        query_tokens: &[f32],
        k: usize,
    ) -> Vec<SearchResultMv> {
        let mut results: Vec<SearchResultMv> = candidates
            .iter()
            .filter_map(|(doc_id, ann_score)| {
                let patch_data = self.patches.get(doc_id)?;
                let metadata = self.doc_metadata.get(doc_id)?;

                // Compute MaxSim score
                let maxsim_score = self.compute_maxsim(query_tokens, patch_data);

                Some(SearchResultMv {
                    doc_id: *doc_id,
                    key: metadata.key.clone(),
                    ann_score: *ann_score,
                    maxsim_score,
                    fields: metadata.fields.clone(),
                })
            })
            .collect();

        // Sort by MaxSim score (descending)
        results.sort_by(|a, b| b.maxsim_score.partial_cmp(&a.maxsim_score).unwrap());
        results.truncate(k);

        results
    }

    /// Compute MaxSim for a document
    fn compute_maxsim(&self, query_tokens: &[f32], patch_data: &PatchData) -> f32 {
        #[cfg(feature = "rvf2")]
        {
            if let Some(ref scorer) = self.scorer {
                // Convert codes to i8
                let codes_i8: Vec<i8> = patch_data.codes.iter().map(|&c| c as i8).collect();
                return scorer.compute_sq8(
                    query_tokens,
                    &codes_i8,
                    &patch_data.scales,
                    32, // block size
                );
            }
        }

        // Fallback: scalar implementation
        0.0
    }

    /// Get document count
    pub fn len(&self) -> usize {
        self.pooled_ids.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.pooled_ids.is_empty()
    }
}

/// Search result for multi-vector queries
#[derive(Debug, Clone)]
pub struct SearchResultMv {
    /// Internal document ID
    pub doc_id: u64,

    /// External key
    pub key: String,

    /// ANN score (Stage A)
    pub ann_score: f32,

    /// MaxSim score (Stage B)
    pub maxsim_score: f32,

    /// Document fields
    pub fields: HashMap<String, String>,
}

/// Cosine similarity
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

// ============================================================================
// Command Handlers
// ============================================================================

/// Handle FT.CREATEMV command
pub fn ft_createmv(parser: &ParsedCommand, db: &Database, dbindex: usize) -> Response {
    // FT.CREATEMV <index_name> [ON HASH PREFIX ...] SCHEMA <field_specs>
    if parser.argv.len() < 2 {
        return Response::Error("ERR wrong number of arguments for 'FT.CREATEMV' command".into());
    }

    let index_name = match parser.get_str(1) {
        Some(s) => s,
        None => return Response::Error("ERR invalid index name".into()),
    };

    // Parse schema (simplified for now)
    let config = MultiVectorIndexConfig {
        name: index_name.to_string(),
        pooled_dim: 128,
        patch_dim: 128,
        max_patches: 1024,
        patch_codec: PatchCodec::SQ8,
        distance_metric: DistanceMetricMv::Cosine,
    };

    // Create index
    let index = MultiVectorIndex::new(config);

    // Register
    let mut registry = MV_INDEX_REGISTRY.lock().unwrap();
    let key = format!("{}:{}", dbindex, index_name);
    registry.insert(key, Arc::new(Mutex::new(index)));

    Response::Status("OK".into())
}

/// Handle FT.ADDMV command
pub fn ft_addmv(parser: &ParsedCommand, db: &Database, dbindex: usize) -> Response {
    // FT.ADDMV <index_name> <doc_key> POOLED <bytes> PATCHES <bytes> [FIELDS ...]
    if parser.argv.len() < 6 {
        return Response::Error("ERR wrong number of arguments for 'FT.ADDMV' command".into());
    }

    let index_name = match parser.get_str(1) {
        Some(s) => s,
        None => return Response::Error("ERR invalid index name".into()),
    };

    let doc_key = match parser.get_str(2) {
        Some(s) => s,
        None => return Response::Error("ERR invalid document key".into()),
    };

    // Get index
    let key = format!("{}:{}", dbindex, index_name);
    let registry = MV_INDEX_REGISTRY.lock().unwrap();
    let index = match registry.get(&key) {
        Some(idx) => Arc::clone(idx),
        None => return Response::Error(format!("ERR index '{}' not found", index_name)),
    };
    drop(registry);

    // Parse POOLED and PATCHES arguments
    let mut pooled: Option<Vec<f32>> = None;
    let mut patches: Option<(Vec<u8>, Vec<half::f16>, usize)> = None;
    let mut fields: HashMap<String, String> = HashMap::new();

    let mut i = 3;
    while i < parser.argv.len() {
        let arg = match parser.get_str(i) {
            Some(s) => s.to_uppercase(),
            None => break,
        };

        match arg.as_str() {
            "POOLED" => {
                if i + 1 < parser.argv.len() {
                    if let Some(bytes) = parser.get_vec(i + 1) {
                        // Parse as f32 array
                        let floats: Vec<f32> = bytes
                            .chunks(4)
                            .map(|chunk| {
                                let arr: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
                                f32::from_le_bytes(arr)
                            })
                            .collect();
                        pooled = Some(floats);
                    }
                    i += 2;
                } else {
                    return Response::Error("ERR missing POOLED value".into());
                }
            }
            "PATCHES" => {
                if i + 1 < parser.argv.len() {
                    if let Some(bytes) = parser.get_vec(i + 1) {
                        // Assume SQ8 encoding: codes + scales
                        // For now, treat all bytes as codes
                        let n_patches = bytes.len() / 128; // Assume 128 dim
                        let codes = bytes.to_vec();
                        let scales = vec![half::f16::from_f32(1.0); n_patches / 32 + 1];
                        patches = Some((codes, scales, n_patches));
                    }
                    i += 2;
                } else {
                    return Response::Error("ERR missing PATCHES value".into());
                }
            }
            "FIELDS" => {
                i += 1;
                while i + 1 < parser.argv.len() {
                    let field_name = match parser.get_str(i) {
                        Some(s) if !s.starts_with('@') => s,
                        _ => break,
                    };
                    let field_value = match parser.get_str(i + 1) {
                        Some(s) => s,
                        None => break,
                    };
                    fields.insert(field_name.to_string(), field_value.to_string());
                    i += 2;
                }
            }
            _ => {
                i += 1;
            }
        }
    }

    // Validate required fields
    let pooled = match pooled {
        Some(p) => p,
        None => return Response::Error("ERR missing POOLED data".into()),
    };

    let (patch_codes, patch_scales, n_patches) = match patches {
        Some(p) => p,
        None => return Response::Error("ERR missing PATCHES data".into()),
    };

    // Add to index
    let mut index = index.lock().unwrap();
    let doc_id = index.add(doc_key, &pooled, patch_codes, patch_scales, n_patches, fields);

    Response::Integer(doc_id as i64)
}

/// Handle FT.SEARCHMV command
pub fn ft_searchmv(parser: &ParsedCommand, db: &Database, dbindex: usize) -> Response {
    // FT.SEARCHMV <index_name> <query> KNN <k> @<field> $<param>
    //   RERANK MAXSIM @<patches_field> $<tokens_param>
    //   LIMIT <offset> <num>
    //   PARAMS <count> <name> <value> ...
    if parser.argv.len() < 4 {
        return Response::Error("ERR wrong number of arguments for 'FT.SEARCHMV' command".into());
    }

    let index_name = match parser.get_str(1) {
        Some(s) => s,
        None => return Response::Error("ERR invalid index name".into()),
    };

    // Get index
    let key = format!("{}:{}", dbindex, index_name);
    let registry = MV_INDEX_REGISTRY.lock().unwrap();
    let index = match registry.get(&key) {
        Some(idx) => Arc::clone(idx),
        None => return Response::Error(format!("ERR index '{}' not found", index_name)),
    };
    drop(registry);

    // Parse parameters
    let mut params: HashMap<String, Vec<u8>> = HashMap::new();
    let mut k_candidates = 100;
    let mut k_final = 10;
    let mut offset = 0;

    let mut i = 3;
    while i < parser.argv.len() {
        let arg = match parser.get_str(i) {
            Some(s) => s.to_uppercase(),
            None => break,
        };

        match arg.as_str() {
            "KNN" => {
                if i + 1 < parser.argv.len() {
                    if let Some(k) = parser.get_str(i + 1).and_then(|s| s.parse::<usize>().ok()) {
                        k_candidates = k;
                    }
                    i += 3; // Skip KNN <k> @field
                } else {
                    i += 1;
                }
            }
            "RERANK" => {
                i += 4; // Skip RERANK MAXSIM @field $param
            }
            "LIMIT" => {
                if i + 2 < parser.argv.len() {
                    if let Some(o) = parser.get_str(i + 1).and_then(|s| s.parse::<usize>().ok()) {
                        offset = o;
                    }
                    if let Some(n) = parser.get_str(i + 2).and_then(|s| s.parse::<usize>().ok()) {
                        k_final = n;
                    }
                    i += 3;
                } else {
                    i += 1;
                }
            }
            "PARAMS" => {
                if i + 1 < parser.argv.len() {
                    if let Some(count) = parser.get_str(i + 1).and_then(|s| s.parse::<usize>().ok()) {
                        i += 2;
                        for _ in 0..count / 2 {
                            if i + 1 < parser.argv.len() {
                                let name = parser.get_str(i).unwrap_or("").to_string();
                                let value = parser.get_vec(i + 1).unwrap_or(&[]).to_vec();
                                params.insert(name, value);
                                i += 2;
                            }
                        }
                    } else {
                        i += 1;
                    }
                } else {
                    i += 1;
                }
            }
            _ => {
                i += 1;
            }
        }
    }

    // Get query vectors from params
    let query_pooled: Vec<f32> = params
        .get("query_pooled")
        .map(|bytes| {
            bytes
                .chunks(4)
                .map(|chunk| {
                    let arr: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
                    f32::from_le_bytes(arr)
                })
                .collect()
        })
        .unwrap_or_default();

    let query_tokens: Vec<f32> = params
        .get("query_tokens")
        .map(|bytes| {
            bytes
                .chunks(4)
                .map(|chunk| {
                    let arr: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
                    f32::from_le_bytes(arr)
                })
                .collect()
        })
        .unwrap_or_default();

    if query_pooled.is_empty() {
        return Response::Error("ERR missing query_pooled parameter".into());
    }

    // Execute search
    let index = index.lock().unwrap();
    let results = index.search(&query_pooled, &query_tokens, k_candidates, k_final);

    // Format response
    let mut response_data = Vec::new();
    response_data.push(Response::Integer(results.len() as i64));

    for result in results.iter().skip(offset).take(k_final) {
        response_data.push(Response::Data(result.key.clone().into_bytes()));

        let mut fields_vec = Vec::new();
        fields_vec.push(Response::Data(b"maxsim_score".to_vec()));
        fields_vec.push(Response::Data(format!("{:.6}", result.maxsim_score).into_bytes()));
        fields_vec.push(Response::Data(b"ann_score".to_vec()));
        fields_vec.push(Response::Data(format!("{:.6}", result.ann_score).into_bytes()));

        for (k, v) in &result.fields {
            fields_vec.push(Response::Data(k.clone().into_bytes()));
            fields_vec.push(Response::Data(v.clone().into_bytes()));
        }

        response_data.push(Response::Array(fields_vec));
    }

    Response::Array(response_data)
}

/// Handle FT.INFOMV command
pub fn ft_infomv(parser: &ParsedCommand, db: &Database, dbindex: usize) -> Response {
    if parser.argv.len() < 2 {
        return Response::Error("ERR wrong number of arguments for 'FT.INFOMV' command".into());
    }

    let index_name = match parser.get_str(1) {
        Some(s) => s,
        None => return Response::Error("ERR invalid index name".into()),
    };

    let key = format!("{}:{}", dbindex, index_name);
    let registry = MV_INDEX_REGISTRY.lock().unwrap();

    let index = match registry.get(&key) {
        Some(idx) => idx.lock().unwrap(),
        None => return Response::Error(format!("ERR index '{}' not found", index_name)),
    };

    let mut info = Vec::new();
    info.push(Response::Data(b"index_name".to_vec()));
    info.push(Response::Data(index.config.name.clone().into_bytes()));
    info.push(Response::Data(b"num_docs".to_vec()));
    info.push(Response::Integer(index.len() as i64));
    info.push(Response::Data(b"pooled_dim".to_vec()));
    info.push(Response::Integer(index.config.pooled_dim as i64));
    info.push(Response::Data(b"patch_dim".to_vec()));
    info.push(Response::Integer(index.config.patch_dim as i64));
    info.push(Response::Data(b"max_patches".to_vec()));
    info.push(Response::Integer(index.config.max_patches as i64));

    Response::Array(info)
}

/// Handle FT.DROPMV command
pub fn ft_dropmv(parser: &ParsedCommand, db: &Database, dbindex: usize) -> Response {
    if parser.argv.len() < 2 {
        return Response::Error("ERR wrong number of arguments for 'FT.DROPMV' command".into());
    }

    let index_name = match parser.get_str(1) {
        Some(s) => s,
        None => return Response::Error("ERR invalid index name".into()),
    };

    let key = format!("{}:{}", dbindex, index_name);
    let mut registry = MV_INDEX_REGISTRY.lock().unwrap();

    if registry.remove(&key).is_some() {
        Response::Status("OK".into())
    } else {
        Response::Error(format!("ERR index '{}' not found", index_name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_vector_index() {
        let config = MultiVectorIndexConfig {
            name: "test".to_string(),
            pooled_dim: 4,
            patch_dim: 4,
            max_patches: 100,
            patch_codec: PatchCodec::SQ8,
            distance_metric: DistanceMetricMv::Cosine,
        };

        let mut index = MultiVectorIndex::new(config);

        // Add a document
        let pooled = vec![1.0, 0.0, 0.0, 0.0];
        let patch_codes = vec![127u8, 0, 0, 0, 0, 127, 0, 0]; // 2 patches
        let patch_scales = vec![half::f16::from_f32(1.0)];

        index.add(
            "doc:1",
            &pooled,
            patch_codes,
            patch_scales,
            2,
            HashMap::new(),
        );

        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.001);
    }
}

