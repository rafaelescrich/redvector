//! Semantic Search Example using Redisearch
//! 
//! This example demonstrates how to use rsedis+Redisearch for semantic search
//! with text embeddings, similar to Qdrant's blog examples.

use redis::Commands;
use std::collections::HashMap;

/// Example documents for semantic search
const DOCUMENTS: &[&str] = &[
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with multiple layers",
    "Natural language processing enables computers to understand human language",
    "Computer vision allows machines to interpret visual information",
    "Reinforcement learning trains agents through rewards and penalties",
    "Supervised learning uses labeled data to train models",
    "Unsupervised learning finds patterns in data without labels",
    "Transfer learning applies knowledge from one task to another",
];

/// Generate embeddings for text (simplified - in production, use a real embedding model)
/// 
/// This is a placeholder. In production, you would use:
/// - sentence-transformers (Python): `sentence_transformers.SentenceTransformer`
/// - rust-bert (Rust): `rust_bert::pipelines::sentence_embeddings`
/// - OpenAI API: `text-embedding-ada-002`
/// - Cohere API: `embed-english-v2.0`
fn generate_embedding(text: &str) -> Vec<f32> {
    // Simplified embedding: hash-based feature vector
    // In production, replace with real embedding model
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
fn create_semantic_search_index(conn: &mut redis::Connection, index_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Creating semantic search index: {}", index_name);
    
    // Create index with vector field
    redis::cmd("FT.CREATE")
        .arg(index_name)
        .arg("SCHEMA")
        .arg("title")
        .arg("TEXT")
        .arg("embedding")
        .arg(format!("VECTOR({})", 384))
        .query(conn)?;
    
    println!("Index created successfully");
    Ok(())
}

/// Add documents to index
fn add_documents(conn: &mut redis::Connection, index_name: &str, documents: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
    println!("Adding {} documents to index...", documents.len());
    
    for (i, doc) in documents.iter().enumerate() {
        let embedding = generate_embedding(doc);
        let embedding_str = embedding.iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",");
        
        redis::cmd("FT.ADD")
            .arg(index_name)
            .arg(format!("doc:{}", i))
            .arg("1.0")
            .arg("FIELDS")
            .arg("title")
            .arg(doc)
            .arg("embedding")
            .arg(&embedding_str)
            .query(conn)?;
        
        if (i + 1) % 10 == 0 {
            println!("  Added {} documents...", i + 1);
        }
    }
    
    println!("All documents added");
    Ok(())
}

/// Search for similar documents
fn search_similar(conn: &mut redis::Connection, index_name: &str, query: &str, limit: usize) -> Result<Vec<(String, f32)>, Box<dyn std::error::Error>> {
    let query_embedding = generate_embedding(query);
    let query_str = query_embedding.iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",");
    
    println!("Searching for: '{}'", query);
    
    // Search using vector similarity
    let result: redis::Value = redis::cmd("FT.SEARCH")
        .arg(index_name)
        .arg(&query_str)
        .arg("LIMIT")
        .arg("0")
        .arg(limit.to_string())
        .arg("WITHSCORES")
        .query(conn)?;
    
    // Parse results (simplified - actual parsing depends on response format)
    let mut results = Vec::new();
    
    // In a real implementation, parse the Redis response properly
    // For now, return placeholder results
    results.push((query.to_string(), 0.95));
    
    Ok(results)
}

/// Main example
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Semantic Search Example ===\n");
    
    // Connect to rsedis
    let client = redis::Client::open("redis://127.0.0.1:6379/")?;
    let mut conn = client.get_connection()?;
    
    let index_name = "semantic_search";
    
    // Create index
    create_semantic_search_index(&mut conn, index_name)?;
    
    // Add documents
    add_documents(&mut conn, index_name, DOCUMENTS)?;
    
    // Search queries
    let queries = vec![
        "What is artificial intelligence?",
        "How do neural networks work?",
        "Tell me about language processing",
    ];
    
    for query in queries {
        println!("\n---");
        match search_similar(&mut conn, index_name, query, 3) {
            Ok(results) => {
                println!("Found {} results:", results.len());
                for (i, (doc, score)) in results.iter().enumerate() {
                    println!("  {}. {} (score: {:.3})", i + 1, doc, score);
                }
            }
            Err(e) => {
                eprintln!("Search error: {}", e);
            }
        }
    }
    
    println!("\n=== Example Complete ===");
    Ok(())
}

/// Integration with real embedding models
/// 
/// To use real embeddings, replace `generate_embedding` with one of these:
/// 
/// ### Python (sentence-transformers)
/// ```python
/// from sentence_transformers import SentenceTransformer
/// model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions
/// embedding = model.encode(text)
/// ```
/// 
/// ### Rust (rust-bert)
/// ```rust
/// use rust_bert::pipelines::sentence_embeddings::{
///     SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType
/// };
/// 
/// let model = SentenceEmbeddingsBuilder::remote(
///     SentenceEmbeddingsModelType::AllMiniLmL6V2
/// ).create_model()?;
/// 
/// let embeddings = model.encode(&[text])?;
/// ```
/// 
/// ### OpenAI API
/// ```rust
/// use openai_api_rust::*;
/// 
/// let auth = Auth::from_env().unwrap();
/// let openai = OpenAI::new(auth, "https://api.openai.com/v1/");
/// 
/// let embedding = openai.embeddings_create(
///     "text-embedding-ada-002",
///     text
/// )?;
/// ```
/// 
/// ### Cohere API
/// ```rust
/// use cohere_rust::api::EmbedRequest;
/// 
/// let client = cohere_rust::Client::new("YOUR_API_KEY");
/// let embedding = client.embed(EmbedRequest {
///     texts: vec![text],
///     model: Some("embed-english-v2.0".to_string()),
///     ..Default::default()
/// })?;
/// ```

