use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};
use redisearch_platform_core::persistent_index::PersistentVectorIndex;
use redisearch_platform_core::vector_index::VectorMetric;
use std::time::Instant;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    println!("\n=== RedVector + FastEmbed + Snowflake Arctic RAG Benchmark ===\n");

    // 1. Initialize FastEmbed with Snowflake Arctic model
    println!("1. Initializing FastEmbed with Snowflake Arctic (Downloading weights if needed)...");
    let init_start = Instant::now();
    let mut model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::SnowflakeArcticEmbedM)
            .with_show_download_progress(true)
    )?;
    println!("   Model loaded in: {:?}", init_start.elapsed());

    // 2. Prepare some real-world documents (e.g., from Wikipedia or a small corpus)
    let documents = vec![
        "Rust is a multi-paradigm, general-purpose programming language designed for performance and safety, especially safe concurrency.",
        "Redis is an in-memory data structure store, used as a distributed, in-memory key-value database, cache and message broker, with optional durability.",
        "Vector databases index and store vector embeddings for fast retrieval and similarity search, with capabilities like CRUD operations, metadata filtering, and horizontal scaling.",
        "Machine learning is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data.",
        "The quick brown fox jumps over the lazy dog.",
        "Product quantization is a technique used in approximate nearest neighbor search to compress vectors.",
        "Hierarchical Navigable Small World (HNSW) is a popular algorithm for approximate nearest neighbor search.",
        "FastEmbed is a fast, lightweight, and efficient library for local text embeddings.",
        "Snowflake Arctic is a series of open enterprise-grade models.",
        "Agentic RAG involves using AI agents to dynamically retrieve, analyze, and synthesize information for complex tasks."
    ];

    println!("\n2. Generating embeddings for {} documents...", documents.len());
    let embed_start = Instant::now();
    let embeddings = model.embed(documents.clone(), None)?;
    println!("   Generated {} embeddings in: {:?}", embeddings.len(), embed_start.elapsed());
    
    // Validate embedding dimension (Snowflake Arctic M is 768)
    let dim = embeddings[0].len();
    println!("   Embedding dimension: {}", dim);

    // 3. Initialize RedVector Persistent Index
    println!("\n3. Initializing RedVector Persistent Index...");
    let db_path = PathBuf::from("arctic_bench.redb");
    // Clean up old run if it exists
    if db_path.exists() {
        std::fs::remove_file(&db_path)?;
    }
    
    let mut index = PersistentVectorIndex::new(
        "rag_index".to_string(),
        dim,
        VectorMetric::Cosine,
        Some(&db_path),
        1000,
    )?;

    // 4. Ingest embeddings into RedVector
    println!("4. Ingesting embeddings into RedVector...");
    let ingest_start = Instant::now();
    for (i, emb) in embeddings.iter().enumerate() {
        index.add(i as u64, emb.clone())?;
    }
    println!("   Ingested {} vectors in: {:?}", embeddings.len(), ingest_start.elapsed());

    // 5. Perform semantic queries
    println!("\n5. Running Semantic Queries...");
    let queries = vec![
        "What is Rust known for?",
        "How do you compress vectors for search?",
        "What is an in-memory datastore?"
    ];

    for query in queries {
        println!("\n   Query: \"{}\"", query);
        let q_start = Instant::now();
        
        // Generate embedding for the query
        let query_emb = model.embed(vec![query], None)?.pop().unwrap();
        
        // Search RedVector
        let results = index.search(&query_emb, 3)?;
        
        println!("   Latency (Encode + Search): {:?}", q_start.elapsed());
        println!("   Top Results:");
        for (doc_id, score) in results {
            println!("     [Score: {:.4}] {}", score, documents[doc_id as usize]);
        }
    }

    // Cleanup
    drop(index);
    if db_path.exists() {
        std::fs::remove_file(db_path)?;
    }

    println!("\nBenchmark complete.");
    Ok(())
}
