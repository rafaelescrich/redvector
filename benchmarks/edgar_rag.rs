use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};
use redisearch_platform_core::persistent_index::PersistentVectorIndex;
use redisearch_platform_core::vector_index::VectorMetric;
use std::time::Instant;
use std::path::PathBuf;
use reqwest::blocking::Client;
use scraper::{Html, Selector};
use serde_json::Value;

const CIK_NVDA: &str = "0001045810";
const USER_AGENT: &str = "UnblockTheChain AgenticRAG admin@unblockthechain.com";

fn main() -> anyhow::Result<()> {
    println!("\n=== RedVector + Snowflake Arctic: SEC EDGAR 2025/2026 RAG ===\n");

    let client = Client::builder()
        .user_agent(USER_AGENT)
        .build()?;

    println!("1. Fetching latest SEC filings metadata for NVDA (CIK: {})...", CIK_NVDA);
    let url = format!("https://data.sec.gov/submissions/CIK{}.json", CIK_NVDA);
    let resp: Value = client.get(&url).send()?.json()?;

    let recent = &resp["filings"]["recent"];
    let forms = recent["form"].as_array().unwrap();
    let accessions = recent["accessionNumber"].as_array().unwrap();
    let primary_docs = recent["primaryDocument"].as_array().unwrap();
    let filing_dates = recent["filingDate"].as_array().unwrap();

    let mut target_url = String::new();
    let mut target_date = String::new();
    let mut target_form = String::new();

    // Look for 2026 or 2025 10-K or 10-Q
    for i in 0..forms.len() {
        let form = forms[i].as_str().unwrap();
        let date = filing_dates[i].as_str().unwrap();
        
        if (form == "10-K" || form == "10-Q") && (date.starts_with("2026") || date.starts_with("2025")) {
            let acc_full = accessions[i].as_str().unwrap();
            let acc_no_dash = acc_full.replace("-", "");
            let p_doc = primary_docs[i].as_str().unwrap();
            
            target_url = format!("https://www.sec.gov/Archives/edgar/data/{}/{}/{}", 
                CIK_NVDA.parse::<u64>().unwrap(), acc_no_dash, p_doc);
            target_date = date.to_string();
            target_form = form.to_string();
            break;
        }
    }

    // Fallback if none found for 25/26 (just in case)
    if target_url.is_empty() {
        println!("   No 2025/2026 10-K/10-Q found. Falling back to the most recent...");
        for i in 0..forms.len() {
            let form = forms[i].as_str().unwrap();
            if form == "10-K" || form == "10-Q" {
                let date = filing_dates[i].as_str().unwrap();
                let acc_full = accessions[i].as_str().unwrap();
                let acc_no_dash = acc_full.replace("-", "");
                let p_doc = primary_docs[i].as_str().unwrap();
                
                target_url = format!("https://www.sec.gov/Archives/edgar/data/{}/{}/{}", 
                    CIK_NVDA.parse::<u64>().unwrap(), acc_no_dash, p_doc);
                target_date = date.to_string();
                target_form = form.to_string();
                break;
            }
        }
    }

    if target_url.is_empty() {
        anyhow::bail!("Could not find any 10-K or 10-Q filings.");
    }

    println!("   Found {} filed on {}.", target_form, target_date);
    println!("   URL: {}", target_url);

    println!("\n2. Downloading and Parsing HTML Document...");
    let doc_html = client.get(&target_url).send()?.text()?;
    
    // Parse HTML to extract text
    let document = Html::parse_document(&doc_html);
    // Simple selector to grab paragraphs and span text (to avoid massive tables if possible)
    let p_selector = Selector::parse("p, span").unwrap();
    
    let mut raw_text = String::new();
    for element in document.select(&p_selector) {
        let text: String = element.text().collect::<Vec<_>>().join(" ");
        let cleaned = text.trim();
        if cleaned.len() > 50 { // Skip short noise
            raw_text.push_str(cleaned);
            raw_text.push_str(" ");
        }
    }
    
    // Chunking text (simple naive chunking by words)
    let words: Vec<&str> = raw_text.split_whitespace().collect();
    let chunk_size = 150; // ~150 words per chunk for better semantic capture
    let mut chunks = Vec::new();
    
    for chunk in words.chunks(chunk_size) {
        let chunk_text = chunk.join(" ");
        // Ensure chunk is meaningful
        if chunk_text.len() > 100 {
            chunks.push(chunk_text);
        }
    }
    
    // Cap to 500 chunks to ensure the test runs quickly (approx 75,000 words)
    let chunks = if chunks.len() > 500 { &chunks[0..500] } else { &chunks[..] };
    let chunks_vec = chunks.to_vec();

    println!("   Extracted {} meaningful text chunks.", chunks_vec.len());

    println!("\n3. Initializing FastEmbed with Snowflake Arctic (ONNX Runtime)...");
    let init_start = Instant::now();
    let mut model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::SnowflakeArcticEmbedM)
    )?;
    println!("   Model loaded in: {:?}", init_start.elapsed());

    println!("\n4. Generating embeddings for chunks...");
    let embed_start = Instant::now();
    // fastembed handles parallel batching natively using Rayon
    let embeddings = model.embed(chunks_vec.clone(), None)?;
    println!("   Generated {} embeddings in: {:?}", embeddings.len(), embed_start.elapsed());
    
    let dim = embeddings[0].len();
    
    println!("\n5. Initializing RedVector Index...");
    let db_path = PathBuf::from("edgar_bench.redb");
    if db_path.exists() {
        std::fs::remove_file(&db_path)?;
    }
    
    let mut index = PersistentVectorIndex::new(
        "edgar_idx".to_string(),
        dim,
        VectorMetric::Cosine,
        Some(&db_path),
        1000,
    )?;

    println!("6. Ingesting embeddings into RedVector (Pure Rust & SIMD)...");
    for (i, emb) in embeddings.iter().enumerate() {
        index.add(i as u64, emb.clone())?;
    }

    println!("\n7. Running RAG Semantic Queries against NVDA {}...", target_form);
    let queries = vec![
        "What are the primary risk factors related to competition?",
        "How is artificial intelligence (AI) and data center demand impacting revenue?",
        "What is the total revenue or gross margin reported for the year?",
        "Who are the major competitors in the GPU market or AI computing?"
    ];

    for query in queries {
        println!("\n   Query: \"{}\"", query);
        let q_start = Instant::now();
        
        let query_emb = model.embed(vec![query.to_string()], None)?.pop().unwrap();
        let results = index.search(&query_emb, 3)?;
        
        println!("   Latency (Encode + Search): {:?}", q_start.elapsed());
        for (doc_id, score) in results {
            let snippet = &chunks_vec[doc_id as usize];
            // Format snippet to avoid blowing up the console
            let display_text = if snippet.len() > 300 {
                format!("{}...", &snippet[0..300])
            } else {
                snippet.to_string()
            };
            // Replace newlines to keep console clean
            let display_text = display_text.replace('\n', " ");
            println!("     [Score: {:.4}] {}", score, display_text);
        }
    }

    drop(index);
    if db_path.exists() {
        std::fs::remove_file(db_path)?;
    }

    println!("\nPipeline complete.");
    Ok(())
}
