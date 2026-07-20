use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};
use std::time::Instant;
use reqwest::blocking::Client;
use scraper::{Html, Selector};
use serde_json::Value;
use redis::Commands;

const CIK_NVDA: &str = "0001045810";
const USER_AGENT: &str = "UnblockTheChain AgenticRAG admin@unblockthechain.com";

fn main() -> anyhow::Result<()> {
    println!("\n=== RedVector + Snowflake Arctic: SEC EDGAR 2025/2026 Networked RAG ===\n");

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
    
    let document = Html::parse_document(&doc_html);
    let p_selector = Selector::parse("p, span").unwrap();
    
    let mut raw_text = String::new();
    for element in document.select(&p_selector) {
        let text: String = element.text().collect::<Vec<_>>().join(" ");
        let cleaned = text.trim();
        if cleaned.len() > 50 {
            raw_text.push_str(cleaned);
            raw_text.push_str(" ");
        }
    }
    
    let words: Vec<&str> = raw_text.split_whitespace().collect();
    let chunk_size = 150;
    let mut chunks = Vec::new();
    
    for chunk in words.chunks(chunk_size) {
        let chunk_text = chunk.join(" ");
        if chunk_text.len() > 100 {
            chunks.push(chunk_text);
        }
    }
    
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
    let embeddings = model.embed(chunks_vec.clone(), None)?;
    println!("   Generated {} embeddings in: {:?}", embeddings.len(), embed_start.elapsed());
    
    let dim = embeddings[0].len();
    
    println!("\n5. Connecting to Running RedVector Server (redis://127.0.0.1:6379)...");
    let redis_client = redis::Client::open("redis://127.0.0.1:6379")?;
    let mut con = redis_client.get_connection()?;

    // Drop index if exists, ignoring errors
    let _: redis::RedisResult<()> = redis::cmd("FT.DROP").arg("edgar_idx").query(&mut con);

    println!("6. Creating index 'edgar_idx' via FT.CREATE...");
    redis::cmd("FT.CREATE")
        .arg("edgar_idx")
        .arg("SCHEMA")
        .arg("vec")
        .arg("VECTOR")
        .arg(&dim.to_string())
        .query::<redis::Value>(&mut con)?;

    println!("7. Ingesting embeddings into RedVector via FT.ADD over network...");
    let ingest_start = Instant::now();
    for (i, emb) in embeddings.iter().enumerate() {
        let vec_str = emb.iter().map(|f| f.to_string()).collect::<Vec<String>>().join(",");
        redis::cmd("FT.ADD")
            .arg("edgar_idx")
            .arg(format!("doc:{}", i))
            .arg("1.0")
            .arg("FIELDS")
            .arg("vector")
            .arg(&vec_str)
            .query::<redis::Value>(&mut con)?;
    }
    println!("   Network Ingestion Latency: {:?}", ingest_start.elapsed());

    println!("\n8. Running TOUGH Networked RAG Semantic Queries against NVDA 10-Q...");
    let queries = vec![
        "Explain the $4.5 billion charge related to H20 inventory and its exact impact on the YoY gross margin comparison.",
        "What specific percentage of total revenue comes from customers outside the U.S., and how does the company define geographic revenue designation?",
        "Detail the risks mentioned regarding antitrust investigations and regulators' interest in the AI business worldwide.",
        "What are the specific inventory purchase obligations or commitments mentioned that could affect future liquidity or supply?"
    ];

    for query in queries {
        println!("\n   Query: \"{}\"", query);
        let q_start = Instant::now();
        
        let query_emb = model.embed(vec![query.to_string()], None)?.pop().unwrap();
        let query_str = query_emb.iter().map(|f| f.to_string()).collect::<Vec<String>>().join(",");
        
        // FT.SEARCH edgar_idx {query_str}
        let redis_results: redis::Value = redis::cmd("FT.SEARCH")
            .arg("edgar_idx")
            .arg(&query_str)
            .query(&mut con)?;
        
        println!("   Latency (Encode + Network Search): {:?}", q_start.elapsed());
        
        // Parse the results
        if let redis::Value::Bulk(items) = redis_results {
            if items.len() > 1 {
                for i in 1..items.len() {
                    if let redis::Value::Bulk(ref doc_info) = items[i] {
                        if doc_info.len() >= 3 {
                            let doc_id_str = match &doc_info[0] {
                                redis::Value::Data(bytes) => String::from_utf8_lossy(bytes).into_owned(),
                                _ => continue,
                            };
                            let score_str = match &doc_info[2] {
                                redis::Value::Data(bytes) => String::from_utf8_lossy(bytes).into_owned(),
                                _ => continue,
                            };
                            
                            // RedVector returns the hashed/internal doc_id by default unless we keep a payload map.
                            // We will print the internal ID and the score.
                            println!("     [Internal ID: {}] Score: {}", doc_id_str, score_str);
                            
                            // If it happens to be formatted as "doc:X", we can show the snippet
                            if doc_id_str.starts_with("doc:") {
                                if let Ok(doc_id) = doc_id_str.replace("doc:", "").parse::<usize>() {
                                    if doc_id < chunks_vec.len() {
                                        let snippet = &chunks_vec[doc_id];
                                        let display_text = if snippet.len() > 300 {
                                            format!("{}...", &snippet[0..300])
                                        } else {
                                            snippet.to_string()
                                        };
                                        let display_text = display_text.replace('\n', " ");
                                        println!("     -> {}", display_text);
                                    }
                                }
                            } else if let Ok(hash_id) = doc_id_str.parse::<u64>() {
                                // RedVector currently uses internal numeric hash IDs. 
                                // Since we didn't store the payloads in the basic FT.ADD test, 
                                // we can just log that a hit was found.
                                // In a full production app, the REDIS HASH map would hold the actual text.
                            }
                        }
                    }
                }
            } else {
                println!("     No results found.");
            }
        }
    }

    println!("\nPipeline complete. The data is now fully indexed and persistent in your running RedVector server!");
    Ok(())
}
