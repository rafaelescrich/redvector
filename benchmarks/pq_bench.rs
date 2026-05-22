use redisearch_platform_core::persistent_index::PersistentVectorIndex;
use redisearch_platform_core::vector_index::VectorMetric;
use std::time::Instant;
use rand::Rng;

fn main() {
    println!("\n=== RedVector PQ Scale Benchmark ===\n");

    let dim = 768;
    let num_vectors = 1000;
    let query_count = 10;
    
    let mut rng = rand::thread_rng();
    
    println!("1. Creating Index and Ingesting {} vectors...", num_vectors);
    let mut index = PersistentVectorIndex::new(
        "scale_bench".to_string(),
        dim,
        VectorMetric::Cosine,
        None, // In-memory redb
        1000,
    ).unwrap();
    
    let mut all_vectors = Vec::with_capacity(num_vectors);
    for i in 0..num_vectors {
        let mut v = vec![0.0; dim];
        for j in 0..dim {
            v[j] = rng.gen();
        }
        index.add(i as u64, v.clone()).unwrap();
        all_vectors.push(v);
    }
    
    // Baseline Search
    println!("2. Benchmarking Baseline Search (Full-Precision HNSW)...");
    let mut total_duration = std::time::Duration::default();
    for _ in 0..query_count {
        let q = &all_vectors[rng.gen_range(0..num_vectors)];
        let start = Instant::now();
        let _ = index.search(q, 10).unwrap();
        total_duration += start.elapsed();
    }
    println!("   Avg Latency: {:?}", total_duration / query_count);
    
    // Train PQ
    println!("3. Training Product Quantizer (PQ)...");
    let start = Instant::now();
    index.train_quantizer(32, 256, 500).unwrap();
    println!("   Training took: {:?}", start.elapsed());
    
    // Two-Stage Search
    println!("4. Benchmarking Two-Stage Search (PQ Candidates + Disk Re-ranking)...");
    let mut total_duration = std::time::Duration::default();
    for _ in 0..query_count {
        let q = &all_vectors[rng.gen_range(0..num_vectors)];
        let start = Instant::now();
        let _ = index.search(q, 10).unwrap();
        total_duration += start.elapsed();
    }
    println!("   Avg Latency: {:?}", total_duration / query_count);
    
    println!("\nBenchmark complete.");
}
