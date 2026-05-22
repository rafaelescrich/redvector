use redisearch_platform_core::persistent_index::PersistentVectorIndex;
use redisearch_platform_core::vector_index::VectorMetric;
use std::time::Instant;
use rand::Rng;
use reqwest::blocking::get;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

// A small subset of GIST-960 or similar can be used here.
// For demonstration and to keep it completely self-contained without massive downloads,
// we will generate "clustered" synthetic data that mimics real-world embeddings.
// Pure uniform random data destroys PQ accuracy because there are no natural clusters.
// Real embeddings (like LLM outputs) have distinct clusters.

fn main() {
    println!("\n=== RedVector Recall@10 Benchmark ===\n");

    let dim = 128;
    let num_vectors = 10000;
    let num_queries = 100;
    let num_clusters = 64; // Simulate 64 semantic topics

    println!("1. Generating Clustered Synthetic Dataset ({} vectors, {}D)...", num_vectors, dim);
    let mut rng = rand::thread_rng();
    
    // Generate cluster centroids
    let mut centroids = Vec::with_capacity(num_clusters);
    for _ in 0..num_clusters {
        let mut c = vec![0.0; dim];
        for j in 0..dim {
            c[j] = rng.gen_range(-1.0..1.0);
        }
        centroids.push(c);
    }

    // Generate vectors around centroids
    let mut dataset = Vec::with_capacity(num_vectors);
    for _ in 0..num_vectors {
        let cluster_idx = rng.gen_range(0..num_clusters);
        let centroid = &centroids[cluster_idx];
        
        let mut v = vec![0.0; dim];
        for j in 0..dim {
            // Add gaussian noise around the centroid
            let noise: f32 = rng.gen_range(-0.2..0.2); 
            v[j] = centroid[j] + noise;
        }
        dataset.push(v);
    }

    // Generate queries from the same distribution
    let mut queries = Vec::with_capacity(num_queries);
    for _ in 0..num_queries {
        let cluster_idx = rng.gen_range(0..num_clusters);
        let centroid = &centroids[cluster_idx];
        
        let mut q = vec![0.0; dim];
        for j in 0..dim {
            let noise: f32 = rng.gen_range(-0.2..0.2); 
            q[j] = centroid[j] + noise;
        }
        queries.push(q);
    }

    // Baseline: Exact Brute-Force Search
    println!("2. Establishing Ground Truth (Exact Brute-Force Search)...");
    let mut index_exact = PersistentVectorIndex::new(
        "exact_bench".to_string(),
        dim,
        VectorMetric::Euclidean,
        None, // In-memory
        20000,
    ).unwrap();

    for (i, v) in dataset.iter().enumerate() {
        index_exact.add(i as u64, v.clone()).unwrap();
    }

    let mut ground_truth = Vec::with_capacity(num_queries);
    let start_exact = Instant::now();
    for q in &queries {
        let results = index_exact.search(q, 10).unwrap();
        let top_ids: Vec<u64> = results.into_iter().map(|(id, _)| id).collect();
        ground_truth.push(top_ids);
    }
    println!("   Exact Search Latency (Avg): {:?}", start_exact.elapsed() / num_queries as u32);

    // Test: Two-Stage PQ Search
    println!("\n3. Training Product Quantizer (PQ)...");
    let mut index_pq = PersistentVectorIndex::new(
        "pq_bench".to_string(),
        dim,
        VectorMetric::Euclidean,
        None, 
        20000,
    ).unwrap();

    for (i, v) in dataset.iter().enumerate() {
        index_pq.add(i as u64, v.clone()).unwrap();
    }

    // Train PQ: 16 subspaces, 256 clusters per subspace
    let train_start = Instant::now();
    index_pq.train_quantizer(16, 256, 5000).unwrap();
    println!("   PQ Training Latency: {:?}", train_start.elapsed());

    println!("4. Running Two-Stage Search (PQ Candidates + Disk Re-ranking)...");
    let mut pq_results = Vec::with_capacity(num_queries);
    let start_pq = Instant::now();
    for q in &queries {
        let results = index_pq.search(q, 10).unwrap();
        let top_ids: Vec<u64> = results.into_iter().map(|(id, _)| id).collect();
        pq_results.push(top_ids);
    }
    println!("   Two-Stage Latency (Avg): {:?}", start_pq.elapsed() / num_queries as u32);

    // Calculate Recall@10
    println!("\n5. Calculating Recall@10...");
    let mut total_hits = 0;
    let mut total_expected = 0;

    for (i, gt) in ground_truth.iter().enumerate() {
        let pq_res = &pq_results[i];
        
        let mut hits = 0;
        for id in pq_res {
            if gt.contains(id) {
                hits += 1;
            }
        }
        total_hits += hits;
        total_expected += gt.len();
    }

    let recall = (total_hits as f64 / total_expected as f64) * 100.0;
    println!("   Recall@10: {:.2}%", recall);
    
    if recall >= 90.0 {
        println!("   ✅ Success! Recall is above 90% production threshold.");
    } else {
        println!("   ❌ Warning: Recall is below 90%. Consider increasing PQ candidates before re-ranking.");
    }
}
