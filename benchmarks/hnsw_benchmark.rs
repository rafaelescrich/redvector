//! HNSW Performance Benchmark
//! 
//! This benchmark tests the HNSW vector index performance with various
//! dataset sizes and configurations.

use std::time::{Duration, Instant};
use std::sync::Arc;
use std::sync::Mutex;

#[cfg(feature = "hnsw-backend")]
use database::vector_index::{HnswVectorIndex, VectorMetric, VectorIndexTrait};

/// Benchmark configuration
#[derive(Clone)]
pub struct BenchmarkConfig {
    pub dataset_size: usize,
    pub vector_dimension: usize,
    pub search_queries: usize,
    pub k: usize, // Number of results to return
    pub m: usize, // HNSW M parameter
    pub ef_construction: usize,
    pub ef_search: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            dataset_size: 10000,
            vector_dimension: 384,
            search_queries: 1000,
            k: 10,
            m: 16,
            ef_construction: 200,
            ef_search: 50,
        }
    }
}

/// Benchmark results
#[derive(Debug, Clone)]
pub struct HnswBenchmarkResults {
    pub config: BenchmarkConfig,
    pub insertion_time: Duration,
    pub insertion_throughput: f64, // vectors per second
    pub search_times: Vec<Duration>,
    pub search_latency_p50: Duration,
    pub search_latency_p95: Duration,
    pub search_latency_p99: Duration,
    pub search_qps: f64,
    pub memory_estimate_mb: f64,
}

impl HnswBenchmarkResults {
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            insertion_time: Duration::ZERO,
            insertion_throughput: 0.0,
            search_times: Vec::new(),
            search_latency_p50: Duration::ZERO,
            search_latency_p95: Duration::ZERO,
            search_latency_p99: Duration::ZERO,
            search_qps: 0.0,
            memory_estimate_mb: 0.0,
        }
    }
    
    pub fn calculate_percentiles(&mut self) {
        if self.search_times.is_empty() {
            return;
        }
        
        let mut sorted = self.search_times.clone();
        sorted.sort();
        
        let len = sorted.len();
        self.search_latency_p50 = sorted[len * 50 / 100];
        self.search_latency_p95 = sorted[len * 95 / 100];
        self.search_latency_p99 = sorted[len * 99 / 100];
        
        // Calculate QPS (queries per second)
        let total_time: Duration = self.search_times.iter().sum();
        if total_time > Duration::ZERO {
            self.search_qps = (self.config.search_queries as f64) / total_time.as_secs_f64();
        }
    }
    
    pub fn print_summary(&self) {
        println!("\n=== HNSW Benchmark Results ===");
        println!("Configuration:");
        println!("  Dataset Size: {} vectors", self.config.dataset_size);
        println!("  Vector Dimension: {}", self.config.vector_dimension);
        println!("  HNSW Parameters: m={}, ef_construction={}, ef_search={}", 
                 self.config.m, self.config.ef_construction, self.config.ef_search);
        println!("\nInsertion Performance:");
        println!("  Time: {:.2}s", self.insertion_time.as_secs_f64());
        println!("  Throughput: {:.2} vectors/sec", self.insertion_throughput);
        println!("\nSearch Performance:");
        println!("  Queries: {}", self.config.search_queries);
        println!("  QPS: {:.2}", self.search_qps);
        println!("  Latency P50: {:.2}ms", self.search_latency_p50.as_secs_f64() * 1000.0);
        println!("  Latency P95: {:.2}ms", self.search_latency_p95.as_secs_f64() * 1000.0);
        println!("  Latency P99: {:.2}ms", self.search_latency_p99.as_secs_f64() * 1000.0);
        println!("\nMemory:");
        println!("  Estimated: {:.2} MB", self.memory_estimate_mb);
    }
}

/// Generate random normalized vectors
pub fn generate_random_vectors(count: usize, dimension: usize, seed: u64) -> Vec<Vec<f32>> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    (0..count)
        .map(|i| {
            let mut hasher = DefaultHasher::new();
            (seed + i as u64).hash(&mut hasher);
            let mut rng = hasher.finish();
            
            let mut v = Vec::with_capacity(dimension);
            let mut norm_sq = 0.0;
            
            for _ in 0..dimension {
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                let val = ((rng % 2000) as f32 / 1000.0) - 1.0;
                norm_sq += val * val;
                v.push(val);
            }
            
            // Normalize
            let norm = norm_sq.sqrt();
            if norm > 0.0 {
                for val in &mut v {
                    *val /= norm;
                }
            }
            
            v
        })
        .collect()
}

/// Run HNSW benchmark
#[cfg(feature = "hnsw-backend")]
pub fn run_benchmark(config: BenchmarkConfig) -> HnswBenchmarkResults {
    let mut results = HnswBenchmarkResults::new(config.clone());
    
    println!("Generating {} vectors of dimension {}...", 
             config.dataset_size, config.vector_dimension);
    let vectors = generate_random_vectors(config.dataset_size, config.vector_dimension, 42);
    
    println!("Creating HNSW index...");
    let mut index = HnswVectorIndex::new(
        config.vector_dimension,
        VectorMetric::Cosine,
        Some(config.m),
        Some(config.ef_construction),
    );
    
    // Benchmark insertion
    println!("Inserting vectors...");
    let start = Instant::now();
    for (i, vector) in vectors.iter().enumerate() {
        index.add(i as u64, vector.clone()).expect("Failed to add vector");
    }
    results.insertion_time = start.elapsed();
    results.insertion_throughput = (config.dataset_size as f64) / results.insertion_time.as_secs_f64();
    
    println!("Inserted {} vectors in {:.2}s ({:.2} vectors/sec)",
             config.dataset_size,
             results.insertion_time.as_secs_f64(),
             results.insertion_throughput);
    
    // Generate query vectors
    println!("Generating {} query vectors...", config.search_queries);
    let query_vectors = generate_random_vectors(config.search_queries, config.vector_dimension, 9999);
    
    // Benchmark search
    println!("Running search benchmarks...");
    results.search_times = Vec::with_capacity(config.search_queries);
    
    for query in query_vectors {
        let start = Instant::now();
        let search_results = index.search(&query, config.k, Some(config.ef_search))
            .expect("Search failed");
        let elapsed = start.elapsed();
        
        results.search_times.push(elapsed);
        
        // Verify we got results
        assert_eq!(search_results.len(), config.k.min(config.dataset_size));
    }
    
    // Calculate percentiles and QPS
    results.calculate_percentiles();
    
    // Estimate memory (rough calculation)
    // HNSW typically uses: dimension * 4 bytes per vector + graph overhead
    // Graph overhead: approximately m * 8 bytes per node
    let vector_memory = config.dataset_size * config.vector_dimension * 4; // f32 = 4 bytes
    let graph_overhead = config.dataset_size * config.m * 8; // Rough estimate
    results.memory_estimate_mb = (vector_memory + graph_overhead) as f64 / (1024.0 * 1024.0);
    
    results
}

/// Compare different HNSW configurations
#[cfg(feature = "hnsw-backend")]
pub fn compare_configurations() {
    let base_config = BenchmarkConfig {
        dataset_size: 10000,
        vector_dimension: 384,
        search_queries: 1000,
        k: 10,
        m: 16,
        ef_construction: 200,
        ef_search: 50,
    };
    
    println!("=== Comparing HNSW Configurations ===\n");
    
    // Test different M values
    println!("Testing different M values (ef_construction=200, ef_search=50)...");
    for m in [8, 16, 32] {
        let mut config = base_config.clone();
        config.m = m;
        let results = run_benchmark(config);
        println!("\nM={}:", m);
        println!("  Insertion: {:.2} vectors/sec", results.insertion_throughput);
        println!("  Search QPS: {:.2}", results.search_qps);
        println!("  P95 Latency: {:.2}ms", results.search_latency_p95.as_secs_f64() * 1000.0);
        println!("  Memory: {:.2} MB", results.memory_estimate_mb);
    }
    
    // Test different ef_search values
    println!("\n\nTesting different ef_search values (m=16, ef_construction=200)...");
    for ef_search in [20, 50, 100, 200] {
        let mut config = base_config.clone();
        config.ef_search = ef_search;
        let results = run_benchmark(config);
        println!("\nef_search={}:", ef_search);
        println!("  Search QPS: {:.2}", results.search_qps);
        println!("  P95 Latency: {:.2}ms", results.search_latency_p95.as_secs_f64() * 1000.0);
    }
}

/// Benchmark different dataset sizes
#[cfg(feature = "hnsw-backend")]
pub fn benchmark_scalability() {
    println!("=== HNSW Scalability Benchmark ===\n");
    
    let sizes = [1000, 10000, 100000];
    
    for size in sizes.iter() {
        let config = BenchmarkConfig {
            dataset_size: *size,
            vector_dimension: 384,
            search_queries: 1000.min(*size / 10), // Scale queries with dataset
            k: 10,
            m: 16,
            ef_construction: 200,
            ef_search: 50,
        };
        
        println!("\nDataset size: {}", size);
        let results = run_benchmark(config);
        results.print_summary();
    }
}

#[cfg(feature = "hnsw-backend")]
fn main() {
    println!("HNSW Performance Benchmark");
    println!("===========================\n");
    
    // Run default benchmark
    let config = BenchmarkConfig::default();
    let results = run_benchmark(config);
    results.print_summary();
    
    // Uncomment to run comparison benchmarks (takes longer)
    // compare_configurations();
    // benchmark_scalability();
}

#[cfg(not(feature = "hnsw-backend"))]
fn main() {
    println!("HNSW backend not enabled. Build with --features hnsw-backend");
}

