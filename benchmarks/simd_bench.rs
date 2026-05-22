use redisearch_platform_core::simd_metrics::*;
use std::time::Instant;
use rand::Rng;

fn main() {
    println!("\n=== RedVector SIMD Performance Benchmark ===\n");

    let dimensions = vec![128, 384, 768, 1536];
    let iterations = 1_000_000;

    for dim in dimensions {
        println!("Testing Dimension: {}", dim);
        
        let mut rng = rand::thread_rng();
        let a: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
        let b: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();

        // Benchmark Cosine Similarity
        
        // Scalar
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = cosine_similarity_scalar_fallback(&a, &b);
        }
        let scalar_duration = start.elapsed();
        
        // SIMD
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = cosine_similarity_simd(&a, &b);
        }
        let simd_duration = start.elapsed();
        
        let speedup = scalar_duration.as_secs_f64() / simd_duration.as_secs_f64();
        
        println!("  Cosine Similarity:");
        println!("    Scalar: {:?} ({} ops/sec)", scalar_duration, (iterations as f64 / scalar_duration.as_secs_f64()) as u64);
        println!("    SIMD:   {:?} ({} ops/sec)", simd_duration, (iterations as f64 / simd_duration.as_secs_f64()) as u64);
        println!("    Speedup: {:.2}x", speedup);

        // Benchmark Euclidean Distance
        
        // Scalar
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = euclidean_distance_scalar_fallback(&a, &b);
        }
        let scalar_duration = start.elapsed();
        
        // SIMD
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = euclidean_distance_simd(&a, &b);
        }
        let simd_duration = start.elapsed();
        
        let speedup = scalar_duration.as_secs_f64() / simd_duration.as_secs_f64();
        
        println!("  Euclidean Distance:");
        println!("    Scalar: {:?} ({} ops/sec)", scalar_duration, (iterations as f64 / scalar_duration.as_secs_f64()) as u64);
        println!("    SIMD:   {:?} ({} ops/sec)", simd_duration, (iterations as f64 / simd_duration.as_secs_f64()) as u64);
        println!("    Speedup: {:.2}x", speedup);

        println!();
    }
}
