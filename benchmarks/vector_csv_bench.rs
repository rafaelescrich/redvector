use parser::{parse_vector_csv_baseline, parse_vector_csv_fast};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    println!("\n=== RedVector Vector CSV Parser Benchmark ===\n");
    println!("Baseline: str::split(',') + trim + parse::<f32>()");
    println!("Fast:     memchr delimiter scan + ASCII trim + parse::<f32>()\n");

    for dim in [128usize, 384, 768, 1536] {
        let input = make_vector_csv(dim);
        let baseline = parse_vector_csv_baseline(&input).expect("baseline parser failed");
        let fast = parse_vector_csv_fast(&input).expect("fast parser failed");
        assert_eq!(baseline.len(), dim);
        assert_eq!(baseline, fast);

        let iterations = (20_000_000usize / dim).max(2_000);

        for _ in 0..100 {
            black_box(parse_vector_csv_baseline(black_box(&input)).unwrap());
            black_box(parse_vector_csv_fast(black_box(&input)).unwrap());
        }

        let start = Instant::now();
        let mut baseline_components = 0usize;
        for _ in 0..iterations {
            baseline_components += black_box(parse_vector_csv_baseline(black_box(&input)).unwrap()).len();
        }
        let baseline_duration = start.elapsed();

        let start = Instant::now();
        let mut fast_components = 0usize;
        for _ in 0..iterations {
            fast_components += black_box(parse_vector_csv_fast(black_box(&input)).unwrap()).len();
        }
        let fast_duration = start.elapsed();

        assert_eq!(baseline_components, fast_components);

        let baseline_vectors_per_sec = iterations as f64 / baseline_duration.as_secs_f64();
        let fast_vectors_per_sec = iterations as f64 / fast_duration.as_secs_f64();
        let speedup = baseline_duration.as_secs_f64() / fast_duration.as_secs_f64();

        println!("Dimension: {}", dim);
        println!(
            "  Baseline: {:?} ({:.0} vectors/sec)",
            baseline_duration, baseline_vectors_per_sec
        );
        println!(
            "  Fast:     {:?} ({:.0} vectors/sec)",
            fast_duration, fast_vectors_per_sec
        );
        println!("  Speedup:  {:.2}x", speedup);
        println!();
    }
}

fn make_vector_csv(dim: usize) -> String {
    (0..dim)
        .map(|i| {
            let value = ((i * 31 % 997) as f32 / 997.0) - 0.5;
            format!("{:.6}", value)
        })
        .collect::<Vec<_>>()
        .join(",")
}
