use std::time::Instant;
use rsi_ml::{matmul, Tensor};

fn build_tensor(rows: usize, cols: usize, scale: f32) -> Tensor {
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| ((i % 97) as f32 - 48.0) * scale)
        .collect();
    Tensor::from_loaded(data, vec![rows, cols], false).unwrap()
}

fn run_case(m: usize, k: usize, n: usize, iters: usize) {
    let a = build_tensor(m, k, 0.01);
    let b = build_tensor(k, n, 0.02);

    let warm = matmul(&a, &b).unwrap();
    let _ = warm.eval();

    let t0 = Instant::now();
    for _ in 0..iters {
        let y = matmul(&a, &b).unwrap();
        let _ = y.eval();
    }
    let elapsed = t0.elapsed();
    let ms = elapsed.as_secs_f64() * 1000.0;
    let gflops = (2.0 * m as f64 * k as f64 * n as f64 * iters as f64) / (elapsed.as_secs_f64() * 1e9);
    println!(
        "matmul {}x{} * {}x{} iters={} time_ms={:.2} throughput_GFLOPS={:.3}",
        m, k, k, n, iters, ms, gflops
    );
}

fn main() {
    run_case(128, 128, 128, 30);
    run_case(256, 256, 256, 12);
    run_case(384, 384, 384, 6);
}
