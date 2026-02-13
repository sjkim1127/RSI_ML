use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rsi_ml::{attention_block, matmul, Tensor};

fn tensor(rows: usize, cols: usize, scale: f32) -> Tensor {
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| ((i % 97) as f32 - 48.0) * scale)
        .collect();
    Tensor::from_loaded(data, vec![rows, cols], false).unwrap()
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    for size in [128usize, 256, 384, 512] {
        let a = tensor(size, size, 0.01);
        let b = tensor(size, size, 0.02);
        group.throughput(Throughput::Elements((size * size * size) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bench, _| {
            bench.iter(|| {
                let y = matmul(&a, &b).unwrap();
                let _ = y.eval();
            });
        });
    }
    group.finish();
}

fn bench_attention_block(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_block");
    for (t, d) in [(64usize, 128usize), (128, 128)] {
        let x = tensor(t, d, 0.01);
        let ident = Tensor::from_loaded(
            {
                let mut v = vec![0.0; d * d];
                for i in 0..d {
                    v[i * d + i] = 1.0;
                }
                v
            },
            vec![d, d],
            false,
        )
        .unwrap();
        group.bench_with_input(BenchmarkId::new("t_d", format!("{t}x{d}")), &(t, d), |bench, _| {
            bench.iter(|| {
                let y = attention_block(&x, &ident, &ident, &ident, &ident, 8, 1e-5).unwrap();
                let _ = y.eval();
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_matmul, bench_attention_block);
criterion_main!(benches);
