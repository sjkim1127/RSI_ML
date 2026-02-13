use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rsi_ml::{layer_norm, scaled_dot_product_attention, softmax, Tensor};

fn tensor(rows: usize, cols: usize, scale: f32) -> Tensor {
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| ((i % 89) as f32 - 44.0) * scale)
        .collect();
    Tensor::from_loaded(data, vec![rows, cols], false).unwrap()
}

fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");
    for (rows, cols) in [(256usize, 256usize), (512, 256), (512, 512)] {
        let x = tensor(rows, cols, 0.01);
        group.bench_with_input(
            BenchmarkId::new("rows_cols", format!("{rows}x{cols}")),
            &(rows, cols),
            |bench, _| {
                bench.iter(|| {
                    let y = softmax(&x, 1).unwrap();
                    let _ = y.eval();
                });
            },
        );
    }
    group.finish();
}

fn bench_layer_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("layer_norm");
    for (rows, cols) in [(256usize, 256usize), (512, 256), (512, 512)] {
        let x = tensor(rows, cols, 0.02);
        group.bench_with_input(
            BenchmarkId::new("rows_cols", format!("{rows}x{cols}")),
            &(rows, cols),
            |bench, _| {
                bench.iter(|| {
                    let y = layer_norm(&x, None, None, 1e-5).unwrap();
                    let _ = y.eval();
                });
            },
        );
    }
    group.finish();
}

fn bench_scaled_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaled_dot_product_attention");
    for (t, d) in [(64usize, 64usize), (128, 64), (128, 128)] {
        let q = tensor(t, d, 0.01);
        let k = tensor(t, d, 0.01);
        let v = tensor(t, d, 0.01);
        group.bench_with_input(BenchmarkId::new("t_d", format!("{t}x{d}")), &(t, d), |bench, _| {
            bench.iter(|| {
                let y = scaled_dot_product_attention(&q, &k, &v, true).unwrap();
                let _ = y.eval();
            });
        });
    }
    group.finish();
}

criterion_group!(ops_benches, bench_softmax, bench_layer_norm, bench_scaled_attention);
criterion_main!(ops_benches);
