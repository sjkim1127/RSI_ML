# RSI-ML Benchmarks

This document defines how performance should be measured and compared in `rsi_ml`.

## Benchmark Commands

- Example benchmark:
  - `cargo run -p rsi_ml --example matmul_bench`
- Standardized Criterion benchmark:
  - `cargo bench -p rsi_ml`

## Environment

For comparable numbers:

- Use `--release` (Criterion uses release by default).
- Close heavy background processes.
- Keep power mode consistent.
- On Windows, prefer plugged-in "Best performance" mode.

Optional matmul controls:

- Fixed tile: `RSI_ML_MATMUL_TILE=64`
- One-time autotune: `RSI_ML_MATMUL_AUTOTUNE=1`

## Baseline (Initial)

These values came from `matmul_bench` on the current development environment and are used as an initial reference only.

- `128x128`: ~`1.30 GFLOPS`
- `256x256`: ~`1.50 GFLOPS`
- `384x384`: ~`1.56 GFLOPS`

Do not treat these as universal targets across machines. CI regression checks should use conservative thresholds.

## Regression Policy

- Track trend by comparing against the previous main-branch run.
- Soft threshold: `5%` slowdown warning.
- Hard threshold: `10%` slowdown failure for hot-path benchmarks.

## Covered Benchmarks

- `matmul`:
  - square sizes: `128`, `256`, `384`, `512`
- `attention_block`:
  - `(t, d) = (64, 128), (128, 128)`
- `softmax`:
  - `(rows, cols) = (256, 256), (512, 256), (512, 512)`
- `layer_norm`:
  - `(rows, cols) = (256, 256), (512, 256), (512, 512)`
- `scaled_dot_product_attention`:
  - `(t, d) = (64, 64), (128, 64), (128, 128)`

Planned additions:

- end-to-end `tiny_lm_train` step time
