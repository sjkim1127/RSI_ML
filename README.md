# RSI-ML

`RSI-ML` is an experimental machine learning library written in Rust, built around a **seed-first** idea:
instead of storing only large static parameter tensors, tensors can be represented by compact procedural rules.

This repository is organized as a Cargo workspace and is designed for research on:

- procedural tensor generation (`Loaded` vs `Procedural`)
- lazy computation graphs
- lightweight autodiff for rapid experimentation

## Workspace Layout

- `crates/rsi_ml_core`  
  Core tensor system and graph engine:
  - `Tensor`
  - `TensorData` (`Loaded`, `Procedural`, `Expression`)
  - lazy evaluation + value caching
  - basic backward propagation

- `crates/rsi_ml_ops`  
  Operation layer (`add`, `mul`, `sum`) as free functions.

- `crates/rsi_ml_autograd`  
  Autograd extension trait (`AutogradExt`) for ergonomic gradient APIs.

- `crates/rsi_ml`  
  Public entry crate re-exporting the common API from all internal crates.

## Current Status

This project is in early experimental stage (MVP).

Implemented:

- procedural tensor materialization from `(seed, generator_func)`
- lazy expression graph (`add`, `mul`, `sum`)
- scalar-loss backward pass for tested operations
- unit tests for lazy behavior, cache behavior, and gradient correctness

Not yet implemented:

- full tensor broadcasting
- advanced ops (e.g., matmul/conv/attention)
- optimizers and training loop utilities
- GPU backend and SIMD specialization
- memory arena graph allocator

## Quick Start

### 1) Run tests

```bash
cargo test --workspace
```

### 2) Use the top-level crate

Add dependency (from another project):

```toml
[dependencies]
rsi_ml = { path = "../RSI_ML/crates/rsi_ml" }
```

Example:

```rust
use rsi_ml::{add, mul, sum, AutogradExt, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Tensor::from_loaded(vec![2.0, 3.0], vec![2], true)?;
    let b = Tensor::from_loaded(vec![4.0, 5.0], vec![2], true)?;

    let y = sum(&mul(&a, &b)?)?;
    y.backward_pass();

    assert_eq!(y.eval(), vec![23.0]);
    assert_eq!(a.grad().unwrap(), vec![4.0, 5.0]);
    assert_eq!(b.grad().unwrap(), vec![2.0, 3.0]);
    Ok(())
}
```

## Procedural Tensor Example

You can define tensors that generate values on demand:

```rust
use rsi_ml::Tensor;

fn seeded_linear(seed: u64, idx: usize) -> f32 {
    seed as f32 + idx as f32
}

fn main() {
    let t = Tensor::procedural(10, vec![4], seeded_linear, false);
    assert_eq!(t.eval(), vec![10.0, 11.0, 12.0, 13.0]);
}
```

## Vision

`RSI-ML` explores a different ML systems direction:

- compact seed equations as model priors
- selective/lazy computation instead of full dense execution
- future pathways for resonance-style operators and compressed state handling

The goal is not to replace mainstream deep learning overnight, but to provide a robust Rust research playground for new computational ideas.

## License

License is not defined yet.
Add a `LICENSE` file before public distribution.
