use rsi_ml::{transformer_block, Tensor, TransformerBlockWeights};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let t = 6usize;
    let d = 4usize;
    let ff = 8usize;
    let x = Tensor::from_loaded(
        (0..t * d).map(|i| i as f32 * 0.05).collect(),
        vec![t, d],
        true,
    )?;

    let ident = Tensor::from_loaded(
        vec![
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ],
        vec![d, d],
        true,
    )?;
    let w_ff1 = Tensor::from_loaded(vec![0.03; d * ff], vec![d, ff], true)?;
    let b_ff1 = Tensor::from_loaded(vec![0.0; ff], vec![1, ff], true)?;
    let w_ff2 = Tensor::from_loaded(vec![0.02; ff * d], vec![ff, d], true)?;
    let b_ff2 = Tensor::from_loaded(vec![0.0; d], vec![1, d], true)?;

    let y = transformer_block(
        &x,
        TransformerBlockWeights {
            w_q: &ident,
            w_k: &ident,
            w_v: &ident,
            w_o: &ident,
            w_ff1: &w_ff1,
            b_ff1: Some(&b_ff1),
            w_ff2: &w_ff2,
            b_ff2: Some(&b_ff2),
        },
        2,
        1e-5,
    )?;
    println!("input_shape={:?}", x.shape());
    println!("output_shape={:?}", y.shape());
    println!("output_head={:?}", &y.eval()[0..d]);
    Ok(())
}
