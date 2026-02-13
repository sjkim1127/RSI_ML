use rsi_ml::{embedding, scaled_dot_product_attention, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vocab = 6;
    let d_model = 4;

    let embed_init: Vec<f32> = (0..vocab * d_model)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.2)
        .collect();
    let embed_table = Tensor::from_loaded(embed_init, vec![vocab, d_model], true)?;

    let token_ids = vec![1, 2, 3, 1];
    let x = embedding(&token_ids, &embed_table)?;

    // Tiny single-head self-attention demo: Q=K=V=X
    let out = scaled_dot_product_attention(&x, &x, &x, true)?;
    println!("input_shape={:?}", x.shape());
    println!("output_shape={:?}", out.shape());
    println!("output={:?}", out.eval());
    Ok(())
}
