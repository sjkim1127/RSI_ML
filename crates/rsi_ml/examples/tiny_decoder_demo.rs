use rsi_ml::{
    decoder_next_token_logits, embedding, sinusoidal_positional_encoding, softmax, Tensor,
};

fn argmax(slice: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, v) in slice.iter().enumerate() {
        if *v > best_v {
            best_v = *v;
            best_i = i;
        }
    }
    best_i
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vocab = 8usize;
    let d_model = 6usize;
    let token_table = Tensor::from_loaded(
        (0..vocab * d_model).map(|i| ((i % 9) as f32 - 4.0) * 0.05).collect(),
        vec![vocab, d_model],
        true,
    )?;
    let lm_head = Tensor::from_loaded(
        (0..d_model * vocab).map(|i| ((i % 7) as f32 - 3.0) * 0.07).collect(),
        vec![d_model, vocab],
        true,
    )?;

    let ids = vec![1, 2, 3, 2, 1];
    let logits = decoder_next_token_logits(&ids, &token_table, &lm_head)?;
    let probs = softmax(&logits, 1)?;
    let p = probs.eval();
    let last = &p[(ids.len() - 1) * vocab..ids.len() * vocab];
    let next = argmax(last);

    let emb = embedding(&ids, &token_table)?;
    let pos = sinusoidal_positional_encoding(ids.len(), d_model)?;
    println!("emb_shape={:?} pos_shape={:?}", emb.shape(), pos.shape());
    println!("logits_shape={:?} predicted_next_token={}", logits.shape(), next);
    Ok(())
}
