use rsi_ml::{
    add_positional_encoding, cross_entropy, embedding, full_train_loop, matmul, one_hot,
    sinusoidal_positional_encoding, transformer_block, CharTokenizer, SGD, SequenceBatchIter,
    SequenceDataset, Tensor, TrainLoopConfig, TransformerBlockWeights,
};

fn argmax(slice: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, v) in slice.iter().enumerate() {
        if *v > best_val {
            best_val = *v;
            best_idx = idx;
        }
    }
    best_idx
}

fn next_rand01(state: &mut u64) -> f32 {
    // Simple LCG for deterministic sampling without extra dependencies.
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    let x = (*state >> 32) as u32;
    (x as f32) / (u32::MAX as f32)
}

fn sample_top_k_temperature(logits: &[f32], temperature: f32, top_k: usize, rng: &mut u64) -> usize {
    if top_k == 0 {
        return argmax(logits);
    }

    let t = if temperature <= 1e-6 { 1e-6 } else { temperature };
    let mut pairs: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let k = top_k.min(pairs.len());
    pairs.truncate(k);

    let max_logit = pairs[0].1;
    let mut weights = Vec::with_capacity(k);
    let mut total = 0.0;
    for (_, logit) in &pairs {
        let w = ((*logit - max_logit) / t).exp();
        weights.push(w);
        total += w;
    }
    if total <= 0.0 {
        return pairs[0].0;
    }

    let mut r = next_rand01(rng) * total;
    for (i, w) in weights.iter().enumerate() {
        if r <= *w {
            return pairs[i].0;
        }
        r -= *w;
    }
    pairs[k - 1].0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let corpus = "hello rust world. hello rust ml. ".repeat(40);
    let tokenizer = CharTokenizer::from_text(&corpus);
    let tokens = tokenizer.encode(&corpus);
    let vocab = tokenizer.vocab_size();
    let d_model = 16usize;
    let ff_hidden = 32usize;
    let num_heads = 2usize;

    let dataset = SequenceDataset::new(tokens, 8);
    let mut iter = SequenceBatchIter::new(&dataset, 1);

    let token_embedding = Tensor::from_loaded(
        (0..vocab * d_model)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
            .collect(),
        vec![vocab, d_model],
        true,
    )?;
    let w_q = Tensor::from_loaded(vec![0.01; d_model * d_model], vec![d_model, d_model], true)?;
    let w_k = Tensor::from_loaded(vec![0.01; d_model * d_model], vec![d_model, d_model], true)?;
    let w_v = Tensor::from_loaded(vec![0.01; d_model * d_model], vec![d_model, d_model], true)?;
    let w_o = Tensor::from_loaded(vec![0.01; d_model * d_model], vec![d_model, d_model], true)?;
    let w_ff1 = Tensor::from_loaded(
        (0..d_model * ff_hidden)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.01)
            .collect(),
        vec![d_model, ff_hidden],
        true,
    )?;
    let b_ff1 = Tensor::from_loaded(vec![0.0; ff_hidden], vec![1, ff_hidden], true)?;
    let w_ff2 = Tensor::from_loaded(
        (0..ff_hidden * d_model)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.01)
            .collect(),
        vec![ff_hidden, d_model],
        true,
    )?;
    let b_ff2 = Tensor::from_loaded(vec![0.0; d_model], vec![1, d_model], true)?;
    let lm_head = Tensor::from_loaded(
        (0..d_model * vocab)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.01)
            .collect(),
        vec![d_model, vocab],
        true,
    )?;

    let mut optim = SGD::new(
        vec![
            token_embedding.clone(),
            w_q.clone(),
            w_k.clone(),
            w_v.clone(),
            w_o.clone(),
            w_ff1.clone(),
            b_ff1.clone(),
            w_ff2.clone(),
            b_ff2.clone(),
            lm_head.clone(),
        ],
        0.08,
    );

    let config = TrainLoopConfig {
        epochs: 6,
        steps_per_epoch: 120,
    };

    full_train_loop(
        &mut optim,
        config,
        |_epoch, _step| {
            let start = iter.next_batch_indices()[0];
            let (x_seq, y_seq) = dataset.sample_pair(start);
            let x_emb = embedding(x_seq, &token_embedding)?;
            let pos = sinusoidal_positional_encoding(x_seq.len(), d_model)?;
            let h0 = add_positional_encoding(&x_emb, &pos)?;
            let h1 = transformer_block(
                &h0,
                TransformerBlockWeights {
                    w_q: &w_q,
                    w_k: &w_k,
                    w_v: &w_v,
                    w_o: &w_o,
                    w_ff1: &w_ff1,
                    b_ff1: Some(&b_ff1),
                    w_ff2: &w_ff2,
                    b_ff2: Some(&b_ff2),
                },
                num_heads,
                1e-5,
            )?;
            let logits = matmul(&h1, &lm_head)?;
            let y_ids = y_seq.to_vec();
            let y = Tensor::from_loaded(one_hot(&y_ids, vocab), vec![y_ids.len(), vocab], false)?;
            cross_entropy(&logits, &y, 1)
        },
        |m| {
            if m.step % 40 == 0 {
                println!("epoch={} step={} loss={:.6}", m.epoch, m.step, m.loss);
            }
        },
    )?;

    let prompt = "hell";
    let prompt_ids = tokenizer.encode(prompt);
    let x_emb = embedding(&prompt_ids, &token_embedding)?;
    let pos = sinusoidal_positional_encoding(prompt_ids.len(), d_model)?;
    let h0 = add_positional_encoding(&x_emb, &pos)?;
    let h1 = transformer_block(
        &h0,
        TransformerBlockWeights {
            w_q: &w_q,
            w_k: &w_k,
            w_v: &w_v,
            w_o: &w_o,
            w_ff1: &w_ff1,
            b_ff1: Some(&b_ff1),
            w_ff2: &w_ff2,
            b_ff2: Some(&b_ff2),
        },
        num_heads,
        1e-5,
    )?;
    let logits = matmul(&h1, &lm_head)?.eval();
    let start = (prompt_ids.len() - 1) * vocab;
    let next_id = argmax(&logits[start..start + vocab]);
    println!("prompt='{}' next='{}'", prompt, tokenizer.decode(&[next_id]));

    // Autoregressive generation with temperature + top-k sampling.
    let mut generated = prompt_ids.clone();
    let seq_len = dataset.seq_len;
    let mut rng_state = 123456789_u64;
    let temperature = 0.9_f32;
    let top_k = 5usize;
    let gen_tokens = 40usize;

    for _ in 0..gen_tokens {
        let start_idx = generated.len().saturating_sub(seq_len);
        let ctx = &generated[start_idx..];
        let x_emb = embedding(ctx, &token_embedding)?;
        let pos = sinusoidal_positional_encoding(ctx.len(), d_model)?;
        let h0 = add_positional_encoding(&x_emb, &pos)?;
        let h1 = transformer_block(
            &h0,
            TransformerBlockWeights {
                w_q: &w_q,
                w_k: &w_k,
                w_v: &w_v,
                w_o: &w_o,
                w_ff1: &w_ff1,
                b_ff1: Some(&b_ff1),
                w_ff2: &w_ff2,
                b_ff2: Some(&b_ff2),
            },
            num_heads,
            1e-5,
        )?;
        let logits = matmul(&h1, &lm_head)?.eval();
        let row_start = (ctx.len() - 1) * vocab;
        let next = sample_top_k_temperature(
            &logits[row_start..row_start + vocab],
            temperature,
            top_k,
            &mut rng_state,
        );
        generated.push(next);
    }

    let generated_text = tokenizer.decode(&generated);
    println!("generated(text): {}", generated_text);
    Ok(())
}
