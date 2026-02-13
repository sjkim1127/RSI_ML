use rsi_ml_core::{Tensor, TensorError};
use std::cell::RefCell;
use std::collections::HashMap;

pub fn add(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, TensorError> {
    lhs.add(rhs)
}

pub fn sub(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, TensorError> {
    lhs.sub(rhs)
}

pub fn mul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, TensorError> {
    lhs.mul(rhs)
}

pub fn matmul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, TensorError> {
    lhs.matmul(rhs)
}

pub fn reshape(tensor: &Tensor, new_shape: Vec<usize>) -> Result<Tensor, TensorError> {
    tensor.reshape(new_shape)
}

pub fn transpose2d(tensor: &Tensor) -> Result<Tensor, TensorError> {
    tensor.transpose2d()
}

pub fn sum(tensor: &Tensor) -> Result<Tensor, TensorError> {
    tensor.sum()
}

pub fn linear(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Tensor, TensorError> {
    let out = input.matmul(weight)?;
    match bias {
        Some(b) => {
            if b.shape() == out.shape() {
                out.add(b)
            } else {
                let out_shape = out.shape();
                let b_shape = b.shape();
                if b_shape.len() == 2 && b_shape[0] == 1 && b_shape[1] == out_shape[1] {
                    let b_expanded = b.repeat_rows(out_shape[0])?;
                    out.add(&b_expanded)
                } else {
                    Err(TensorError::ShapeMismatch {
                        lhs: out_shape,
                        rhs: b_shape,
                        op: "linear_bias",
                    })
                }
            }
        }
        None => Ok(out),
    }
}

pub fn relu(tensor: &Tensor) -> Tensor {
    tensor.relu()
}

pub fn tanh(tensor: &Tensor) -> Tensor {
    tensor.tanh()
}

pub fn softmax(tensor: &Tensor, dim: usize) -> Result<Tensor, TensorError> {
    tensor.softmax(dim)
}

pub fn log_softmax(tensor: &Tensor, dim: usize) -> Result<Tensor, TensorError> {
    tensor.log_softmax(dim)
}

pub fn linear_relu(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Tensor, TensorError> {
    Ok(linear(input, weight, bias)?.relu())
}

fn full_like(shape: Vec<usize>, value: f32) -> Result<Tensor, TensorError> {
    let len: usize = shape.iter().product();
    Tensor::from_loaded(vec![value; len], shape, false)
}

pub fn gelu(input: &Tensor) -> Result<Tensor, TensorError> {
    // GELU approximation:
    // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let shape = input.shape();
    let x2 = input.mul(input)?;
    let x3 = x2.mul(input)?;
    let c = full_like(shape.clone(), 0.044715)?;
    let inner = input.add(&x3.mul(&c)?)?;
    let k = full_like(shape.clone(), 0.7978846)?; // sqrt(2/pi)
    let t = inner.mul(&k)?.tanh();
    let one = full_like(shape.clone(), 1.0)?;
    let half = full_like(shape, 0.5)?;
    half.mul(input)?.mul(&one.add(&t)?)
}

pub fn linear_gelu(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Tensor, TensorError> {
    gelu(&linear(input, weight, bias)?)
}

pub fn embedding(token_ids: &[usize], table: &Tensor) -> Result<Tensor, TensorError> {
    let shape = table.shape();
    if shape.len() != 2 {
        return Err(TensorError::InvalidShape { shape, data_len: 0 });
    }
    let vocab = shape[0];
    let dim = shape[1];

    // Fast gather path for inference/non-trainable embedding tables.
    if !table.requires_grad() {
        let t = table.eval();
        let mut out = vec![0.0; token_ids.len() * dim];
        for (row, token_id) in token_ids.iter().enumerate() {
            if *token_id >= vocab {
                return Err(TensorError::InvalidShape {
                    shape: vec![token_ids.len(), vocab],
                    data_len: *token_id,
                });
            }
            let src = &t[*token_id * dim..(*token_id + 1) * dim];
            let dst = &mut out[row * dim..(row + 1) * dim];
            dst.copy_from_slice(src);
        }
        return Tensor::from_loaded(out, vec![token_ids.len(), dim], false);
    }

    // Gradient-preserving fallback path for trainable embedding tables.
    let mut one_hot = vec![0.0; token_ids.len() * vocab];
    for (row, token_id) in token_ids.iter().enumerate() {
        if *token_id >= vocab {
            return Err(TensorError::InvalidShape {
                shape: vec![token_ids.len(), vocab],
                data_len: *token_id,
            });
        }
        one_hot[row * vocab + *token_id] = 1.0;
    }

    let x = Tensor::from_loaded(one_hot, vec![token_ids.len(), vocab], false)?;
    let out = x.matmul(table)?;
    if out.shape() != vec![token_ids.len(), dim] {
        return Err(TensorError::InvalidShape {
            shape: out.shape(),
            data_len: token_ids.len() * dim,
        });
    }
    Ok(out)
}

pub fn sinusoidal_positional_encoding(seq_len: usize, d_model: usize) -> Result<Tensor, TensorError> {
    if d_model == 0 {
        return Err(TensorError::InvalidShape {
            shape: vec![seq_len, d_model],
            data_len: 0,
        });
    }
    let mut data = vec![0.0; seq_len * d_model];
    for pos in 0..seq_len {
        for i in 0..d_model {
            let pair = (i / 2) as f32;
            let denom = 10000.0_f32.powf((2.0 * pair) / d_model as f32);
            let angle = pos as f32 / denom;
            data[pos * d_model + i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
        }
    }
    Tensor::from_loaded(data, vec![seq_len, d_model], false)
}

pub fn add_positional_encoding(token_embeddings: &Tensor, pos_encoding: &Tensor) -> Result<Tensor, TensorError> {
    token_embeddings.add(pos_encoding)
}

pub fn causal_mask(seq_len: usize, fill_value: f32) -> Result<Tensor, TensorError> {
    let mut v = vec![0.0; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            v[i * seq_len + j] = fill_value;
        }
    }
    Tensor::from_loaded(v, vec![seq_len, seq_len], false)
}

pub fn apply_causal_mask(scores: &Tensor, fill_value: f32) -> Result<Tensor, TensorError> {
    let shape = scores.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(TensorError::InvalidShape { shape, data_len: 0 });
    }
    let mask = causal_mask(shape[0], fill_value)?;
    scores.add(&mask)
}

pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    use_causal_mask: bool,
) -> Result<Tensor, TensorError> {
    let q_shape = q.shape();
    let k_shape = k.shape();
    let v_shape = v.shape();
    if q_shape.len() != 2 || k_shape.len() != 2 || v_shape.len() != 2 {
        return Err(TensorError::InvalidShape {
            shape: q_shape,
            data_len: 0,
        });
    }
    if q_shape[0] != k_shape[0] || k_shape[0] != v_shape[0] {
        return Err(TensorError::ShapeMismatch {
            lhs: q_shape,
            rhs: k_shape,
            op: "attention_seq_len",
        });
    }
    if q_shape[1] != k_shape[1] {
        return Err(TensorError::ShapeMismatch {
            lhs: q_shape,
            rhs: k_shape,
            op: "attention_qk_dim",
        });
    }

    let k_t = k.transpose2d()?;
    let mut scores = q.matmul(&k_t)?;
    let scale = 1.0 / (q_shape[1] as f32).sqrt();
    let scale_tensor = Tensor::from_loaded(
        vec![scale; scores.shape().iter().product()],
        scores.shape(),
        false,
    )?;
    scores = scores.mul(&scale_tensor)?;

    if use_causal_mask {
        scores = apply_causal_mask(&scores, -1e9)?;
    }

    let probs = scores.softmax(1)?;
    probs.matmul(v)
}

pub fn layer_norm(
    input: &Tensor,
    gamma: Option<&Tensor>,
    beta: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor, TensorError> {
    let shape = input.shape();
    if shape.len() != 2 {
        return Err(TensorError::InvalidShape { shape, data_len: 0 });
    }
    let rows = shape[0];
    let cols = shape[1];
    if cols == 0 {
        return Err(TensorError::EmptyReduction);
    }

    let inv_cols = Tensor::from_loaded(vec![1.0 / cols as f32; rows], vec![rows, 1], false)?;
    let mean = input.sum_axis(1)?.mul(&inv_cols)?;
    let centered = input.sub(&mean.repeat_cols(cols)?)?;
    let var = centered.mul(&centered)?.sum_axis(1)?.mul(&inv_cols)?;
    let eps_t = Tensor::from_loaded(vec![eps; rows], vec![rows, 1], false)?;
    let std = var.add(&eps_t)?.sqrt();
    let mut out = centered.div(&std.repeat_cols(cols)?)?;

    if let Some(g) = gamma {
        let g_shape = g.shape();
        if g_shape != vec![1, cols] {
            return Err(TensorError::ShapeMismatch {
                lhs: g_shape,
                rhs: vec![1, cols],
                op: "layer_norm_gamma",
            });
        }
        out = out.mul(&g.repeat_rows(rows)?)?;
    }
    if let Some(b) = beta {
        let b_shape = b.shape();
        if b_shape != vec![1, cols] {
            return Err(TensorError::ShapeMismatch {
                lhs: b_shape,
                rhs: vec![1, cols],
                op: "layer_norm_beta",
            });
        }
        out = out.add(&b.repeat_rows(rows)?)?;
    }

    Ok(out)
}

fn select_matrix(d_model: usize, d_head: usize, offset: usize) -> Result<Tensor, TensorError> {
    let mut m = vec![0.0; d_model * d_head];
    for i in 0..d_head {
        m[(offset + i) * d_head + i] = 1.0;
    }
    Tensor::from_loaded(m, vec![d_model, d_head], false)
}

fn place_matrix(d_head: usize, d_model: usize, offset: usize) -> Result<Tensor, TensorError> {
    let mut m = vec![0.0; d_head * d_model];
    for i in 0..d_head {
        m[i * d_model + (offset + i)] = 1.0;
    }
    Tensor::from_loaded(m, vec![d_head, d_model], false)
}

fn head_projection_matrices(
    d_model: usize,
    num_heads: usize,
) -> Result<Vec<(Tensor, Tensor)>, TensorError> {
    thread_local! {
        static CACHE: RefCell<HashMap<(usize, usize), Vec<(Tensor, Tensor)>>> = RefCell::new(HashMap::new());
    }

    if let Some(v) = CACHE.with(|c| c.borrow().get(&(d_model, num_heads)).cloned()) {
        return Ok(v);
    }

    let d_head = d_model / num_heads;
    let mut mats = Vec::with_capacity(num_heads);
    for h in 0..num_heads {
        let offset = h * d_head;
        let select = select_matrix(d_model, d_head, offset)?;
        let place = place_matrix(d_head, d_model, offset)?;
        mats.push((select, place));
    }

    CACHE.with(|c| {
        c.borrow_mut().insert((d_model, num_heads), mats.clone());
    });
    Ok(mats)
}

pub fn multi_head_self_attention(
    input: &Tensor,
    w_q: &Tensor,
    w_k: &Tensor,
    w_v: &Tensor,
    w_o: &Tensor,
    num_heads: usize,
    use_causal_mask: bool,
) -> Result<Tensor, TensorError> {
    let in_shape = input.shape();
    if in_shape.len() != 2 {
        return Err(TensorError::InvalidShape {
            shape: in_shape,
            data_len: 0,
        });
    }
    let d_model = in_shape[1];
    if num_heads == 0 || d_model % num_heads != 0 {
        return Err(TensorError::InvalidShape {
            shape: vec![in_shape[0], d_model],
            data_len: num_heads,
        });
    }

    let q = input.matmul(w_q)?;
    let k = input.matmul(w_k)?;
    let v = input.matmul(w_v)?;
    let projection_mats = head_projection_matrices(d_model, num_heads)?;

    let mut combined: Option<Tensor> = None;
    for (select, place) in projection_mats.iter().take(num_heads) {

        let q_h = q.matmul(select)?;
        let k_h = k.matmul(select)?;
        let v_h = v.matmul(select)?;
        let attn_h = scaled_dot_product_attention(&q_h, &k_h, &v_h, use_causal_mask)?;
        let packed = attn_h.matmul(place)?;

        combined = Some(match combined {
            Some(acc) => acc.add(&packed)?,
            None => packed,
        });
    }

    let merged = combined.ok_or(TensorError::InvalidShape {
        shape: vec![in_shape[0], d_model],
        data_len: 0,
    })?;
    merged.matmul(w_o)
}

pub fn attention_block(
    input: &Tensor,
    w_q: &Tensor,
    w_k: &Tensor,
    w_v: &Tensor,
    w_o: &Tensor,
    num_heads: usize,
    eps: f32,
) -> Result<Tensor, TensorError> {
    let norm = layer_norm(input, None, None, eps)?;
    let attn = multi_head_self_attention(&norm, w_q, w_k, w_v, w_o, num_heads, true)?;
    input.add(&attn)
}

pub fn transformer_ffn(
    input: &Tensor,
    w1: &Tensor,
    b1: Option<&Tensor>,
    w2: &Tensor,
    b2: Option<&Tensor>,
) -> Result<Tensor, TensorError> {
    let h = linear_gelu(input, w1, b1)?;
    linear(&h, w2, b2)
}

pub struct TransformerBlockWeights<'a> {
    pub w_q: &'a Tensor,
    pub w_k: &'a Tensor,
    pub w_v: &'a Tensor,
    pub w_o: &'a Tensor,
    pub w_ff1: &'a Tensor,
    pub b_ff1: Option<&'a Tensor>,
    pub w_ff2: &'a Tensor,
    pub b_ff2: Option<&'a Tensor>,
}

pub fn transformer_block(
    input: &Tensor,
    weights: TransformerBlockWeights<'_>,
    num_heads: usize,
    eps: f32,
) -> Result<Tensor, TensorError> {
    // Pre-Norm attention + residual
    let x1 = attention_block(
        input,
        weights.w_q,
        weights.w_k,
        weights.w_v,
        weights.w_o,
        num_heads,
        eps,
    )?;

    // Pre-Norm FFN + residual
    let norm2 = layer_norm(&x1, None, None, eps)?;
    let ffn = transformer_ffn(&norm2, weights.w_ff1, weights.b_ff1, weights.w_ff2, weights.b_ff2)?;
    x1.add(&ffn)
}

pub fn decoder_next_token_logits(
    token_ids: &[usize],
    token_embedding_table: &Tensor,
    lm_head_weight: &Tensor,
) -> Result<Tensor, TensorError> {
    let token_emb = embedding(token_ids, token_embedding_table)?;
    let seq_len = token_emb.shape()[0];
    let d_model = token_emb.shape()[1];
    let pos = sinusoidal_positional_encoding(seq_len, d_model)?;
    let hidden = add_positional_encoding(&token_emb, &pos)?;
    hidden.matmul(lm_head_weight)
}

pub fn mse(pred: &Tensor, target: &Tensor) -> Result<Tensor, TensorError> {
    let diff = pred.sub(target)?;
    let sq = diff.mul(&diff)?;
    mean_of(&sq)
}

pub fn l1(pred: &Tensor, target: &Tensor) -> Result<Tensor, TensorError> {
    let diff = pred.sub(target)?;
    let abs = diff.abs();
    mean_of(&abs)
}

pub fn huber(pred: &Tensor, target: &Tensor, delta: f32) -> Result<Tensor, TensorError> {
    let diff = pred.sub(target)?;
    let huber_vals = diff.huber(delta);
    mean_of(&huber_vals)
}

pub fn cross_entropy(logits: &Tensor, target_probs: &Tensor, dim: usize) -> Result<Tensor, TensorError> {
    if logits.shape() != target_probs.shape() {
        return Err(TensorError::ShapeMismatch {
            lhs: logits.shape(),
            rhs: target_probs.shape(),
            op: "cross_entropy",
        });
    }
    let shape = logits.shape();
    if shape.len() != 2 || dim > 1 {
        return Err(TensorError::InvalidShape { shape, data_len: dim });
    }

    let log_probs = logits.log_softmax(dim)?;
    let neg_target = Tensor::from_loaded(
        target_probs.eval().into_iter().map(|v| -v).collect(),
        target_probs.shape(),
        false,
    )?;
    let weighted = log_probs.mul(&neg_target)?;
    let per_sample = weighted.sum_axis(dim)?;
    let total = per_sample.sum()?;

    let batch = if dim == 1 {
        logits.shape()[0]
    } else {
        logits.shape()[1]
    };
    if batch == 0 {
        return Err(TensorError::EmptyReduction);
    }
    let inv_batch = Tensor::from_loaded(vec![1.0 / batch as f32], vec![1], false)?;
    total.mul(&inv_batch)
}

fn mean_of(tensor: &Tensor) -> Result<Tensor, TensorError> {
    let sum_v = tensor.sum()?;
    let count = tensor.shape().iter().product::<usize>();
    if count == 0 {
        return Err(TensorError::EmptyReduction);
    }
    let inv_count = Tensor::from_loaded(vec![1.0 / count as f32], vec![1], false)?;
    sum_v.mul(&inv_count)
}
