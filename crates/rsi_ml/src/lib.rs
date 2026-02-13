pub mod train;
pub mod data;
pub mod genome;
pub mod tokenizer;

pub use rsi_ml_autograd::AutogradExt;
pub use rsi_ml_core::{GeneratorFn, Tensor, TensorData, TensorError};
pub use rsi_ml_optim::{Adam, Optimizer, SGD, Sgd};
pub use rsi_ml_ops::{
    add, apply_causal_mask, attention_block, causal_mask, cross_entropy, embedding, huber, l1,
    layer_norm, linear, linear_gelu, linear_relu, log_softmax, matmul, mse, mul,
    multi_head_self_attention, relu, reshape, scaled_dot_product_attention, softmax, sub, sum,
    tanh, transpose2d, add_positional_encoding, decoder_next_token_logits, gelu,
    sinusoidal_positional_encoding, transformer_block, transformer_ffn, TransformerBlockWeights,
};
pub use train::{
    full_hybrid_train_loop, full_train_loop, HybridTrainConfig, HybridTrainStepMetrics,
    TrainLoopConfig, TrainStepMetrics,
};
pub use data::{one_hot, SequenceBatchIter, SequenceDataset};
pub use genome::{
    evolutionary_search_step, Genome, GenomeInstruction, GenomeMutator, MutationEvent,
    TransformerBlockWeightsOwned,
};
pub use tokenizer::CharTokenizer;

#[cfg(test)]
mod tests {
    use super::{
        apply_causal_mask, causal_mask, cross_entropy, embedding, full_train_loop, huber, l1,
        layer_norm, linear, linear_relu, log_softmax, matmul, mse, mul, relu, reshape, gelu,
        scaled_dot_product_attention, softmax, tanh, transpose2d, attention_block,
        multi_head_self_attention, transformer_block, transformer_ffn, TransformerBlockWeights,
        sinusoidal_positional_encoding, add_positional_encoding, decoder_next_token_logits, Adam,
        CharTokenizer, Genome, GenomeInstruction, GenomeMutator, MutationEvent, Optimizer, SGD, SequenceBatchIter,
        SequenceDataset, Tensor, TrainLoopConfig, HybridTrainConfig, full_hybrid_train_loop,
        evolutionary_search_step,
    };

    #[test]
    fn sgd_learns_y_eq_2x() {
        let weight = Tensor::from_loaded(vec![0.0], vec![1], true).unwrap();
        let mut optim = SGD::new(vec![weight.clone()], 0.1);

        for _step in 0..80 {
            let x = Tensor::from_loaded(vec![3.0], vec![1], false).unwrap();
            let y_true = Tensor::from_loaded(vec![6.0], vec![1], false).unwrap();

            let y_pred = mul(&weight, &x).unwrap();
            let loss = mse(&y_pred, &y_true).unwrap();

            loss.backward();
            optim.step();
            optim.zero_grad();
        }

        let learned = weight.eval()[0];
        assert!((learned - 2.0).abs() < 0.02, "learned weight = {learned}");
    }

    #[test]
    fn adam_learns_y_eq_2x() {
        let weight = Tensor::from_loaded(vec![0.0], vec![1], true).unwrap();
        let mut optim = Adam::new(vec![weight.clone()], 0.05, 0.9, 0.999, 1e-8);

        for _step in 0..120 {
            let x = Tensor::from_loaded(vec![3.0], vec![1], false).unwrap();
            let y_true = Tensor::from_loaded(vec![6.0], vec![1], false).unwrap();

            let y_pred = mul(&weight, &x).unwrap();
            let loss = mse(&y_pred, &y_true).unwrap();

            loss.backward();
            optim.step();
            optim.zero_grad();
        }

        let learned = weight.eval()[0];
        assert!((learned - 2.0).abs() < 0.03, "learned weight = {learned}");
    }

    #[test]
    fn l1_and_huber_loss_are_zero_on_match() {
        let pred = Tensor::from_loaded(vec![1.0, 2.0, 3.0], vec![3], false).unwrap();
        let target = Tensor::from_loaded(vec![1.0, 2.0, 3.0], vec![3], false).unwrap();

        let l1_loss = l1(&pred, &target).unwrap().eval()[0];
        let huber_loss = huber(&pred, &target, 1.0).unwrap().eval()[0];

        assert!(l1_loss.abs() < 1e-6);
        assert!(huber_loss.abs() < 1e-6);
    }

    #[test]
    fn matmul_api_matches_expected_output() {
        let a = Tensor::from_loaded(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false).unwrap();
        let b = Tensor::from_loaded(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], false).unwrap();
        let y = matmul(&a, &b).unwrap();
        assert_eq!(y.eval(), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn reshape_and_transpose_public_api_work() {
        let x = Tensor::from_loaded(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false).unwrap();
        let r = reshape(&x, vec![4]).unwrap();
        assert_eq!(r.eval(), vec![1.0, 2.0, 3.0, 4.0]);
        let t = transpose2d(&x).unwrap();
        assert_eq!(t.eval(), vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn two_layer_linear_stack_learns_target() {
        let w1 = Tensor::from_loaded(vec![0.1, -0.2], vec![1, 2], true).unwrap();
        let b1 = Tensor::from_loaded(vec![0.0, 0.0], vec![1, 2], true).unwrap();
        let w2 = Tensor::from_loaded(vec![0.3, -0.4], vec![2, 1], true).unwrap();
        let b2 = Tensor::from_loaded(vec![0.0], vec![1, 1], true).unwrap();

        let mut optim = SGD::new(vec![w1.clone(), b1.clone(), w2.clone(), b2.clone()], 0.05);

        for _ in 0..300 {
            let x = Tensor::from_loaded(vec![2.0], vec![1, 1], false).unwrap();
            let y_true = Tensor::from_loaded(vec![5.0], vec![1, 1], false).unwrap();

            let h = linear(&x, &w1, Some(&b1)).unwrap();
            let y_pred = linear(&h, &w2, Some(&b2)).unwrap();
            let loss = mse(&y_pred, &y_true).unwrap();

            loss.backward();
            optim.step();
            optim.zero_grad();
        }

        let x = Tensor::from_loaded(vec![2.0], vec![1, 1], false).unwrap();
        let y_true = Tensor::from_loaded(vec![5.0], vec![1, 1], false).unwrap();
        let h = linear(&x, &w1, Some(&b1)).unwrap();
        let y_pred = linear(&h, &w2, Some(&b2)).unwrap();
        let final_err = (y_pred.eval()[0] - y_true.eval()[0]).abs();
        assert!(final_err < 0.2, "final error = {final_err}");
    }

    #[test]
    fn relu_and_tanh_public_api_work() {
        let x = Tensor::from_loaded(vec![-1.0, 0.0, 1.0], vec![3], false).unwrap();
        let r = relu(&x).eval();
        let t = tanh(&x).eval();
        assert_eq!(r, vec![0.0, 0.0, 1.0]);
        assert!((t[0] - (-1.0_f32).tanh()).abs() < 1e-6);
        assert!(t[1].abs() < 1e-6);
        assert!((t[2] - 1.0_f32.tanh()).abs() < 1e-6);
    }

    #[test]
    fn two_layer_linear_relu_stack_learns_target() {
        let w1 = Tensor::from_loaded(vec![0.5, 0.5], vec![1, 2], true).unwrap();
        let b1 = Tensor::from_loaded(vec![0.1, 0.1], vec![1, 2], true).unwrap();
        let w2 = Tensor::from_loaded(vec![0.5, 0.5], vec![2, 1], true).unwrap();
        let b2 = Tensor::from_loaded(vec![0.0], vec![1, 1], true).unwrap();

        let mut optim = SGD::new(vec![w1.clone(), b1.clone(), w2.clone(), b2.clone()], 0.03);
        for _ in 0..400 {
            let x = Tensor::from_loaded(vec![2.0], vec![1, 1], false).unwrap();
            let y_true = Tensor::from_loaded(vec![5.0], vec![1, 1], false).unwrap();

            let h = linear_relu(&x, &w1, Some(&b1)).unwrap();
            let y_pred = linear(&h, &w2, Some(&b2)).unwrap();
            let loss = mse(&y_pred, &y_true).unwrap();

            loss.backward();
            optim.step();
            optim.zero_grad();
        }

        let x = Tensor::from_loaded(vec![2.0], vec![1, 1], false).unwrap();
        let y_true = Tensor::from_loaded(vec![5.0], vec![1, 1], false).unwrap();
        let y_pred = linear(&linear_relu(&x, &w1, Some(&b1)).unwrap(), &w2, Some(&b2)).unwrap();
        let final_err = (y_pred.eval()[0] - y_true.eval()[0]).abs();
        assert!(final_err < 0.3, "final error = {final_err}");
    }

    #[test]
    fn softmax_and_log_softmax_api_work() {
        let x = Tensor::from_loaded(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false).unwrap();
        let s = softmax(&x, 1).unwrap();
        let ls = log_softmax(&x, 1).unwrap();
        let s_eval = s.eval();
        let ls_eval = ls.eval();
        for i in 0..s_eval.len() {
            assert!((s_eval[i].ln() - ls_eval[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn cross_entropy_decreases_with_training() {
        let w = Tensor::from_loaded(vec![0.1, -0.1, -0.1, 0.1], vec![2, 2], true).unwrap();
        let mut optim = SGD::new(vec![w.clone()], 0.5);

        let x = Tensor::from_loaded(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], false).unwrap();
        let y = Tensor::from_loaded(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], false).unwrap();

        let initial_logits = matmul(&x, &w).unwrap();
        let initial_loss = cross_entropy(&initial_logits, &y, 1).unwrap().eval()[0];

        for _ in 0..120 {
            let logits = matmul(&x, &w).unwrap();
            let loss = cross_entropy(&logits, &y, 1).unwrap();
            loss.backward();
            optim.step();
            optim.zero_grad();
        }

        let final_logits = matmul(&x, &w).unwrap();
        let final_loss = cross_entropy(&final_logits, &y, 1).unwrap().eval()[0];
        assert!(final_loss < initial_loss, "{final_loss} !< {initial_loss}");
    }

    #[test]
    fn full_train_loop_updates_parameter() {
        let weight = Tensor::from_loaded(vec![0.0], vec![1], true).unwrap();
        let mut optim = SGD::new(vec![weight.clone()], 0.1);
        let config = TrainLoopConfig {
            epochs: 2,
            steps_per_epoch: 40,
        };

        full_train_loop(
            &mut optim,
            config,
            |_epoch, _step| {
                let x = Tensor::from_loaded(vec![3.0], vec![1], false)?;
                let y_true = Tensor::from_loaded(vec![6.0], vec![1], false)?;
                let y_pred = mul(&weight, &x)?;
                mse(&y_pred, &y_true)
            },
            |_m| {},
        )
        .unwrap();

        let learned = weight.eval()[0];
        assert!((learned - 2.0).abs() < 0.03, "learned weight = {learned}");
    }

    #[test]
    fn tokenizer_and_sequence_batch_iter_work() {
        let tok = CharTokenizer::from_text("abca");
        let ids = tok.encode("abca");
        assert_eq!(tok.decode(&ids), "abca");

        let ds = SequenceDataset::new(ids, 2);
        let mut it = SequenceBatchIter::new(&ds, 3);
        let batch = it.next_batch_indices();
        assert_eq!(batch.len(), 3);
        let (x, y) = ds.sample_pair(batch[0]);
        assert_eq!(x.len(), 2);
        assert_eq!(y.len(), 2);
    }

    #[test]
    fn embedding_shape_and_value_work() {
        let table = Tensor::from_loaded(
            vec![
                1.0, 0.0, // token 0
                0.0, 1.0, // token 1
                0.5, 0.5, // token 2
            ],
            vec![3, 2],
            true,
        )
        .unwrap();
        let out = embedding(&[2, 0], &table).unwrap();
        assert_eq!(out.shape(), vec![2, 2]);
        assert_eq!(out.eval(), vec![0.5, 0.5, 1.0, 0.0]);
    }

    #[test]
    fn causal_mask_and_attention_work() {
        let m = causal_mask(3, -1000.0).unwrap().eval();
        assert_eq!(m, vec![0.0, -1000.0, -1000.0, 0.0, 0.0, -1000.0, 0.0, 0.0, 0.0]);

        let q = Tensor::from_loaded(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], false).unwrap();
        let k = Tensor::from_loaded(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], false).unwrap();
        let v = Tensor::from_loaded(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false).unwrap();
        let out = scaled_dot_product_attention(&q, &k, &v, true).unwrap();
        assert_eq!(out.shape(), vec![2, 2]);

        let scores = matmul(&q, &k.transpose2d().unwrap()).unwrap();
        let masked = apply_causal_mask(&scores, -1e9).unwrap();
        let probs = softmax(&masked, 1).unwrap().eval();
        assert!(probs[1].abs() < 1e-6);
    }

    #[test]
    fn layer_norm_row_stats_work() {
        let x = Tensor::from_loaded(vec![1.0, 2.0, 3.0, 2.0, 2.0, 2.0], vec![2, 3], false).unwrap();
        let y = layer_norm(&x, None, None, 1e-5).unwrap();
        let v = y.eval();
        // row 1 should be centered
        let m0 = (v[0] + v[1] + v[2]) / 3.0;
        assert!(m0.abs() < 1e-4);
    }

    #[test]
    fn multi_head_attention_and_block_shape_work() {
        let t = 4usize;
        let d = 4usize;
        let x = Tensor::from_loaded(
            (0..t * d).map(|i| i as f32 * 0.1).collect(),
            vec![t, d],
            true,
        )
        .unwrap();
        let ident = Tensor::from_loaded(
            vec![
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 1.0,
            ],
            vec![d, d],
            true,
        )
        .unwrap();

        let attn = multi_head_self_attention(&x, &ident, &ident, &ident, &ident, 2, true).unwrap();
        assert_eq!(attn.shape(), vec![t, d]);

        let block = attention_block(&x, &ident, &ident, &ident, &ident, 2, 1e-5).unwrap();
        assert_eq!(block.shape(), vec![t, d]);
    }

    #[test]
    fn gelu_and_ffn_shape_work() {
        let x = Tensor::from_loaded(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2], true).unwrap();
        let g = gelu(&x).unwrap();
        assert_eq!(g.shape(), vec![2, 2]);

        let w1 = Tensor::from_loaded(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2], true).unwrap();
        let b1 = Tensor::from_loaded(vec![0.0, 0.0], vec![1, 2], true).unwrap();
        let w2 = Tensor::from_loaded(vec![0.5, 0.6, 0.7, 0.8], vec![2, 2], true).unwrap();
        let b2 = Tensor::from_loaded(vec![0.0, 0.0], vec![1, 2], true).unwrap();
        let y = transformer_ffn(&x, &w1, Some(&b1), &w2, Some(&b2)).unwrap();
        assert_eq!(y.shape(), vec![2, 2]);
    }

    #[test]
    fn transformer_block_forward_backward_work() {
        let t = 4usize;
        let d = 4usize;
        let ff = 8usize;
        let x = Tensor::from_loaded(
            (0..t * d).map(|i| i as f32 * 0.01).collect(),
            vec![t, d],
            true,
        )
        .unwrap();

        let ident = Tensor::from_loaded(
            vec![
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 1.0,
            ],
            vec![d, d],
            true,
        )
        .unwrap();
        let w_ff1 = Tensor::from_loaded(
            (0..d * ff).map(|i| ((i % 5) as f32 - 2.0) * 0.05).collect(),
            vec![d, ff],
            true,
        )
        .unwrap();
        let b_ff1 = Tensor::from_loaded(vec![0.0; ff], vec![1, ff], true).unwrap();
        let w_ff2 = Tensor::from_loaded(
            (0..ff * d).map(|i| ((i % 7) as f32 - 3.0) * 0.03).collect(),
            vec![ff, d],
            true,
        )
        .unwrap();
        let b_ff2 = Tensor::from_loaded(vec![0.0; d], vec![1, d], true).unwrap();

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
        )
        .unwrap();
        assert_eq!(y.shape(), vec![t, d]);

        let loss = y.sum().unwrap();
        loss.backward();
        assert!(x.grad().is_some());
        assert!(w_ff1.grad().is_some());
        assert!(w_ff2.grad().is_some());
    }

    #[test]
    fn positional_encoding_and_decoder_logits_work() {
        let pos = sinusoidal_positional_encoding(4, 6).unwrap();
        assert_eq!(pos.shape(), vec![4, 6]);

        let token_table = Tensor::from_loaded(vec![0.1; 5 * 6], vec![5, 6], true).unwrap();
        let lm_head = Tensor::from_loaded(vec![0.2; 6 * 5], vec![6, 5], true).unwrap();
        let logits = decoder_next_token_logits(&[1, 2, 3, 4], &token_table, &lm_head).unwrap();
        assert_eq!(logits.shape(), vec![4, 5]);

        let emb = embedding(&[1, 2, 3, 4], &token_table).unwrap();
        let added = add_positional_encoding(&emb, &pos).unwrap();
        assert_eq!(added.shape(), vec![4, 6]);
    }

    #[test]
    fn genome_seed_forward_and_complexity_work() {
        let genome = Genome::from_seed_mlp(42, 4, 6, 3).unwrap();
        let x = Tensor::from_loaded(vec![0.1; 8], vec![2, 4], false).unwrap();
        let y = genome.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 3]);
        assert!(genome.complexity_score() > 0);
    }

    #[test]
    fn genome_kolmogorov_loss_and_mutation_work() {
        let mut genome = Genome::with_instructions(vec![GenomeInstruction::Relu]);
        let task_loss = Tensor::from_loaded(vec![1.0], vec![1], false).unwrap();
        let total = genome.kolmogorov_loss(&task_loss, 0.001).unwrap().eval()[0];
        assert!(total > 1.0);

        let changed = genome.mutate_toggle_activation(0);
        assert!(changed);
    }

    #[test]
    fn genome_mutator_and_search_step_work() {
        let mut genome = Genome::from_seed_mlp(7, 2, 4, 1).unwrap();
        let mut mutator = GenomeMutator::new(123);
        let event = mutator.mutate_once(&mut genome);
        assert!(event != MutationEvent::Noop || !genome.instructions.is_empty());

        let x = Tensor::from_loaded(vec![1.0, 2.0, 2.0, 1.0], vec![2, 2], false).unwrap();
        let y = Tensor::from_loaded(vec![3.0, 3.0], vec![2, 1], false).unwrap();
        let (best, best_score) = evolutionary_search_step(&genome, &mut mutator, 8, |g| {
            let pred = g.forward(&x)?;
            let task = mse(&pred, &y)?.eval()[0];
            Ok(task + g.complexity_score() as f32 * 0.0001)
        })
        .unwrap();
        assert!(best_score.is_finite());
        assert!(!best.instructions.is_empty());
    }

    #[test]
    fn hybrid_train_loop_runs() {
        let mut genome = Genome::from_seed_mlp(77, 2, 6, 1).unwrap();
        let mut mutator = GenomeMutator::new(5);
        let mut params = Vec::new();
        for inst in &genome.instructions {
            if let GenomeInstruction::Linear { weight, bias } = inst {
                params.push(weight.clone());
                if let Some(b) = bias {
                    params.push(b.clone());
                }
            }
        }
        let mut optim = SGD::new(params, 0.05);
        let x = Tensor::from_loaded(vec![1.0, 2.0, 2.0, 1.0], vec![2, 2], false).unwrap();
        let y = Tensor::from_loaded(vec![3.0, 3.0], vec![2, 1], false).unwrap();

        let cfg = HybridTrainConfig {
            epochs: 1,
            steps_per_epoch: 10,
            mutation_every: 2,
            mutation_trials: 4,
        };

        full_hybrid_train_loop(
            &mut optim,
            &mut genome,
            &mut mutator,
            cfg,
            |g, _, _| {
                let pred = g.forward(&x)?;
                mse(&pred, &y)
            },
            |g, _, _| {
                let pred = g.forward(&x)?;
                let task = mse(&pred, &y)?.eval()[0];
                Ok(task + g.complexity_score() as f32 * 0.0001)
            },
            |_m| {},
        )
        .unwrap();
    }
}
