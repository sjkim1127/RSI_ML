use rsi_ml::{
    full_hybrid_train_loop_with_sleep, mse, Genome, GenomeInstruction, GenomeMutator,
    HybridTrainConfig, HybridTrainStepMetrics, SleepPhaseConfig, SGD, Tensor,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut genome = Genome::from_seed_mlp(123, 2, 8, 1)?;
    let mut mutator = GenomeMutator::new(321);
    let x_a = Tensor::from_loaded(vec![1.0, 2.0, 2.0, 1.0], vec![2, 2], false)?;
    let y_a = Tensor::from_loaded(vec![3.0, 3.0], vec![2, 1], false)?;
    let x_b = Tensor::from_loaded(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], false)?;
    let y_b = Tensor::from_loaded(vec![1.0, 1.0], vec![2, 1], false)?;

    let mut params = Vec::new();
    for inst in &genome.instructions {
        if let GenomeInstruction::Linear { weight, bias } = inst {
            params.push(weight.clone());
            if let Some(b) = bias {
                params.push(b.clone());
            }
        }
    }
    let mut optim = SGD::new(params, 0.04);
    let baseline_a = mse(&genome.forward(&x_a)?, &y_a)?.eval()[0];
    let baseline_b = mse(&genome.forward(&x_b)?, &y_b)?.eval()[0];

    full_hybrid_train_loop_with_sleep(
        &mut optim,
        &mut genome,
        &mut mutator,
        HybridTrainConfig {
            epochs: 2,
            steps_per_epoch: 20,
            mutation_every: 5,
            mutation_trials: 6,
        },
        SleepPhaseConfig {
            every_steps: 4,
            dream_steps: 2,
        },
        |g, _, step| {
            // Alternate tasks in wake phase to observe forgetting per task.
            if step % 2 == 0 {
                let pred = g.forward(&x_a)?;
                mse(&pred, &y_a)
            } else {
                let pred = g.forward(&x_b)?;
                mse(&pred, &y_b)
            }
        },
        |g, _, step| {
            let task = if step % 2 == 0 {
                let pred = g.forward(&x_a)?;
                mse(&pred, &y_a)?.eval()[0]
            } else {
                let pred = g.forward(&x_b)?;
                mse(&pred, &y_b)?.eval()[0]
            };
            Ok(task + g.complexity_score() as f32 * 0.0001)
        },
        |g, _, _, dream_idx| {
            let noise = Tensor::from_loaded(
                vec![
                    0.1 + dream_idx as f32 * 0.02,
                    -0.1,
                    0.05,
                    -0.05,
                ],
                vec![2, 2],
                false,
            )?;
            // Pseudo-target: EMA blend between current prediction and damped self target.
            // This keeps dream learning self-consistent while avoiding collapse to exact identity.
            let pred = g.forward(&noise)?;
            let target = Tensor::from_loaded(
                pred.eval().into_iter().map(|v| v * 0.7).collect(),
                pred.shape(),
                false,
            )?;
            let task = mse(&pred, &target)?;
            g.kolmogorov_loss(&task, 0.00005)
        },
        |g, _, _| {
            let mut pruned = Vec::with_capacity(g.instructions.len());
            for inst in &g.instructions {
                let same = matches!(
                    (pruned.last(), inst),
                    (Some(GenomeInstruction::Relu), GenomeInstruction::Relu)
                        | (Some(GenomeInstruction::Tanh), GenomeInstruction::Tanh)
                );
                if !same {
                    pruned.push(inst.clone());
                }
            }
            g.instructions = pruned;
        },
        |m: HybridTrainStepMetrics| {
            if m.step % 5 == 0 {
                println!(
                    "epoch={} step={} loss={:.6} sleep={} dream_loss={:?}",
                    m.epoch, m.step, m.loss, m.sleep_applied, m.dream_loss
                );
            }
        },
    )?;

    let final_pred_a = genome.forward(&x_a)?.eval();
    let final_pred_b = genome.forward(&x_b)?.eval();
    let final_loss_a = mse(&genome.forward(&x_a)?, &y_a)?.eval()[0];
    let final_loss_b = mse(&genome.forward(&x_b)?, &y_b)?.eval()[0];
    let forgetting_delta_a = final_loss_a - baseline_a;
    let forgetting_delta_b = final_loss_b - baseline_b;
    println!(
        "final_pred_a={:?} final_pred_b={:?} final_complexity={} baseline_a={:.6} final_a={:.6} delta_a={:.6} baseline_b={:.6} final_b={:.6} delta_b={:.6}",
        final_pred_a,
        final_pred_b,
        genome.complexity_score()
            ,
        baseline_a,
        final_loss_a,
        forgetting_delta_a,
        baseline_b,
        final_loss_b,
        forgetting_delta_b
    );
    Ok(())
}
