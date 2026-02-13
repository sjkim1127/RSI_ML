use rsi_ml::{
    full_hybrid_train_loop, mse, Genome, GenomeInstruction, GenomeMutator, HybridTrainConfig,
    SGD, Tensor,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut genome = Genome::from_seed_mlp(1234, 2, 6, 1)?;
    let mut mutator = GenomeMutator::new(999);

    let x = Tensor::from_loaded(vec![1.0, 2.0, 2.0, 1.0], vec![2, 2], false)?;
    let y = Tensor::from_loaded(vec![3.0, 3.0], vec![2, 1], false)?;
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

    let base_complexity = genome.complexity_score();
    full_hybrid_train_loop(
        &mut optim,
        &mut genome,
        &mut mutator,
        HybridTrainConfig {
            epochs: 2,
            steps_per_epoch: 30,
            mutation_every: 5,
            mutation_trials: 8,
        },
        |g, _, _| {
            let pred = g.forward(&x)?;
            mse(&pred, &y)
        },
        |g, _, _| {
            let pred = g.forward(&x)?;
            let task = mse(&pred, &y)?.eval()[0];
            Ok(task + g.complexity_score() as f32 * 0.0001)
        },
        |m| {
            if m.step % 10 == 0 {
                println!(
                    "epoch={} step={} loss={:.6} search_applied={} search_score={:?}",
                    m.epoch, m.step, m.loss, m.search_applied, m.search_score
                );
            }
        },
    )?;
    let final_pred = genome.forward(&x)?.eval();
    println!(
        "base_complexity={} final_complexity={} pred={:?}",
        base_complexity,
        genome.complexity_score(),
        final_pred
    );
    Ok(())
}
