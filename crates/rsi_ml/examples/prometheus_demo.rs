use std::fs::File;
use std::io::Write;

use rsi_ml::{
    guided_evolutionary_search_step, mse, Genome, GenomeMutator, GuidedSearchConfig, MutationOracle,
    Tensor,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut genome = Genome::from_seed_mlp(2026, 2, 4, 1)?;
    let mut mutator = GenomeMutator::new(99);
    let mut oracle = MutationOracle::new_mlp(0.01, 512);

    let x = Tensor::from_loaded(vec![1.0, 2.0, 2.0, 1.0], vec![2, 2], false)?;
    let y = Tensor::from_loaded(vec![3.0, 3.0], vec![2, 1], false)?;

    let mut csv = File::create("prometheus_logs.csv")?;
    writeln!(
        csv,
        "generation,base_score,candidate_score,predicted_delta,observed_delta,accepted,event,base_complexity,candidate_complexity"
    )?;

    for generation in 0..40usize {
        let (best, outcome) = guided_evolutionary_search_step(
            generation,
            &genome,
            &mut mutator,
            &mut oracle,
            GuidedSearchConfig {
                candidate_pool: 16,
                evaluate_top_k: 4,
                max_evaluate_top_k: 12,
                dynamic_top_k: true,
            },
            |g| {
                let pred = g.forward(&x)?;
                let task = mse(&pred, &y)?.eval()[0];
                Ok(task + g.complexity_score() as f32 * 0.0001)
            },
        )?;

        if outcome.accepted {
            genome = best;
        }

        for log in outcome.logs {
            writeln!(
                csv,
                "{},{},{},{},{},{:?},{:?},{},{}",
                log.generation,
                log.base_score,
                log.candidate_score.unwrap_or(f32::NAN),
                log.predicted_delta,
                log.observed_delta.unwrap_or(f32::NAN),
                log.accepted,
                log.event,
                log.base_complexity,
                log.candidate_complexity
            )?;
        }

        if generation % 10 == 0 {
            println!(
                "gen={} best_score={:.6} accepted={} evaluated={} top_k={} oracle_err={:.6}",
                generation,
                outcome.best_score,
                outcome.accepted,
                outcome.evaluated_candidates,
                outcome.selected_top_k,
                outcome.oracle_ema_abs_error
            );
        }
    }

    let final_pred = genome.forward(&x)?.eval();
    println!(
        "prometheus done final_complexity={} pred={:?} log=prometheus_logs.csv",
        genome.complexity_score(),
        final_pred
    );
    Ok(())
}
