use std::env;
use std::fs::File;
use std::io::Write;
use rsi_ml::{
    full_evolution_train_loop, mse, EvolutionTrainConfig, Genome, GenomeInstruction, GenomeMutator,
    Tensor,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let use_lookahead = args.get(1).map(|s| s == "lookahead").unwrap_or(true);

    let mut genome = Genome::from_seed_mlp(1234, 2, 2, 1)?;
    let mut mutator = GenomeMutator::new(999);

    let x = Tensor::from_loaded(vec![1.0, 2.0, 2.0, 1.0], vec![2, 2], false)?;
    let y = Tensor::from_loaded(vec![3.0, 3.0], vec![2, 1], false)?;

    let mut csv = File::create("genome_search_metrics.csv")?;
    writeln!(
        csv,
        "generation,base_score,future_score,effective_score,accepted,growth_applied,complexity"
    )?;

    let base_complexity = genome.complexity_score();
    full_evolution_train_loop(
        &mut genome,
        &mut mutator,
        EvolutionTrainConfig {
            generations: 60,
            trials_per_generation: 12,
            growth_every: 15,
            lookahead_steps: if use_lookahead { 3 } else { 0 },
            lookahead_alpha: if use_lookahead { 0.5 } else { 0.0 },
        },
        |g, _generation| {
            let pred = g.forward(&x)?;
            let task = mse(&pred, &y)?.eval()[0];
            Ok(task + g.complexity_score() as f32 * 0.0001)
        },
        |g, generation| {
            // Constructive growth: periodically add a small nonlinearity.
            let insert_at = g.instructions.len().saturating_sub(1);
            if generation % 2 == 0 {
                g.instructions.insert(insert_at, GenomeInstruction::Relu);
            } else {
                g.instructions.insert(insert_at, GenomeInstruction::Tanh);
            }
            Ok(true)
        },
        |m| {
            let _ = writeln!(
                csv,
                "{},{},{},{},{},{},{}",
                m.generation,
                m.base_score,
                m.future_score.unwrap_or(f32::NAN),
                m.effective_score,
                m.accepted,
                m.growth_applied,
                m.complexity
            );
            if m.generation % 10 == 0 {
                println!(
                    "generation={} base={:.6} future={:?} effective={:.6} accepted={} growth={} complexity={}",
                    m.generation,
                    m.base_score,
                    m.future_score,
                    m.effective_score,
                    m.accepted,
                    m.growth_applied,
                    m.complexity
                );
            }
        },
    )?;
    let final_pred = genome.forward(&x)?.eval();
    println!(
        "mode=evolution_only use_lookahead={} base_complexity={} final_complexity={} pred={:?} csv=genome_search_metrics.csv",
        use_lookahead,
        base_complexity,
        genome.complexity_score(),
        final_pred
    );
    Ok(())
}
