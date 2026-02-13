use rsi_ml_core::{Tensor, TensorError};
use rsi_ml_optim::Optimizer;
use crate::genome::{evolutionary_search_step, Genome, GenomeMutator};

#[derive(Clone, Copy, Debug)]
pub struct TrainLoopConfig {
    pub epochs: usize,
    pub steps_per_epoch: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct TrainStepMetrics {
    pub epoch: usize,
    pub step: usize,
    pub loss: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct HybridTrainConfig {
    pub epochs: usize,
    pub steps_per_epoch: usize,
    pub mutation_every: usize,
    pub mutation_trials: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct SleepPhaseConfig {
    pub every_steps: usize,
    pub dream_steps: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct EvolutionTrainConfig {
    pub generations: usize,
    pub trials_per_generation: usize,
    pub growth_every: usize,
    pub lookahead_steps: usize,
    pub lookahead_alpha: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct HybridTrainStepMetrics {
    pub epoch: usize,
    pub step: usize,
    pub loss: f32,
    pub search_applied: bool,
    pub search_score: Option<f32>,
    pub sleep_applied: bool,
    pub dream_loss: Option<f32>,
}

#[derive(Clone, Copy, Debug)]
pub struct EvolutionTrainStepMetrics {
    pub generation: usize,
    pub base_score: f32,
    pub future_score: Option<f32>,
    pub effective_score: f32,
    pub accepted: bool,
    pub growth_applied: bool,
    pub complexity: usize,
}

pub fn full_train_loop<O, F, C>(
    optimizer: &mut O,
    config: TrainLoopConfig,
    mut make_loss: F,
    mut on_step: C,
) -> Result<(), TensorError>
where
    O: Optimizer,
    F: FnMut(usize, usize) -> Result<Tensor, TensorError>,
    C: FnMut(TrainStepMetrics),
{
    for epoch in 0..config.epochs {
        for step in 0..config.steps_per_epoch {
            let loss_tensor = make_loss(epoch, step)?;
            let loss_value = loss_tensor.eval()[0];
            loss_tensor.backward();
            optimizer.step();
            optimizer.zero_grad();
            on_step(TrainStepMetrics {
                epoch,
                step,
                loss: loss_value,
            });
        }
    }
    Ok(())
}

pub fn full_hybrid_train_loop<O, FG, FS, C>(
    optimizer: &mut O,
    genome: &mut Genome,
    mutator: &mut GenomeMutator,
    config: HybridTrainConfig,
    mut make_gradient_loss: FG,
    mut score_genome: FS,
    mut on_step: C,
) -> Result<(), TensorError>
where
    O: Optimizer,
    FG: FnMut(&Genome, usize, usize) -> Result<Tensor, TensorError>,
    FS: FnMut(&Genome, usize, usize) -> Result<f32, TensorError>,
    C: FnMut(HybridTrainStepMetrics),
{
    for epoch in 0..config.epochs {
        for step in 0..config.steps_per_epoch {
            let loss_tensor = make_gradient_loss(genome, epoch, step)?;
            let loss_value = loss_tensor.eval()[0];
            loss_tensor.backward();
            optimizer.step();
            optimizer.zero_grad();

            let mut search_applied = false;
            let mut search_score = None;
            if config.mutation_every > 0
                && config.mutation_trials > 0
                && (step + 1) % config.mutation_every == 0
            {
                let base_score = score_genome(genome, epoch, step)?;
                let (best, best_score) = evolutionary_search_step(
                    genome,
                    mutator,
                    config.mutation_trials,
                    |g| score_genome(g, epoch, step),
                )?;
                if best_score <= base_score {
                    *genome = best;
                    search_applied = true;
                    search_score = Some(best_score);
                }
            }

            on_step(HybridTrainStepMetrics {
                epoch,
                step,
                loss: loss_value,
                search_applied,
                search_score,
                sleep_applied: false,
                dream_loss: None,
            });
        }
    }
    Ok(())
}

pub fn full_hybrid_train_loop_with_sleep<O, FG, FS, FD, FP, C>(
    optimizer: &mut O,
    genome: &mut Genome,
    mutator: &mut GenomeMutator,
    config: HybridTrainConfig,
    sleep: SleepPhaseConfig,
    mut make_gradient_loss: FG,
    mut score_genome: FS,
    mut make_dream_loss: FD,
    mut prune_genome: FP,
    mut on_step: C,
) -> Result<(), TensorError>
where
    O: Optimizer,
    FG: FnMut(&Genome, usize, usize) -> Result<Tensor, TensorError>,
    FS: FnMut(&Genome, usize, usize) -> Result<f32, TensorError>,
    FD: FnMut(&Genome, usize, usize, usize) -> Result<Tensor, TensorError>,
    FP: FnMut(&mut Genome, usize, usize),
    C: FnMut(HybridTrainStepMetrics),
{
    for epoch in 0..config.epochs {
        for step in 0..config.steps_per_epoch {
            let loss_tensor = make_gradient_loss(genome, epoch, step)?;
            let loss_value = loss_tensor.eval()[0];
            loss_tensor.backward();
            optimizer.step();
            optimizer.zero_grad();

            let mut search_applied = false;
            let mut search_score = None;
            if config.mutation_every > 0
                && config.mutation_trials > 0
                && (step + 1) % config.mutation_every == 0
            {
                let base_score = score_genome(genome, epoch, step)?;
                let (best, best_score) = evolutionary_search_step(
                    genome,
                    mutator,
                    config.mutation_trials,
                    |g| score_genome(g, epoch, step),
                )?;
                if best_score <= base_score {
                    *genome = best;
                    search_applied = true;
                    search_score = Some(best_score);
                }
            }

            let mut sleep_applied = false;
            let mut dream_loss = None;
            if sleep.every_steps > 0
                && sleep.dream_steps > 0
                && (step + 1) % sleep.every_steps == 0
            {
                sleep_applied = true;
                for dream_idx in 0..sleep.dream_steps {
                    let dream = make_dream_loss(genome, epoch, step, dream_idx)?;
                    let dl = dream.eval()[0];
                    dream.backward();
                    optimizer.step();
                    optimizer.zero_grad();
                    dream_loss = Some(dl);
                }
                prune_genome(genome, epoch, step);
            }

            on_step(HybridTrainStepMetrics {
                epoch,
                step,
                loss: loss_value,
                search_applied,
                search_score,
                sleep_applied,
                dream_loss,
            });
        }
    }
    Ok(())
}

pub fn full_evolution_train_loop<FS, FG, C>(
    genome: &mut Genome,
    mutator: &mut GenomeMutator,
    config: EvolutionTrainConfig,
    mut score_genome: FS,
    mut grow_genome: FG,
    mut on_step: C,
) -> Result<(), TensorError>
where
    FS: FnMut(&Genome, usize) -> Result<f32, TensorError>,
    FG: FnMut(&mut Genome, usize) -> Result<bool, TensorError>,
    C: FnMut(EvolutionTrainStepMetrics),
{
    let lookahead_enabled = config.lookahead_steps > 0 && config.lookahead_alpha != 0.0;

    for generation in 0..config.generations {
        let base_score = score_genome(genome, generation)?;
        let base_future_score = if lookahead_enabled {
            Some(score_genome(genome, generation + config.lookahead_steps)?)
        } else {
            None
        };
        let mut base_effective_score = base_score;
        if let Some(future) = base_future_score {
            let drift = future - base_score;
            base_effective_score += config.lookahead_alpha * drift;
        }

        let mut accepted = false;
        let mut final_base_score = base_score;
        let mut final_future_score = base_future_score;
        let mut final_effective_score = base_effective_score;

        if config.trials_per_generation > 0 {
            let (best, best_score) = evolutionary_search_step(
                genome,
                mutator,
                config.trials_per_generation,
                |g| {
                    let now = score_genome(g, generation)?;
                    if lookahead_enabled {
                        let future = score_genome(g, generation + config.lookahead_steps)?;
                        let drift = future - now;
                        Ok(now + config.lookahead_alpha * drift)
                    } else {
                        Ok(now)
                    }
                },
            )?;
            if best_score <= base_effective_score {
                *genome = best;
                accepted = true;
                final_base_score = score_genome(genome, generation)?;
                final_future_score = if lookahead_enabled {
                    Some(score_genome(genome, generation + config.lookahead_steps)?)
                } else {
                    None
                };
                final_effective_score = best_score;
            }
        }

        let mut growth_applied = false;
        if config.growth_every > 0 && (generation + 1) % config.growth_every == 0 {
            growth_applied = grow_genome(genome, generation)?;
            if growth_applied {
                final_base_score = score_genome(genome, generation)?;
                final_future_score = if lookahead_enabled {
                    Some(score_genome(genome, generation + config.lookahead_steps)?)
                } else {
                    None
                };
                final_effective_score = if let Some(future) = final_future_score {
                    let drift = future - final_base_score;
                    final_base_score + config.lookahead_alpha * drift
                } else {
                    final_base_score
                };
            }
        }

        on_step(EvolutionTrainStepMetrics {
            generation,
            base_score: final_base_score,
            future_score: final_future_score,
            effective_score: final_effective_score,
            accepted,
            growth_applied,
            complexity: genome.complexity_score(),
        });
    }
    Ok(())
}
