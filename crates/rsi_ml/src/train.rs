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
pub struct HybridTrainStepMetrics {
    pub epoch: usize,
    pub step: usize,
    pub loss: f32,
    pub search_applied: bool,
    pub search_score: Option<f32>,
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
            });
        }
    }
    Ok(())
}
