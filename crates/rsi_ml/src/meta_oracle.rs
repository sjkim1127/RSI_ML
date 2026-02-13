use crate::genome::{Genome, GenomeMutator, MutationEvent};
use rsi_ml_core::TensorError;

const FEATURE_DIM: usize = 9;
const MLP_HIDDEN_DIM: usize = 16;

#[derive(Clone, Debug)]
pub struct EvolutionLogEntry {
    pub generation: usize,
    pub base_score: f32,
    pub candidate_score: Option<f32>,
    pub predicted_delta: f32,
    pub observed_delta: Option<f32>,
    pub accepted: bool,
    pub event: MutationEvent,
    pub base_complexity: usize,
    pub candidate_complexity: usize,
}

#[derive(Clone, Debug)]
enum OracleBackend {
    Linear {
        weights: [f32; FEATURE_DIM],
        bias: f32,
    },
    Mlp {
        w1: [[f32; FEATURE_DIM]; MLP_HIDDEN_DIM],
        b1: [f32; MLP_HIDDEN_DIM],
        w2: [f32; MLP_HIDDEN_DIM],
        b2: f32,
    },
}

#[derive(Clone, Debug)]
pub struct MutationOracle {
    backend: OracleBackend,
    pub lr: f32,
    replay: Vec<([f32; FEATURE_DIM], f32)>,
    replay_capacity: usize,
    replay_updates_per_step: usize,
    rand_state: u64,
    ema_abs_error: f32,
}

impl Default for MutationOracle {
    fn default() -> Self {
        Self::new(0.01)
    }
}

impl MutationOracle {
    pub fn new(lr: f32) -> Self {
        Self::new_linear(lr)
    }

    pub fn new_linear(lr: f32) -> Self {
        Self {
            backend: OracleBackend::Linear {
                weights: [0.0; FEATURE_DIM],
                bias: 0.0,
            },
            lr,
            replay: Vec::new(),
            replay_capacity: 0,
            replay_updates_per_step: 0,
            rand_state: 0xC0DEC0DE_u64,
            ema_abs_error: 0.0,
        }
    }

    pub fn new_mlp(lr: f32, replay_capacity: usize) -> Self {
        let mut state = 0xA11CE_u64;
        let mut w1 = [[0.0; FEATURE_DIM]; MLP_HIDDEN_DIM];
        let mut w2 = [0.0; MLP_HIDDEN_DIM];
        for row in &mut w1 {
            for v in row.iter_mut() {
                *v = (rand01(&mut state) - 0.5) * 0.1;
            }
        }
        for v in &mut w2 {
            *v = (rand01(&mut state) - 0.5) * 0.1;
        }

        Self {
            backend: OracleBackend::Mlp {
                w1,
                b1: [0.0; MLP_HIDDEN_DIM],
                w2,
                b2: 0.0,
            },
            lr,
            replay: Vec::new(),
            replay_capacity,
            replay_updates_per_step: 2,
            rand_state: state,
            ema_abs_error: 0.0,
        }
    }

    pub fn ema_abs_error(&self) -> f32 {
        self.ema_abs_error
    }

    pub fn suggested_top_k(&self, pool: usize, min_top_k: usize, max_top_k: usize) -> usize {
        let min_k = min_top_k.max(1).min(pool.max(1));
        let max_k = max_top_k.max(min_k).min(pool.max(1));
        if min_k == max_k {
            return min_k;
        }
        // Higher oracle error => evaluate more real candidates.
        let scale = (self.ema_abs_error / 0.25).clamp(0.0, 1.0);
        let span = (max_k - min_k) as f32;
        min_k + (span * scale).round() as usize
    }

    fn train_single(&mut self, features: &[f32; FEATURE_DIM], target_delta: f32) -> f32 {
        match &mut self.backend {
            OracleBackend::Linear { weights, bias } => {
                let mut pred = *bias;
                for (w, x) in weights.iter().zip(features.iter()) {
                    pred += w * x;
                }
                let err = pred - target_delta;
                for (w, x) in weights.iter_mut().zip(features.iter()) {
                    *w -= self.lr * (2.0 * err * *x);
                }
                *bias -= self.lr * (2.0 * err);
                err
            }
            OracleBackend::Mlp { w1, b1, w2, b2 } => {
                let mut z1 = [0.0_f32; MLP_HIDDEN_DIM];
                let mut h1 = [0.0_f32; MLP_HIDDEN_DIM];
                for i in 0..MLP_HIDDEN_DIM {
                    let mut z = b1[i];
                    for (wj, xj) in w1[i].iter().zip(features.iter()) {
                        z += wj * xj;
                    }
                    z1[i] = z;
                    h1[i] = if z > 0.0 { z } else { 0.0 };
                }

                let mut pred = *b2;
                for i in 0..MLP_HIDDEN_DIM {
                    pred += w2[i] * h1[i];
                }
                let err = pred - target_delta;
                let g = 2.0 * err;
                let w2_old = *w2;

                for i in 0..MLP_HIDDEN_DIM {
                    w2[i] -= self.lr * g * h1[i];
                }
                *b2 -= self.lr * g;

                for i in 0..MLP_HIDDEN_DIM {
                    if z1[i] > 0.0 {
                        let d = g * w2_old[i];
                        for j in 0..FEATURE_DIM {
                            w1[i][j] -= self.lr * d * features[j];
                        }
                        b1[i] -= self.lr * d;
                    }
                }
                err
            }
        }
    }

    fn push_replay(&mut self, features: &[f32; FEATURE_DIM], target_delta: f32) {
        if self.replay_capacity == 0 {
            return;
        }
        if self.replay.len() >= self.replay_capacity {
            self.replay.remove(0);
        }
        self.replay.push((*features, target_delta));
    }

    fn replay_sample(&mut self) -> Option<([f32; FEATURE_DIM], f32)> {
        if self.replay.is_empty() {
            return None;
        }
        self.rand_state = self
            .rand_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        let idx = ((self.rand_state >> 32) as usize) % self.replay.len();
        self.replay.get(idx).copied()
    }

    pub fn train_step(&mut self, features: &[f32; FEATURE_DIM], target_delta: f32) -> f32 {
        let err = self.train_single(features, target_delta);
        self.push_replay(features, target_delta);
        self.ema_abs_error = 0.95 * self.ema_abs_error + 0.05 * err.abs();

        for _ in 0..self.replay_updates_per_step {
            if let Some((fx, ty)) = self.replay_sample() {
                let _ = self.train_single(&fx, ty);
            } else {
                break;
            }
        }
        err * err
    }

    pub fn predict_delta(&self, features: &[f32; FEATURE_DIM]) -> f32 {
        match &self.backend {
            OracleBackend::Linear { weights, bias } => {
                let mut out = *bias;
                for (w, x) in weights.iter().zip(features.iter()) {
                    out += w * x;
                }
                out
            }
            OracleBackend::Mlp { w1, b1, w2, b2 } => {
                let mut h = [0.0_f32; MLP_HIDDEN_DIM];
                for i in 0..MLP_HIDDEN_DIM {
                    let mut z = b1[i];
                    for (wj, xj) in w1[i].iter().zip(features.iter()) {
                        z += wj * xj;
                    }
                    h[i] = if z > 0.0 { z } else { 0.0 };
                }
                let mut out = *b2;
                for i in 0..MLP_HIDDEN_DIM {
                    out += w2[i] * h[i];
                }
                out
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GuidedSearchConfig {
    pub candidate_pool: usize,
    pub evaluate_top_k: usize,
    pub max_evaluate_top_k: usize,
    pub dynamic_top_k: bool,
}

#[derive(Clone, Debug)]
pub struct GuidedSearchOutcome {
    pub accepted: bool,
    pub best_score: f32,
    pub evaluated_candidates: usize,
    pub selected_top_k: usize,
    pub oracle_ema_abs_error: f32,
    pub logs: Vec<EvolutionLogEntry>,
}

pub fn featurize_mutation(
    before: &Genome,
    after: &Genome,
    event: MutationEvent,
) -> [f32; FEATURE_DIM] {
    let base_c = before.complexity_score() as f32;
    let cand_c = after.complexity_score() as f32;
    let delta_c = cand_c - base_c;
    let mut v = [0.0_f32; FEATURE_DIM];
    v[0] = base_c / 1000.0;
    v[1] = cand_c / 1000.0;
    v[2] = delta_c / 1000.0;
    v[3] = matches!(event, MutationEvent::ToggleActivation { .. }) as u8 as f32;
    v[4] = matches!(event, MutationEvent::InsertRelu { .. }) as u8 as f32;
    v[5] = matches!(event, MutationEvent::InsertTanh { .. }) as u8 as f32;
    v[6] = matches!(event, MutationEvent::RemoveActivation { .. }) as u8 as f32;
    v[7] = matches!(event, MutationEvent::Noop) as u8 as f32;
    v[8] = before.instructions.len() as f32 / 128.0;
    v
}

pub fn guided_evolutionary_search_step<F>(
    generation: usize,
    base: &Genome,
    mutator: &mut GenomeMutator,
    oracle: &mut MutationOracle,
    config: GuidedSearchConfig,
    mut score_fn: F,
) -> Result<(Genome, GuidedSearchOutcome), TensorError>
where
    F: FnMut(&Genome) -> Result<f32, TensorError>,
{
    let pool = config.candidate_pool.max(1);
    let static_top_k = config.evaluate_top_k.max(1).min(pool);
    let top_k = if config.dynamic_top_k {
        oracle.suggested_top_k(pool, static_top_k, config.max_evaluate_top_k.max(static_top_k))
    } else {
        static_top_k
    };

    let base_score = score_fn(base)?;
    let base_complexity = base.complexity_score();

    let mut candidates = Vec::with_capacity(pool);
    for _ in 0..pool {
        let mut candidate = base.clone();
        let event = mutator.mutate_once(&mut candidate);
        let features = featurize_mutation(base, &candidate, event);
        let predicted_delta = oracle.predict_delta(&features);
        candidates.push((candidate, event, features, predicted_delta));
    }

    candidates.sort_by(|a, b| {
        a.3.partial_cmp(&b.3)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut best_genome = base.clone();
    let mut best_score = base_score;
    let mut accepted = false;
    let mut logs = Vec::new();

    // First log all non-evaluated candidates.
    for (candidate, event, _, predicted_delta) in candidates.iter().skip(top_k) {
        logs.push(EvolutionLogEntry {
            generation,
            base_score,
            candidate_score: None,
            predicted_delta: *predicted_delta,
            observed_delta: None,
            accepted: false,
            event: *event,
            base_complexity,
            candidate_complexity: candidate.complexity_score(),
        });
    }

    // Evaluate only top-k candidates predicted by oracle.
    for (candidate, event, features, predicted_delta) in candidates.into_iter().take(top_k) {
        let candidate_score = score_fn(&candidate)?;
        let observed_delta = candidate_score - base_score;
        let _ = oracle.train_step(&features, observed_delta);

        let mut this_accepted = false;
        if candidate_score < best_score {
            best_score = candidate_score;
            best_genome = candidate.clone();
            accepted = true;
            this_accepted = true;
        }

        logs.push(EvolutionLogEntry {
            generation,
            base_score,
            candidate_score: Some(candidate_score),
            predicted_delta,
            observed_delta: Some(observed_delta),
            accepted: this_accepted,
            event,
            base_complexity,
            candidate_complexity: candidate.complexity_score(),
        });
    }

    Ok((
        best_genome,
        GuidedSearchOutcome {
            accepted,
            best_score,
            evaluated_candidates: top_k,
            selected_top_k: top_k,
            oracle_ema_abs_error: oracle.ema_abs_error(),
            logs,
        },
    ))
}

fn rand01(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    let x = (*state >> 32) as u32;
    (x as f32) / (u32::MAX as f32)
}
