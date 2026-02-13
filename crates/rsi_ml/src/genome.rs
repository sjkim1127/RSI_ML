use rsi_ml_core::{Tensor, TensorError};
use rsi_ml_ops::{layer_norm, linear, transformer_block, TransformerBlockWeights};

#[derive(Clone)]
pub struct TransformerBlockWeightsOwned {
    pub w_q: Tensor,
    pub w_k: Tensor,
    pub w_v: Tensor,
    pub w_o: Tensor,
    pub w_ff1: Tensor,
    pub b_ff1: Option<Tensor>,
    pub w_ff2: Tensor,
    pub b_ff2: Option<Tensor>,
}

impl TransformerBlockWeightsOwned {
    pub fn as_borrowed(&self) -> TransformerBlockWeights<'_> {
        TransformerBlockWeights {
            w_q: &self.w_q,
            w_k: &self.w_k,
            w_v: &self.w_v,
            w_o: &self.w_o,
            w_ff1: &self.w_ff1,
            b_ff1: self.b_ff1.as_ref(),
            w_ff2: &self.w_ff2,
            b_ff2: self.b_ff2.as_ref(),
        }
    }

    pub fn parameter_count(&self) -> usize {
        let mut total = self.w_q.parameter_len()
            + self.w_k.parameter_len()
            + self.w_v.parameter_len()
            + self.w_o.parameter_len()
            + self.w_ff1.parameter_len()
            + self.w_ff2.parameter_len();
        if let Some(b) = &self.b_ff1 {
            total += b.parameter_len();
        }
        if let Some(b) = &self.b_ff2 {
            total += b.parameter_len();
        }
        total
    }
}

#[derive(Clone)]
pub enum GenomeInstruction {
    Linear { weight: Tensor, bias: Option<Tensor> },
    Relu,
    Tanh,
    LayerNorm { eps: f32 },
    TransformerBlock {
        weights: TransformerBlockWeightsOwned,
        num_heads: usize,
        eps: f32,
    },
}

#[derive(Clone, Default)]
pub struct Genome {
    pub instructions: Vec<GenomeInstruction>,
}

impl Genome {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
        }
    }

    pub fn with_instructions(instructions: Vec<GenomeInstruction>) -> Self {
        Self { instructions }
    }

    pub fn push(&mut self, inst: GenomeInstruction) {
        self.instructions.push(inst);
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, TensorError> {
        let mut x = input.clone();
        for inst in &self.instructions {
            x = match inst {
                GenomeInstruction::Linear { weight, bias } => linear(&x, weight, bias.as_ref())?,
                GenomeInstruction::Relu => x.relu(),
                GenomeInstruction::Tanh => x.tanh(),
                GenomeInstruction::LayerNorm { eps } => layer_norm(&x, None, None, *eps)?,
                GenomeInstruction::TransformerBlock {
                    weights,
                    num_heads,
                    eps,
                } => transformer_block(&x, weights.as_borrowed(), *num_heads, *eps)?,
            };
        }
        Ok(x)
    }

    pub fn complexity_score(&self) -> usize {
        let mut score = self.instructions.len();
        for inst in &self.instructions {
            score += match inst {
                GenomeInstruction::Linear { weight, bias } => {
                    let mut c = weight.parameter_len();
                    if let Some(b) = bias {
                        c += b.parameter_len();
                    }
                    c
                }
                GenomeInstruction::Relu => 1,
                GenomeInstruction::Tanh => 1,
                GenomeInstruction::LayerNorm { .. } => 2,
                GenomeInstruction::TransformerBlock { weights, .. } => weights.parameter_count(),
            };
        }
        score
    }

    pub fn kolmogorov_loss(&self, task_loss: &Tensor, lambda: f32) -> Result<Tensor, TensorError> {
        if task_loss.shape() != vec![1] {
            return Err(TensorError::InvalidShape {
                shape: task_loss.shape(),
                data_len: 1,
            });
        }
        let penalty_value = self.complexity_score() as f32 * lambda;
        let penalty = Tensor::from_loaded(vec![penalty_value], vec![1], false)?;
        task_loss.add(&penalty)
    }

    pub fn mutate_toggle_activation(&mut self, idx: usize) -> bool {
        if idx >= self.instructions.len() {
            return false;
        }
        match &self.instructions[idx] {
            GenomeInstruction::Relu => {
                self.instructions[idx] = GenomeInstruction::Tanh;
                true
            }
            GenomeInstruction::Tanh => {
                self.instructions[idx] = GenomeInstruction::Relu;
                true
            }
            _ => false,
        }
    }

    pub fn from_seed_mlp(
        seed: u64,
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
    ) -> Result<Self, TensorError> {
        let mut s = seed;
        let w1 = Tensor::from_loaded(
            seeded_values(&mut s, input_dim * hidden_dim),
            vec![input_dim, hidden_dim],
            true,
        )?;
        let b1 = Tensor::from_loaded(seeded_values(&mut s, hidden_dim), vec![1, hidden_dim], true)?;
        let w2 = Tensor::from_loaded(
            seeded_values(&mut s, hidden_dim * output_dim),
            vec![hidden_dim, output_dim],
            true,
        )?;
        let b2 = Tensor::from_loaded(seeded_values(&mut s, output_dim), vec![1, output_dim], true)?;

        Ok(Self::with_instructions(vec![
            GenomeInstruction::Linear {
                weight: w1,
                bias: Some(b1),
            },
            GenomeInstruction::Relu,
            GenomeInstruction::Linear {
                weight: w2,
                bias: Some(b2),
            },
        ]))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MutationEvent {
    ToggleActivation { idx: usize },
    InsertRelu { idx: usize },
    InsertTanh { idx: usize },
    RemoveActivation { idx: usize },
    Noop,
}

pub struct GenomeMutator {
    state: u64,
}

impl GenomeMutator {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn mutate_once(&mut self, genome: &mut Genome) -> MutationEvent {
        if genome.instructions.is_empty() {
            return MutationEvent::Noop;
        }

        let action = self.rand_bounded(4);
        match action {
            0 => self.toggle_activation(genome),
            1 => self.insert_activation(genome, true),
            2 => self.insert_activation(genome, false),
            _ => self.remove_activation(genome),
        }
    }

    fn toggle_activation(&mut self, genome: &mut Genome) -> MutationEvent {
        let indices = activation_indices(genome);
        if indices.is_empty() {
            return MutationEvent::Noop;
        }
        let idx = indices[self.rand_bounded(indices.len())];
        if genome.mutate_toggle_activation(idx) {
            MutationEvent::ToggleActivation { idx }
        } else {
            MutationEvent::Noop
        }
    }

    fn insert_activation(&mut self, genome: &mut Genome, relu: bool) -> MutationEvent {
        let idx = self.rand_bounded(genome.instructions.len() + 1);
        let inst = if relu {
            GenomeInstruction::Relu
        } else {
            GenomeInstruction::Tanh
        };
        genome.instructions.insert(idx, inst);
        if relu {
            MutationEvent::InsertRelu { idx }
        } else {
            MutationEvent::InsertTanh { idx }
        }
    }

    fn remove_activation(&mut self, genome: &mut Genome) -> MutationEvent {
        let indices = activation_indices(genome);
        if indices.is_empty() {
            return MutationEvent::Noop;
        }
        let idx = indices[self.rand_bounded(indices.len())];
        genome.instructions.remove(idx);
        MutationEvent::RemoveActivation { idx }
    }

    fn rand_bounded(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.state >> 32) as usize) % max
    }
}

pub fn evolutionary_search_step<F>(
    base: &Genome,
    mutator: &mut GenomeMutator,
    trials: usize,
    mut score_fn: F,
) -> Result<(Genome, f32), TensorError>
where
    F: FnMut(&Genome) -> Result<f32, TensorError>,
{
    let mut best = base.clone();
    let mut best_score = score_fn(&best)?;

    for _ in 0..trials {
        let mut candidate = base.clone();
        let _ = mutator.mutate_once(&mut candidate);
        let score = score_fn(&candidate)?;
        if score < best_score {
            best = candidate;
            best_score = score;
        }
    }

    Ok((best, best_score))
}

fn activation_indices(genome: &Genome) -> Vec<usize> {
    let mut out = Vec::new();
    for (idx, inst) in genome.instructions.iter().enumerate() {
        if matches!(inst, GenomeInstruction::Relu | GenomeInstruction::Tanh) {
            out.push(idx);
        }
    }
    out
}

fn seeded_values(seed: &mut u64, len: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = (*seed >> 32) as u32;
        let centered = (x as f32 / u32::MAX as f32) * 2.0 - 1.0;
        out.push(centered * 0.05);
    }
    out
}
