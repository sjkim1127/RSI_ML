use std::cell::RefCell;
use std::collections::HashMap;

use rsi_ml_core::Tensor;

pub trait Optimizer {
    fn step(&mut self);
    fn zero_grad(&self);
}

pub struct Sgd {
    params: Vec<Tensor>,
    lr: f32,
}

impl Sgd {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        Self { params, lr }
    }
}

pub type SGD = Sgd;

impl Optimizer for Sgd {
    fn step(&mut self) {
        for param in &self.params {
            param.apply_gradient_descent(self.lr);
        }
    }

    fn zero_grad(&self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
}

pub struct Adam {
    params: Vec<Tensor>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    step_count: usize,
    state: RefCell<HashMap<usize, AdamState>>,
}

#[derive(Clone, Debug)]
struct AdamState {
    m: Vec<f32>,
    v: Vec<f32>,
}

impl Adam {
    pub fn new(params: Vec<Tensor>, lr: f32, beta1: f32, beta2: f32, eps: f32) -> Self {
        Self {
            params,
            lr,
            beta1,
            beta2,
            eps,
            step_count: 0,
            state: RefCell::new(HashMap::new()),
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self) {
        self.step_count += 1;
        let t = self.step_count as i32;
        let b1_correction = 1.0 - self.beta1.powi(t);
        let b2_correction = 1.0 - self.beta2.powi(t);

        for param in &self.params {
            let Some(grad) = param.grad() else {
                continue;
            };
            let key = param.id();
            let len = param.parameter_len();
            let mut state = self.state.borrow_mut();
            let entry = state.entry(key).or_insert_with(|| AdamState {
                m: vec![0.0; len],
                v: vec![0.0; len],
            });

            let mut update = vec![0.0; len];
            for (i, g) in grad.iter().enumerate() {
                entry.m[i] = self.beta1 * entry.m[i] + (1.0 - self.beta1) * g;
                entry.v[i] = self.beta2 * entry.v[i] + (1.0 - self.beta2) * g * g;

                let m_hat = entry.m[i] / b1_correction;
                let v_hat = entry.v[i] / b2_correction;
                update[i] = self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }

            let _ = param.apply_update(&update);
        }
    }

    fn zero_grad(&self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
}
