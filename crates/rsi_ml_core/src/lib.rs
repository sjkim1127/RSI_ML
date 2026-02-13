use std::cell::RefCell;
use std::collections::HashSet;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::rc::Rc;
use std::sync::OnceLock;
use std::time::Instant;
use rayon::prelude::*;

pub type GeneratorFn = fn(u64, usize) -> f32;

#[derive(Debug, Clone)]
pub enum TensorError {
    ShapeMismatch {
        lhs: Vec<usize>,
        rhs: Vec<usize>,
        op: &'static str,
    },
    InvalidShape {
        shape: Vec<usize>,
        data_len: usize,
    },
    EmptyReduction,
}

impl Display for TensorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::ShapeMismatch { lhs, rhs, op } => {
                write!(
                    f,
                    "shape mismatch in {op}: left={lhs:?}, right={rhs:?}"
                )
            }
            TensorError::InvalidShape { shape, data_len } => {
                write!(f, "invalid shape {shape:?} for data length {data_len}")
            }
            TensorError::EmptyReduction => write!(f, "cannot reduce empty tensor"),
        }
    }
}

impl Error for TensorError {}

#[derive(Clone)]
pub enum TensorData {
    Loaded(Vec<f32>),
    Procedural {
        seed: u64,
        shape: Vec<usize>,
        generator_func: GeneratorFn,
    },
    Expression,
}

#[derive(Clone, Copy)]
enum Op {
    None,
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    Reshape,
    Transpose2D,
    RepeatCols(usize),
    RepeatRows(usize),
    Sum,
    SumAxis(usize),
    Exp,
    Log,
    Sqrt,
    Softmax(usize),
    LogSoftmax(usize),
    Relu,
    Tanh,
    Abs,
    Huber(f32),
}

#[derive(Clone)]
pub struct Tensor {
    node: Rc<RefCell<Node>>,
}

#[derive(Clone)]
struct Node {
    data: TensorData,
    shape: Vec<usize>,
    requires_grad: bool,
    op: Op,
    parents: Vec<Tensor>,
    value_cache: Option<Vec<f32>>,
    grad: Option<Vec<f32>>,
}

impl Tensor {
    pub fn from_loaded(
        data: Vec<f32>,
        shape: Vec<usize>,
        requires_grad: bool,
    ) -> Result<Self, TensorError> {
        if numel(&shape) != data.len() {
            return Err(TensorError::InvalidShape {
                shape,
                data_len: data.len(),
            });
        }

        Ok(Self {
            node: Rc::new(RefCell::new(Node {
                data: TensorData::Loaded(data),
                shape,
                requires_grad,
                op: Op::None,
                parents: vec![],
                value_cache: None,
                grad: None,
            })),
        })
    }

    pub fn procedural(
        seed: u64,
        shape: Vec<usize>,
        generator_func: GeneratorFn,
        requires_grad: bool,
    ) -> Self {
        Self {
            node: Rc::new(RefCell::new(Node {
                data: TensorData::Procedural {
                    seed,
                    shape: shape.clone(),
                    generator_func,
                },
                shape,
                requires_grad,
                op: Op::None,
                parents: vec![],
                value_cache: None,
                grad: None,
            })),
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        self.node.borrow().shape.clone()
    }

    pub fn requires_grad(&self) -> bool {
        self.node.borrow().requires_grad
    }

    pub fn zero_grad(&self) {
        self.node.borrow_mut().grad = None;
    }

    pub fn grad(&self) -> Option<Vec<f32>> {
        self.node.borrow().grad.clone()
    }

    pub fn id(&self) -> usize {
        Rc::as_ptr(&self.node) as usize
    }

    pub fn parameter_len(&self) -> usize {
        numel(&self.shape())
    }

    pub fn apply_gradient_descent(&self, lr: f32) {
        let mut node = self.node.borrow_mut();
        let grad = node.grad.clone();
        if let (Some(grad), TensorData::Loaded(data)) = (grad, &mut node.data) {
            for (w, g) in data.iter_mut().zip(grad.iter()) {
                *w -= lr * g;
            }
            node.value_cache = None;
        }
    }

    pub fn apply_update(&self, update: &[f32]) -> Result<(), TensorError> {
        let mut node = self.node.borrow_mut();
        if let TensorData::Loaded(data) = &mut node.data {
            if data.len() != update.len() {
                return Err(TensorError::InvalidShape {
                    shape: node.shape.clone(),
                    data_len: update.len(),
                });
            }
            for (w, u) in data.iter_mut().zip(update.iter()) {
                *w -= u;
            }
            node.value_cache = None;
        }
        Ok(())
    }

    pub fn eval(&self) -> Vec<f32> {
        self.materialize()
    }

    pub fn add(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        let lhs_shape = self.shape();
        let rhs_shape = other.shape();
        if lhs_shape != rhs_shape {
            return Err(TensorError::ShapeMismatch {
                lhs: lhs_shape,
                rhs: rhs_shape,
                op: "add",
            });
        }

        Ok(Tensor {
            node: Rc::new(RefCell::new(Node {
                data: TensorData::Expression,
                shape: self.shape(),
                requires_grad: self.requires_grad() || other.requires_grad(),
                op: Op::Add,
                parents: vec![self.clone(), other.clone()],
                value_cache: None,
                grad: None,
            })),
        })
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        let lhs_shape = self.shape();
        let rhs_shape = other.shape();
        if lhs_shape != rhs_shape {
            return Err(TensorError::ShapeMismatch {
                lhs: lhs_shape,
                rhs: rhs_shape,
                op: "mul",
            });
        }

        Ok(Tensor {
            node: Rc::new(RefCell::new(Node {
                data: TensorData::Expression,
                shape: self.shape(),
                requires_grad: self.requires_grad() || other.requires_grad(),
                op: Op::Mul,
                parents: vec![self.clone(), other.clone()],
                value_cache: None,
                grad: None,
            })),
        })
    }

    pub fn div(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        let lhs_shape = self.shape();
        let rhs_shape = other.shape();
        if lhs_shape != rhs_shape {
            return Err(TensorError::ShapeMismatch {
                lhs: lhs_shape,
                rhs: rhs_shape,
                op: "div",
            });
        }

        Ok(Tensor {
            node: Rc::new(RefCell::new(Node {
                data: TensorData::Expression,
                shape: self.shape(),
                requires_grad: self.requires_grad() || other.requires_grad(),
                op: Op::Div,
                parents: vec![self.clone(), other.clone()],
                value_cache: None,
                grad: None,
            })),
        })
    }

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        let lhs_shape = self.shape();
        let rhs_shape = other.shape();

        if lhs_shape.len() != 2 {
            return Err(TensorError::InvalidShape {
                shape: lhs_shape,
                data_len: 0,
            });
        }
        if rhs_shape.len() != 2 {
            return Err(TensorError::InvalidShape {
                shape: rhs_shape,
                data_len: 0,
            });
        }
        if lhs_shape[1] != rhs_shape[0] {
            return Err(TensorError::ShapeMismatch {
                lhs: self.shape(),
                rhs: other.shape(),
                op: "matmul",
            });
        }

        Ok(Tensor {
            node: Rc::new(RefCell::new(Node {
                data: TensorData::Expression,
                shape: vec![lhs_shape[0], rhs_shape[1]],
                requires_grad: self.requires_grad() || other.requires_grad(),
                op: Op::MatMul,
                parents: vec![self.clone(), other.clone()],
                value_cache: None,
                grad: None,
            })),
        })
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Tensor, TensorError> {
        let current_len = numel(&self.shape());
        let new_len = numel(&new_shape);
        if current_len != new_len {
            return Err(TensorError::InvalidShape {
                shape: new_shape,
                data_len: current_len,
            });
        }

        Ok(Tensor {
            node: Rc::new(RefCell::new(Node {
                data: TensorData::Expression,
                shape: new_shape,
                requires_grad: self.requires_grad(),
                op: Op::Reshape,
                parents: vec![self.clone()],
                value_cache: None,
                grad: None,
            })),
        })
    }

    pub fn transpose2d(&self) -> Result<Tensor, TensorError> {
        let shape = self.shape();
        if shape.len() != 2 {
            return Err(TensorError::InvalidShape { shape, data_len: 0 });
        }

        Ok(Tensor {
            node: Rc::new(RefCell::new(Node {
                data: TensorData::Expression,
                shape: vec![shape[1], shape[0]],
                requires_grad: self.requires_grad(),
                op: Op::Transpose2D,
                parents: vec![self.clone()],
                value_cache: None,
                grad: None,
            })),
        })
    }

    pub fn repeat_cols(&self, cols: usize) -> Result<Tensor, TensorError> {
        let shape = self.shape();
        if shape.len() != 2 || shape[1] != 1 {
            return Err(TensorError::InvalidShape { shape, data_len: cols });
        }
        if cols == 0 {
            return Err(TensorError::InvalidShape {
                shape: vec![shape[0], cols],
                data_len: cols,
            });
        }

        Ok(Tensor {
            node: Rc::new(RefCell::new(Node {
                data: TensorData::Expression,
                shape: vec![shape[0], cols],
                requires_grad: self.requires_grad(),
                op: Op::RepeatCols(cols),
                parents: vec![self.clone()],
                value_cache: None,
                grad: None,
            })),
        })
    }

    pub fn repeat_rows(&self, rows: usize) -> Result<Tensor, TensorError> {
        let shape = self.shape();
        if shape.len() != 2 || shape[0] != 1 {
            return Err(TensorError::InvalidShape { shape, data_len: rows });
        }
        if rows == 0 {
            return Err(TensorError::InvalidShape {
                shape: vec![rows, shape[1]],
                data_len: rows,
            });
        }

        Ok(Tensor {
            node: Rc::new(RefCell::new(Node {
                data: TensorData::Expression,
                shape: vec![rows, shape[1]],
                requires_grad: self.requires_grad(),
                op: Op::RepeatRows(rows),
                parents: vec![self.clone()],
                value_cache: None,
                grad: None,
            })),
        })
    }

    pub fn sub(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        let lhs_shape = self.shape();
        let rhs_shape = other.shape();
        if lhs_shape != rhs_shape {
            return Err(TensorError::ShapeMismatch {
                lhs: lhs_shape,
                rhs: rhs_shape,
                op: "sub",
            });
        }

        Ok(Tensor {
            node: Rc::new(RefCell::new(Node {
                data: TensorData::Expression,
                shape: self.shape(),
                requires_grad: self.requires_grad() || other.requires_grad(),
                op: Op::Sub,
                parents: vec![self.clone(), other.clone()],
                value_cache: None,
                grad: None,
            })),
        })
    }

    pub fn abs(&self) -> Tensor {
        Tensor {
            node: Rc::new(RefCell::new(Node {
                data: TensorData::Expression,
                shape: self.shape(),
                requires_grad: self.requires_grad(),
                op: Op::Abs,
                parents: vec![self.clone()],
                value_cache: None,
                grad: None,
            })),
        }
    }

    pub fn huber(&self, delta: f32) -> Tensor {
        Tensor {
            node: Rc::new(RefCell::new(Node {
                data: TensorData::Expression,
                shape: self.shape(),
                requires_grad: self.requires_grad(),
                op: Op::Huber(delta),
                parents: vec![self.clone()],
                value_cache: None,
                grad: None,
            })),
        }
    }

    pub fn relu(&self) -> Tensor {
        Tensor {
            node: Rc::new(RefCell::new(Node {
                data: TensorData::Expression,
                shape: self.shape(),
                requires_grad: self.requires_grad(),
                op: Op::Relu,
                parents: vec![self.clone()],
                value_cache: None,
                grad: None,
            })),
        }
    }

    pub fn tanh(&self) -> Tensor {
        Tensor {
            node: Rc::new(RefCell::new(Node {
                data: TensorData::Expression,
                shape: self.shape(),
                requires_grad: self.requires_grad(),
                op: Op::Tanh,
                parents: vec![self.clone()],
                value_cache: None,
                grad: None,
            })),
        }
    }

    pub fn sum(&self) -> Result<Tensor, TensorError> {
        if numel(&self.shape()) == 0 {
            return Err(TensorError::EmptyReduction);
        }

        Ok(Tensor {
            node: Rc::new(RefCell::new(Node {
                data: TensorData::Expression,
                shape: vec![1],
                requires_grad: self.requires_grad(),
                op: Op::Sum,
                parents: vec![self.clone()],
                value_cache: None,
                grad: None,
            })),
        })
    }

    pub fn sum_axis(&self, dim: usize) -> Result<Tensor, TensorError> {
        let shape = self.shape();
        if shape.len() != 2 || dim > 1 {
            return Err(TensorError::InvalidShape { shape, data_len: dim });
        }
        if shape[dim] == 0 {
            return Err(TensorError::EmptyReduction);
        }

        let out_shape = if dim == 0 {
            vec![1, shape[1]]
        } else {
            vec![shape[0], 1]
        };

        Ok(Tensor {
            node: Rc::new(RefCell::new(Node {
                data: TensorData::Expression,
                shape: out_shape,
                requires_grad: self.requires_grad(),
                op: Op::SumAxis(dim),
                parents: vec![self.clone()],
                value_cache: None,
                grad: None,
            })),
        })
    }

    pub fn exp(&self) -> Tensor {
        Tensor {
            node: Rc::new(RefCell::new(Node {
                data: TensorData::Expression,
                shape: self.shape(),
                requires_grad: self.requires_grad(),
                op: Op::Exp,
                parents: vec![self.clone()],
                value_cache: None,
                grad: None,
            })),
        }
    }

    pub fn log(&self) -> Tensor {
        Tensor {
            node: Rc::new(RefCell::new(Node {
                data: TensorData::Expression,
                shape: self.shape(),
                requires_grad: self.requires_grad(),
                op: Op::Log,
                parents: vec![self.clone()],
                value_cache: None,
                grad: None,
            })),
        }
    }

    pub fn sqrt(&self) -> Tensor {
        Tensor {
            node: Rc::new(RefCell::new(Node {
                data: TensorData::Expression,
                shape: self.shape(),
                requires_grad: self.requires_grad(),
                op: Op::Sqrt,
                parents: vec![self.clone()],
                value_cache: None,
                grad: None,
            })),
        }
    }

    pub fn softmax(&self, dim: usize) -> Result<Tensor, TensorError> {
        let shape = self.shape();
        if shape.len() != 2 || dim > 1 {
            return Err(TensorError::InvalidShape { shape, data_len: dim });
        }

        Ok(Tensor {
            node: Rc::new(RefCell::new(Node {
                data: TensorData::Expression,
                shape: self.shape(),
                requires_grad: self.requires_grad(),
                op: Op::Softmax(dim),
                parents: vec![self.clone()],
                value_cache: None,
                grad: None,
            })),
        })
    }

    pub fn log_softmax(&self, dim: usize) -> Result<Tensor, TensorError> {
        let shape = self.shape();
        if shape.len() != 2 || dim > 1 {
            return Err(TensorError::InvalidShape { shape, data_len: dim });
        }

        Ok(Tensor {
            node: Rc::new(RefCell::new(Node {
                data: TensorData::Expression,
                shape: self.shape(),
                requires_grad: self.requires_grad(),
                op: Op::LogSoftmax(dim),
                parents: vec![self.clone()],
                value_cache: None,
                grad: None,
            })),
        })
    }

    pub fn backward(&self) {
        let mut topo = Vec::new();
        let mut seen = HashSet::new();
        self.build_topo(&mut topo, &mut seen);

        for node in &topo {
            node.zero_grad();
        }

        let out_len = self.eval().len();
        self.accumulate_grad(&vec![1.0; out_len]);

        for tensor in topo.into_iter().rev() {
            let (op, parents, grad_now) = {
                let n = tensor.node.borrow();
                (n.op, n.parents.clone(), n.grad.clone())
            };

            let Some(grad_now) = grad_now else {
                continue;
            };

            match op {
                Op::None => {}
                Op::Add => {
                    parents[0].accumulate_grad(&grad_now);
                    parents[1].accumulate_grad(&grad_now);
                }
                Op::Sub => {
                    parents[0].accumulate_grad(&grad_now);
                    let neg_grad: Vec<f32> = grad_now.iter().map(|g| -g).collect();
                    parents[1].accumulate_grad(&neg_grad);
                }
                Op::Mul => {
                    let left_val = parents[0].eval();
                    let right_val = parents[1].eval();
                    let mut left_grad = vec![0.0; left_val.len()];
                    let mut right_grad = vec![0.0; right_val.len()];

                    for i in 0..grad_now.len() {
                        left_grad[i] = grad_now[i] * right_val[i];
                        right_grad[i] = grad_now[i] * left_val[i];
                    }

                    parents[0].accumulate_grad(&left_grad);
                    parents[1].accumulate_grad(&right_grad);
                }
                Op::Div => {
                    let left_val = parents[0].eval();
                    let right_val = parents[1].eval();
                    let mut left_grad = vec![0.0; left_val.len()];
                    let mut right_grad = vec![0.0; right_val.len()];

                    for i in 0..grad_now.len() {
                        left_grad[i] = grad_now[i] / right_val[i];
                        right_grad[i] =
                            -grad_now[i] * left_val[i] / (right_val[i] * right_val[i]);
                    }

                    parents[0].accumulate_grad(&left_grad);
                    parents[1].accumulate_grad(&right_grad);
                }
                Op::MatMul => {
                    let a = &parents[0];
                    let b = &parents[1];
                    let a_shape = a.shape();
                    let b_shape = b.shape();
                    let m = a_shape[0];
                    let k = a_shape[1];
                    let n = b_shape[1];

                    let a_val = a.eval();
                    let b_val = b.eval();

                    // dL/dA = dL/dY * B^T
                    let b_t = transpose_2d(&b_val, k, n);
                    let da = matmul_tiled_parallel(&grad_now, &b_t, m, n, k);
                    a.accumulate_grad(&da);

                    // dL/dB = A^T * dL/dY
                    let a_t = transpose_2d(&a_val, m, k);
                    let db = matmul_tiled_parallel(&a_t, &grad_now, k, m, n);
                    b.accumulate_grad(&db);
                }
                Op::Reshape => {
                    parents[0].accumulate_grad(&grad_now);
                }
                Op::Transpose2D => {
                    let p = &parents[0];
                    let p_shape = p.shape();
                    let rows = p_shape[0];
                    let cols = p_shape[1];
                    let mut p_grad = vec![0.0; rows * cols];
                    for r in 0..rows {
                        for c in 0..cols {
                            p_grad[r * cols + c] = grad_now[c * rows + r];
                        }
                    }
                    p.accumulate_grad(&p_grad);
                }
                Op::RepeatCols(cols) => {
                    let p = &parents[0];
                    let in_shape = p.shape();
                    let rows = in_shape[0];
                    let mut p_grad = vec![0.0; rows];
                    for r in 0..rows {
                        let mut acc = 0.0;
                        for c in 0..cols {
                            acc += grad_now[r * cols + c];
                        }
                        p_grad[r] = acc;
                    }
                    p.accumulate_grad(&p_grad);
                }
                Op::RepeatRows(rows) => {
                    let p = &parents[0];
                    let in_shape = p.shape();
                    let cols = in_shape[1];
                    let mut p_grad = vec![0.0; cols];
                    for c in 0..cols {
                        let mut acc = 0.0;
                        for r in 0..rows {
                            acc += grad_now[r * cols + c];
                        }
                        p_grad[c] = acc;
                    }
                    p.accumulate_grad(&p_grad);
                }
                Op::Relu => {
                    let p = &parents[0];
                    let p_val = p.eval();
                    let input_grad: Vec<f32> = grad_now
                        .iter()
                        .zip(p_val.iter())
                        .map(|(g, x)| if *x > 0.0 { *g } else { 0.0 })
                        .collect();
                    p.accumulate_grad(&input_grad);
                }
                Op::Tanh => {
                    let p = &parents[0];
                    let p_val = p.eval();
                    let input_grad: Vec<f32> = grad_now
                        .iter()
                        .zip(p_val.iter())
                        .map(|(g, x)| {
                            let t = x.tanh();
                            g * (1.0 - t * t)
                        })
                        .collect();
                    p.accumulate_grad(&input_grad);
                }
                Op::Exp => {
                    let p = &parents[0];
                    let p_val = p.eval();
                    let input_grad: Vec<f32> = grad_now
                        .iter()
                        .zip(p_val.iter())
                        .map(|(g, x)| g * x.exp())
                        .collect();
                    p.accumulate_grad(&input_grad);
                }
                Op::Log => {
                    let p = &parents[0];
                    let p_val = p.eval();
                    let input_grad: Vec<f32> = grad_now
                        .iter()
                        .zip(p_val.iter())
                        .map(|(g, x)| g / x)
                        .collect();
                    p.accumulate_grad(&input_grad);
                }
                Op::Sqrt => {
                    let p = &parents[0];
                    let p_val = p.eval();
                    let input_grad: Vec<f32> = grad_now
                        .iter()
                        .zip(p_val.iter())
                        .map(|(g, x)| g * 0.5 / x.sqrt())
                        .collect();
                    p.accumulate_grad(&input_grad);
                }
                Op::Softmax(dim) => {
                    let p = &parents[0];
                    let y = tensor.eval();
                    let shape = p.shape();
                    let rows = shape[0];
                    let cols = shape[1];
                    let mut input_grad = vec![0.0; y.len()];

                    if dim == 1 {
                        for r in 0..rows {
                            let mut dot = 0.0;
                            for c in 0..cols {
                                dot += grad_now[r * cols + c] * y[r * cols + c];
                            }
                            for c in 0..cols {
                                let idx = r * cols + c;
                                input_grad[idx] = y[idx] * (grad_now[idx] - dot);
                            }
                        }
                    } else {
                        for c in 0..cols {
                            let mut dot = 0.0;
                            for r in 0..rows {
                                dot += grad_now[r * cols + c] * y[r * cols + c];
                            }
                            for r in 0..rows {
                                let idx = r * cols + c;
                                input_grad[idx] = y[idx] * (grad_now[idx] - dot);
                            }
                        }
                    }
                    p.accumulate_grad(&input_grad);
                }
                Op::LogSoftmax(dim) => {
                    let p = &parents[0];
                    let y = tensor.eval();
                    let probs: Vec<f32> = y.iter().map(|v| v.exp()).collect();
                    let shape = p.shape();
                    let rows = shape[0];
                    let cols = shape[1];
                    let mut input_grad = vec![0.0; y.len()];

                    if dim == 1 {
                        for r in 0..rows {
                            let mut row_sum = 0.0;
                            for c in 0..cols {
                                row_sum += grad_now[r * cols + c];
                            }
                            for c in 0..cols {
                                let idx = r * cols + c;
                                input_grad[idx] = grad_now[idx] - probs[idx] * row_sum;
                            }
                        }
                    } else {
                        for c in 0..cols {
                            let mut col_sum = 0.0;
                            for r in 0..rows {
                                col_sum += grad_now[r * cols + c];
                            }
                            for r in 0..rows {
                                let idx = r * cols + c;
                                input_grad[idx] = grad_now[idx] - probs[idx] * col_sum;
                            }
                        }
                    }
                    p.accumulate_grad(&input_grad);
                }
                Op::SumAxis(dim) => {
                    let p = &parents[0];
                    let p_shape = p.shape();
                    let rows = p_shape[0];
                    let cols = p_shape[1];
                    let mut p_grad = vec![0.0; rows * cols];

                    if dim == 0 {
                        for r in 0..rows {
                            for c in 0..cols {
                                p_grad[r * cols + c] = grad_now[c];
                            }
                        }
                    } else {
                        for r in 0..rows {
                            let v = grad_now[r];
                            for c in 0..cols {
                                p_grad[r * cols + c] = v;
                            }
                        }
                    }
                    p.accumulate_grad(&p_grad);
                }
                Op::Sum => {
                    let p = &parents[0];
                    let p_len = p.eval().len();
                    let expanded = vec![grad_now[0]; p_len];
                    p.accumulate_grad(&expanded);
                }
                Op::Abs => {
                    let p = &parents[0];
                    let p_val = p.eval();
                    let local_grad: Vec<f32> = p_val
                        .iter()
                        .map(|v| {
                            if *v > 0.0 {
                                1.0
                            } else if *v < 0.0 {
                                -1.0
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    let input_grad: Vec<f32> = grad_now
                        .iter()
                        .zip(local_grad.iter())
                        .map(|(g, l)| g * l)
                        .collect();
                    p.accumulate_grad(&input_grad);
                }
                Op::Huber(delta) => {
                    let p = &parents[0];
                    let p_val = p.eval();
                    let input_grad: Vec<f32> = grad_now
                        .iter()
                        .zip(p_val.iter())
                        .map(|(g, x)| {
                            let local = if x.abs() <= delta {
                                *x
                            } else {
                                delta * x.signum()
                            };
                            g * local
                        })
                        .collect();
                    p.accumulate_grad(&input_grad);
                }
            }
        }
    }

    fn build_topo(&self, topo: &mut Vec<Tensor>, seen: &mut HashSet<usize>) {
        let key = Rc::as_ptr(&self.node) as usize;
        if seen.contains(&key) {
            return;
        }
        seen.insert(key);

        let parents = self.node.borrow().parents.clone();
        for p in parents {
            p.build_topo(topo, seen);
        }
        topo.push(self.clone());
    }

    fn accumulate_grad(&self, incoming: &[f32]) {
        let mut node = self.node.borrow_mut();
        if !node.requires_grad {
            return;
        }

        match &mut node.grad {
            Some(existing) => {
                for i in 0..existing.len() {
                    existing[i] += incoming[i];
                }
            }
            None => {
                node.grad = Some(incoming.to_vec());
            }
        }
    }

    fn materialize(&self) -> Vec<f32> {
        if let Some(cache) = self.node.borrow().value_cache.clone() {
            return cache;
        }

        let computed = {
            let n = self.node.borrow();
            match &n.data {
                TensorData::Loaded(v) => v.clone(),
                TensorData::Procedural {
                    seed,
                    shape,
                    generator_func,
                } => {
                    let total = numel(shape);
                    (0..total).map(|idx| generator_func(*seed, idx)).collect()
                }
                TensorData::Expression => match n.op {
                    Op::Add => {
                        let a = n.parents[0].materialize();
                        let b = n.parents[1].materialize();
                        a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
                    }
                    Op::Sub => {
                        let a = n.parents[0].materialize();
                        let b = n.parents[1].materialize();
                        a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
                    }
                    Op::Mul => {
                        let a = n.parents[0].materialize();
                        let b = n.parents[1].materialize();
                        a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
                    }
                    Op::Div => {
                        let a = n.parents[0].materialize();
                        let b = n.parents[1].materialize();
                        a.iter().zip(b.iter()).map(|(x, y)| x / y).collect()
                    }
                    Op::MatMul => {
                        let a = n.parents[0].materialize();
                        let b = n.parents[1].materialize();
                        let a_shape = n.parents[0].shape();
                        let b_shape = n.parents[1].shape();
                        let m = a_shape[0];
                        let k = a_shape[1];
                        let out_n = b_shape[1];

                        matmul_tiled_parallel(&a, &b, m, k, out_n)
                    }
                    Op::SumAxis(dim) => {
                        let a = n.parents[0].materialize();
                        let in_shape = n.parents[0].shape();
                        let rows = in_shape[0];
                        let cols = in_shape[1];

                        if dim == 0 {
                            let mut out = vec![0.0; cols];
                            for r in 0..rows {
                                for c in 0..cols {
                                    out[c] += a[r * cols + c];
                                }
                            }
                            out
                        } else {
                            let mut out = vec![0.0; rows];
                            for r in 0..rows {
                                for c in 0..cols {
                                    out[r] += a[r * cols + c];
                                }
                            }
                            out
                        }
                    }
                    Op::Exp => {
                        let a = n.parents[0].materialize();
                        a.iter().map(|x| x.exp()).collect()
                    }
                    Op::Log => {
                        let a = n.parents[0].materialize();
                        a.iter().map(|x| x.ln()).collect()
                    }
                    Op::Sqrt => {
                        let a = n.parents[0].materialize();
                        a.iter().map(|x| x.sqrt()).collect()
                    }
                    Op::Softmax(dim) => {
                        let a = n.parents[0].materialize();
                        let shape = n.parents[0].shape();
                        let rows = shape[0];
                        let cols = shape[1];
                        let mut out = vec![0.0; rows * cols];

                        if dim == 1 {
                            for r in 0..rows {
                                let row = &a[r * cols..(r + 1) * cols];
                                let row_max = row
                                    .iter()
                                    .copied()
                                    .fold(f32::NEG_INFINITY, f32::max);
                                let mut denom = 0.0;
                                for c in 0..cols {
                                    let e = (row[c] - row_max).exp();
                                    out[r * cols + c] = e;
                                    denom += e;
                                }
                                for c in 0..cols {
                                    out[r * cols + c] /= denom;
                                }
                            }
                        } else {
                            for c in 0..cols {
                                let mut col_max = f32::NEG_INFINITY;
                                for r in 0..rows {
                                    col_max = col_max.max(a[r * cols + c]);
                                }
                                let mut denom = 0.0;
                                for r in 0..rows {
                                    let e = (a[r * cols + c] - col_max).exp();
                                    out[r * cols + c] = e;
                                    denom += e;
                                }
                                for r in 0..rows {
                                    out[r * cols + c] /= denom;
                                }
                            }
                        }
                        out
                    }
                    Op::LogSoftmax(dim) => {
                        let a = n.parents[0].materialize();
                        let shape = n.parents[0].shape();
                        let rows = shape[0];
                        let cols = shape[1];
                        let mut out = vec![0.0; rows * cols];

                        if dim == 1 {
                            for r in 0..rows {
                                let row = &a[r * cols..(r + 1) * cols];
                                let row_max = row
                                    .iter()
                                    .copied()
                                    .fold(f32::NEG_INFINITY, f32::max);
                                let mut denom = 0.0;
                                for c in 0..cols {
                                    denom += (row[c] - row_max).exp();
                                }
                                let lse = row_max + denom.ln();
                                for c in 0..cols {
                                    out[r * cols + c] = row[c] - lse;
                                }
                            }
                        } else {
                            for c in 0..cols {
                                let mut col_max = f32::NEG_INFINITY;
                                for r in 0..rows {
                                    col_max = col_max.max(a[r * cols + c]);
                                }
                                let mut denom = 0.0;
                                for r in 0..rows {
                                    denom += (a[r * cols + c] - col_max).exp();
                                }
                                let lse = col_max + denom.ln();
                                for r in 0..rows {
                                    out[r * cols + c] = a[r * cols + c] - lse;
                                }
                            }
                        }
                        out
                    }
                    Op::Reshape => n.parents[0].materialize(),
                    Op::Transpose2D => {
                        let a = n.parents[0].materialize();
                        let in_shape = n.parents[0].shape();
                        let rows = in_shape[0];
                        let cols = in_shape[1];
                        let mut out = vec![0.0; rows * cols];
                        for r in 0..cols {
                            for c in 0..rows {
                                out[r * rows + c] = a[c * cols + r];
                            }
                        }
                        out
                    }
                    Op::RepeatCols(cols) => {
                        let a = n.parents[0].materialize();
                        let rows = n.parents[0].shape()[0];
                        let mut out = vec![0.0; rows * cols];
                        for r in 0..rows {
                            for c in 0..cols {
                                out[r * cols + c] = a[r];
                            }
                        }
                        out
                    }
                    Op::RepeatRows(rows) => {
                        let a = n.parents[0].materialize();
                        let cols = n.parents[0].shape()[1];
                        let mut out = vec![0.0; rows * cols];
                        for r in 0..rows {
                            for c in 0..cols {
                                out[r * cols + c] = a[c];
                            }
                        }
                        out
                    }
                    Op::Relu => {
                        let a = n.parents[0].materialize();
                        a.iter().map(|x| x.max(0.0)).collect()
                    }
                    Op::Tanh => {
                        let a = n.parents[0].materialize();
                        a.iter().map(|x| x.tanh()).collect()
                    }
                    Op::Sum => {
                        let a = n.parents[0].materialize();
                        vec![a.iter().sum()]
                    }
                    Op::Abs => {
                        let a = n.parents[0].materialize();
                        a.iter().map(|v| v.abs()).collect()
                    }
                    Op::Huber(delta) => {
                        let a = n.parents[0].materialize();
                        a.iter()
                            .map(|x| {
                                let ax = x.abs();
                                if ax <= delta {
                                    0.5 * x * x
                                } else {
                                    delta * (ax - 0.5 * delta)
                                }
                            })
                            .collect()
                    }
                    Op::None => vec![],
                },
            }
        };

        self.node.borrow_mut().value_cache = Some(computed.clone());
        computed
    }
}

fn numel(shape: &[usize]) -> usize {
    if shape.is_empty() {
        return 0;
    }
    shape.iter().product()
}

fn transpose_2d(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = src[r * cols + c];
        }
    }
    out
}

fn matmul_tiled_parallel(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let tile = select_tile_size(k, n);
    let mut out = vec![0.0; m * n];

    out.par_chunks_mut(n).enumerate().for_each(|(i, row_out)| {
        let row_a = &a[i * k..(i + 1) * k];
        for kk in (0..k).step_by(tile) {
            let k_end = (kk + tile).min(k);
            for jj in (0..n).step_by(tile) {
                let j_end = (jj + tile).min(n);
                for x in kk..k_end {
                    let a_val = row_a[x];
                    let b_row = &b[x * n..x * n + n];
                    for j in jj..j_end {
                        row_out[j] += a_val * b_row[j];
                    }
                }
            }
        }
    });

    out
}

fn select_tile_size(k: usize, n: usize) -> usize {
    if let Ok(raw) = std::env::var("RSI_ML_MATMUL_TILE") {
        if let Ok(v) = raw.parse::<usize>() {
            if matches!(v, 8 | 16 | 32 | 64 | 128) {
                return v;
            }
        }
    }

    if std::env::var("RSI_ML_MATMUL_AUTOTUNE").ok().as_deref() == Some("1") {
        static BEST_TILE: OnceLock<usize> = OnceLock::new();
        return *BEST_TILE.get_or_init(autotune_tile_once);
    }

    let max_dim = k.max(n);
    if max_dim >= 1024 {
        64
    } else if max_dim >= 256 {
        32
    } else {
        16
    }
}

fn autotune_tile_once() -> usize {
    let candidates = [16usize, 32, 64];
    let m = 192usize;
    let k = 192usize;
    let n = 192usize;
    let a = vec![0.01f32; m * k];
    let b = vec![0.02f32; k * n];
    let mut best_tile = 32usize;
    let mut best_time = f64::INFINITY;

    for tile in candidates {
        let _warm = matmul_tiled_parallel_with_tile(&a, &b, m, k, n, tile);
        let t0 = Instant::now();
        let _ = matmul_tiled_parallel_with_tile(&a, &b, m, k, n, tile);
        let dt = t0.elapsed().as_secs_f64();
        if dt < best_time {
            best_time = dt;
            best_tile = tile;
        }
    }

    best_tile
}

fn matmul_tiled_parallel_with_tile(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    tile: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; m * n];

    out.par_chunks_mut(n).enumerate().for_each(|(i, row_out)| {
        let row_a = &a[i * k..(i + 1) * k];
        for kk in (0..k).step_by(tile) {
            let k_end = (kk + tile).min(k);
            for jj in (0..n).step_by(tile) {
                let j_end = (jj + tile).min(n);
                for x in kk..k_end {
                    let a_val = row_a[x];
                    let b_row = &b[x * n..x * n + n];
                    for j in jj..j_end {
                        row_out[j] += a_val * b_row[j];
                    }
                }
            }
        }
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static CALLS: AtomicUsize = AtomicUsize::new(0);

    fn seeded_linear(seed: u64, idx: usize) -> f32 {
        CALLS.fetch_add(1, Ordering::SeqCst);
        seed as f32 + idx as f32
    }

    #[test]
    fn procedural_tensor_is_lazy_and_cached() {
        CALLS.store(0, Ordering::SeqCst);
        let t = Tensor::procedural(10, vec![4], seeded_linear, false);

        assert_eq!(CALLS.load(Ordering::SeqCst), 0);
        let v1 = t.eval();
        assert_eq!(v1, vec![10.0, 11.0, 12.0, 13.0]);
        assert_eq!(CALLS.load(Ordering::SeqCst), 4);

        let v2 = t.eval();
        assert_eq!(v2, v1);
        assert_eq!(CALLS.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn lazy_graph_and_backward_work() {
        let a = Tensor::from_loaded(vec![2.0, 3.0], vec![2], true).unwrap();
        let b = Tensor::from_loaded(vec![4.0, 5.0], vec![2], true).unwrap();

        let y = a.mul(&b).unwrap().sum().unwrap();
        assert_eq!(y.eval(), vec![23.0]);

        y.backward();

        assert_eq!(a.grad().unwrap(), vec![4.0, 5.0]);
        assert_eq!(b.grad().unwrap(), vec![2.0, 3.0]);
    }

    #[test]
    fn add_backward_gives_ones() {
        let a = Tensor::from_loaded(vec![1.0, 2.0, 3.0], vec![3], true).unwrap();
        let b = Tensor::from_loaded(vec![10.0, 20.0, 30.0], vec![3], true).unwrap();
        let y = a.add(&b).unwrap().sum().unwrap();
        y.backward();

        assert_eq!(a.grad().unwrap(), vec![1.0, 1.0, 1.0]);
        assert_eq!(b.grad().unwrap(), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn matmul_forward_and_backward_work() {
        let a = Tensor::from_loaded(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true).unwrap();
        let b = Tensor::from_loaded(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], true).unwrap();

        let y = a.matmul(&b).unwrap();
        assert_eq!(y.eval(), vec![19.0, 22.0, 43.0, 50.0]);

        let loss = y.sum().unwrap();
        loss.backward();

        // d(sum(A*B))/dA = ones(2x2) * B^T
        assert_eq!(a.grad().unwrap(), vec![11.0, 15.0, 11.0, 15.0]);
        // d(sum(A*B))/dB = A^T * ones(2x2)
        assert_eq!(b.grad().unwrap(), vec![4.0, 4.0, 6.0, 6.0]);
    }

    #[test]
    fn matmul_shape_check_fails_for_incompatible_dims() {
        let a = Tensor::from_loaded(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true).unwrap();
        let b = Tensor::from_loaded(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4], true).unwrap();
        let res = a.matmul(&b);
        assert!(res.is_err());
    }

    #[test]
    fn reshape_forward_and_backward_work() {
        let a = Tensor::from_loaded(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true).unwrap();
        let b = a.reshape(vec![4]).unwrap();
        assert_eq!(b.eval(), vec![1.0, 2.0, 3.0, 4.0]);

        let loss = b.sum().unwrap();
        loss.backward();
        assert_eq!(a.grad().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn transpose2d_forward_and_backward_work() {
        let a = Tensor::from_loaded(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true).unwrap();
        let t = a.transpose2d().unwrap();
        assert_eq!(t.eval(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(t.shape(), vec![3, 2]);

        let loss = t.sum().unwrap();
        loss.backward();
        assert_eq!(a.grad().unwrap(), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn relu_forward_and_backward_work() {
        let x = Tensor::from_loaded(vec![-1.0, 0.0, 2.0], vec![3], true).unwrap();
        let y = x.relu();
        assert_eq!(y.eval(), vec![0.0, 0.0, 2.0]);

        let loss = y.sum().unwrap();
        loss.backward();
        assert_eq!(x.grad().unwrap(), vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn tanh_forward_and_backward_work() {
        let x = Tensor::from_loaded(vec![0.0, 1.0], vec![2], true).unwrap();
        let y = x.tanh();
        let yv = y.eval();
        assert!((yv[0] - 0.0).abs() < 1e-6);
        assert!((yv[1] - 1.0_f32.tanh()).abs() < 1e-6);

        let loss = y.sum().unwrap();
        loss.backward();
        let g = x.grad().unwrap();
        assert!((g[0] - 1.0).abs() < 1e-6);
        assert!((g[1] - (1.0 - 1.0_f32.tanh().powi(2))).abs() < 1e-5);
    }

    #[test]
    fn div_forward_and_backward_work() {
        let a = Tensor::from_loaded(vec![2.0, 4.0], vec![2], true).unwrap();
        let b = Tensor::from_loaded(vec![1.0, 2.0], vec![2], true).unwrap();
        let y = a.div(&b).unwrap();
        assert_eq!(y.eval(), vec![2.0, 2.0]);

        let loss = y.sum().unwrap();
        loss.backward();
        assert_eq!(a.grad().unwrap(), vec![1.0, 0.5]);
        assert_eq!(b.grad().unwrap(), vec![-2.0, -1.0]);
    }

    #[test]
    fn exp_log_sqrt_forward_backward_work() {
        let x = Tensor::from_loaded(vec![1.0, 4.0], vec![2], true).unwrap();
        let y = x.sqrt().log().exp();
        let out = y.eval();
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);

        let loss = y.sum().unwrap();
        loss.backward();
        let g = x.grad().unwrap();
        assert!((g[0] - 0.5).abs() < 1e-5);
        assert!((g[1] - 0.25).abs() < 1e-5);
    }

    #[test]
    fn sum_axis_forward_and_backward_work() {
        let x = Tensor::from_loaded(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true).unwrap();
        let col_sum = x.sum_axis(0).unwrap();
        assert_eq!(col_sum.shape(), vec![1, 2]);
        assert_eq!(col_sum.eval(), vec![4.0, 6.0]);

        let row_sum = x.sum_axis(1).unwrap();
        assert_eq!(row_sum.shape(), vec![2, 1]);
        assert_eq!(row_sum.eval(), vec![3.0, 7.0]);

        let loss = row_sum.sum().unwrap();
        loss.backward();
        assert_eq!(x.grad().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let x = Tensor::from_loaded(vec![1.0, 2.0, 3.0, 1.0], vec![2, 2], true).unwrap();
        let s = x.softmax(1).unwrap();
        let row_sum = s.sum_axis(1).unwrap();
        let v = row_sum.eval();
        assert!((v[0] - 1.0).abs() < 1e-6);
        assert!((v[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn log_softmax_matches_log_of_softmax() {
        let x = Tensor::from_loaded(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true).unwrap();
        let s = x.softmax(1).unwrap().log();
        let ls = x.log_softmax(1).unwrap();
        let a = s.eval();
        let b = ls.eval();
        for i in 0..a.len() {
            assert!((a[i] - b[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn repeat_cols_and_rows_forward_backward_work() {
        let col = Tensor::from_loaded(vec![1.0, 2.0], vec![2, 1], true).unwrap();
        let rc = col.repeat_cols(3).unwrap();
        assert_eq!(rc.eval(), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        rc.sum().unwrap().backward();
        assert_eq!(col.grad().unwrap(), vec![3.0, 3.0]);

        let row = Tensor::from_loaded(vec![1.0, 2.0, 3.0], vec![1, 3], true).unwrap();
        let rr = row.repeat_rows(2).unwrap();
        assert_eq!(rr.eval(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        rr.sum().unwrap().backward();
        assert_eq!(row.grad().unwrap(), vec![2.0, 2.0, 2.0]);
    }
}
