use std::cell::RefCell;
use std::collections::HashSet;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::rc::Rc;

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
    Mul,
    Sum,
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
                Op::Sum => {
                    let p = &parents[0];
                    let p_len = p.eval().len();
                    let expanded = vec![grad_now[0]; p_len];
                    p.accumulate_grad(&expanded);
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
                    Op::Mul => {
                        let a = n.parents[0].materialize();
                        let b = n.parents[1].materialize();
                        a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
                    }
                    Op::Sum => {
                        let a = n.parents[0].materialize();
                        vec![a.iter().sum()]
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
}
