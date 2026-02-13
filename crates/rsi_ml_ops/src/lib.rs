use rsi_ml_core::{Tensor, TensorError};

pub fn add(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, TensorError> {
    lhs.add(rhs)
}

pub fn sub(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, TensorError> {
    lhs.sub(rhs)
}

pub fn mul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, TensorError> {
    lhs.mul(rhs)
}

pub fn sum(tensor: &Tensor) -> Result<Tensor, TensorError> {
    tensor.sum()
}

pub fn mse(pred: &Tensor, target: &Tensor) -> Result<Tensor, TensorError> {
    let diff = pred.sub(target)?;
    let sq = diff.mul(&diff)?;
    mean_of(&sq)
}

pub fn l1(pred: &Tensor, target: &Tensor) -> Result<Tensor, TensorError> {
    let diff = pred.sub(target)?;
    let abs = diff.abs();
    mean_of(&abs)
}

pub fn huber(pred: &Tensor, target: &Tensor, delta: f32) -> Result<Tensor, TensorError> {
    let diff = pred.sub(target)?;
    let huber_vals = diff.huber(delta);
    mean_of(&huber_vals)
}

fn mean_of(tensor: &Tensor) -> Result<Tensor, TensorError> {
    let sum_v = tensor.sum()?;
    let count = tensor.shape().iter().product::<usize>();
    if count == 0 {
        return Err(TensorError::EmptyReduction);
    }
    let inv_count = Tensor::from_loaded(vec![1.0 / count as f32], vec![1], false)?;
    sum_v.mul(&inv_count)
}
