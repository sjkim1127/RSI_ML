use rsi_ml_core::{Tensor, TensorError};

pub fn add(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, TensorError> {
    lhs.add(rhs)
}

pub fn mul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, TensorError> {
    lhs.mul(rhs)
}

pub fn sum(tensor: &Tensor) -> Result<Tensor, TensorError> {
    tensor.sum()
}
