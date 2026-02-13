use rsi_ml_core::Tensor;

pub trait AutogradExt {
    fn backward_pass(&self);
    fn clear_grad(&self);
}

impl AutogradExt for Tensor {
    fn backward_pass(&self) {
        self.backward();
    }

    fn clear_grad(&self) {
        self.zero_grad();
    }
}
