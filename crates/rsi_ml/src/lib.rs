pub use rsi_ml_autograd::AutogradExt;
pub use rsi_ml_core::{GeneratorFn, Tensor, TensorData, TensorError};
pub use rsi_ml_optim::{Adam, Optimizer, SGD, Sgd};
pub use rsi_ml_ops::{add, huber, l1, mse, mul, sub, sum};

#[cfg(test)]
mod tests {
    use super::{huber, l1, mse, mul, Adam, Optimizer, SGD, Tensor};

    #[test]
    fn sgd_learns_y_eq_2x() {
        let weight = Tensor::from_loaded(vec![0.0], vec![1], true).unwrap();
        let mut optim = SGD::new(vec![weight.clone()], 0.1);

        for _step in 0..80 {
            let x = Tensor::from_loaded(vec![3.0], vec![1], false).unwrap();
            let y_true = Tensor::from_loaded(vec![6.0], vec![1], false).unwrap();

            let y_pred = mul(&weight, &x).unwrap();
            let loss = mse(&y_pred, &y_true).unwrap();

            loss.backward();
            optim.step();
            optim.zero_grad();
        }

        let learned = weight.eval()[0];
        assert!((learned - 2.0).abs() < 0.02, "learned weight = {learned}");
    }

    #[test]
    fn adam_learns_y_eq_2x() {
        let weight = Tensor::from_loaded(vec![0.0], vec![1], true).unwrap();
        let mut optim = Adam::new(vec![weight.clone()], 0.05, 0.9, 0.999, 1e-8);

        for _step in 0..120 {
            let x = Tensor::from_loaded(vec![3.0], vec![1], false).unwrap();
            let y_true = Tensor::from_loaded(vec![6.0], vec![1], false).unwrap();

            let y_pred = mul(&weight, &x).unwrap();
            let loss = mse(&y_pred, &y_true).unwrap();

            loss.backward();
            optim.step();
            optim.zero_grad();
        }

        let learned = weight.eval()[0];
        assert!((learned - 2.0).abs() < 0.03, "learned weight = {learned}");
    }

    #[test]
    fn l1_and_huber_loss_are_zero_on_match() {
        let pred = Tensor::from_loaded(vec![1.0, 2.0, 3.0], vec![3], false).unwrap();
        let target = Tensor::from_loaded(vec![1.0, 2.0, 3.0], vec![3], false).unwrap();

        let l1_loss = l1(&pred, &target).unwrap().eval()[0];
        let huber_loss = huber(&pred, &target, 1.0).unwrap().eval()[0];

        assert!(l1_loss.abs() < 1e-6);
        assert!(huber_loss.abs() < 1e-6);
    }
}
