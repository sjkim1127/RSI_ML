use rsi_ml::{linear, mse, Optimizer, SGD, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let w1 = Tensor::from_loaded(vec![0.1, -0.2], vec![1, 2], true)?;
    let b1 = Tensor::from_loaded(vec![0.0, 0.0], vec![1, 2], true)?;
    let w2 = Tensor::from_loaded(vec![0.3, -0.4], vec![2, 1], true)?;
    let b2 = Tensor::from_loaded(vec![0.0], vec![1, 1], true)?;
    let mut optim = SGD::new(vec![w1.clone(), b1.clone(), w2.clone(), b2.clone()], 0.05);

    for step in 0..300 {
        let x = Tensor::from_loaded(vec![2.0], vec![1, 1], false)?;
        let y_true = Tensor::from_loaded(vec![5.0], vec![1, 1], false)?;

        let h = linear(&x, &w1, Some(&b1))?;
        let y_pred = linear(&h, &w2, Some(&b2))?;
        let loss = mse(&y_pred, &y_true)?;

        loss.backward();
        optim.step();
        optim.zero_grad();

        if step % 50 == 0 {
            println!("step={step} loss={:.6}", loss.eval()[0]);
        }
    }

    let x = Tensor::from_loaded(vec![2.0], vec![1, 1], false)?;
    let y_pred = linear(&linear(&x, &w1, Some(&b1))?, &w2, Some(&b2))?;
    println!("final prediction: {:?}", y_pred.eval());
    Ok(())
}
