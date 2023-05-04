use ndarray::Array1;

pub fn sigmoid(x: &f64) -> f64 {
    1.0 / (f64::exp(-x) + 1.0)
}

pub fn d_sigmoid(x: &f64) -> f64 {
    (f64::exp(-x)) / f64::powi(f64::exp(-x)+1.0, 2)
}

pub fn softmax(x: Array1<f64>) -> Array1<f64> {
    let sum : f64 = x.iter().map(|i| i.exp()).sum();
    x.iter().map(|i| i.exp() / sum).collect()
}

pub fn softmax_stable(x: Array1<f64>) -> Array1<f64> {
    let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let shifted_x : Array1<f64> = x.iter().map(|i| (i - max).exp()).collect();
    let sum : f64 = shifted_x.sum();
    shifted_x.iter().map(|i| i / sum).collect()
}

pub fn d_softmax_stable(x: Array1<f64>) -> Array1<f64> {
    let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let shifted_x : Array1<f64> = x.iter().map(|i| (i - max).exp()).collect();
    let sum : f64 = shifted_x.sum();
    shifted_x.iter().map(|i| (i/sum) * (1.0 - (i/sum))).collect()
}

pub fn squared_error(target: &Array1<f64>, output: &Array1<f64>) -> f64 {
    // ||x-y||^2 -> (x-y)^T * (x-y)
    //           -> (x-y) dot (x-y)
    let distance = target - output;
    distance.dot(&distance)
}

pub fn d_squared_error(target: &Array1<f64>, output: &Array1<f64>) -> Array1<f64> {
    2.0 * (output - target)
}
