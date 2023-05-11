use ndarray::Array2;
use ndarray_stats::QuantileExt;


pub fn sigmoid(x: &f64) -> f64 {
    1.0 / (f64::exp(-x) + 1.0)
}

pub fn d_sigmoid(x: &f64) -> f64 {
    (f64::exp(-x)) / f64::powi(f64::exp(-x)+1.0, 2)
}

pub fn softmax(x: Array2<f64>) -> Array2<f64> {
    let sum : f64 = x.iter().map(|i| i.exp()).sum();
    let output_vec = x.iter().map(|i| i.exp() / sum).collect();
    Array2::from_shape_vec((x.len(), 1), output_vec).unwrap()
}

pub fn softmax_stable(x: Array2<f64>) -> Array2<f64> {
    let max = x.max().unwrap();
    let shifted_x = x.map(|i| (i - max).exp());
    let sum : f64 = shifted_x.sum();
    let output_vec = shifted_x.map(|i| i / sum);
    output_vec
}

pub fn d_softmax_stable(x: Array2<f64>) -> Array2<f64> {
    let max = x.max().unwrap();
    let shifted_x = x.map(|i| (i - max).exp());
    let sum : f64 = shifted_x.sum();
    let output_vec = shifted_x.map(|i| (i/sum) * (1.0 - (i/sum)));
    output_vec
}

pub fn squared_error(target: &Array2<f64>, output: &Array2<f64>) -> f64 {
    // ||x-y||^2 -> (x-y)^T * (x-y)
    //           -> (x-y) dot (x-y)
    let distance = target - output;
    // bit hacky but idk how else to do this
    distance.t().dot(&distance)[[0,0]]
}

pub fn d_squared_error(target: &Array2<f64>, output: &Array2<f64>) -> Array2<f64> {
    2.0 * (output - target)
}
