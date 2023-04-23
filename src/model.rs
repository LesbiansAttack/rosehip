use ndarray::{array, Array1, Array2};

#[derive(Debug)]
pub struct Layer {
    number_of_inputs: usize,
    number_of_outputs: usize,
    weights: Array2<f64>,
    biases: Array1<f64>
}

impl Layer {
    pub fn new(inputs: usize, outputs: usize) -> Layer {
        let weights = Array2::ones((inputs, outputs));
        let biases = Array1::zeros(outputs);
        Layer {
            number_of_inputs: inputs,
            number_of_outputs: outputs,
            weights,
            biases
        }
    }

    pub fn calculate_outputs(&self, inputs_array: Array1<f64>) -> Array1<f64> {
        inputs_array.dot(&self.weights) + &self.biases
    }
}

pub struct Model {
    pub layers: Array1<Layer>
}
