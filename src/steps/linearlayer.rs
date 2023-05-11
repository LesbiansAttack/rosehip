use crate::steps::step::*;
use ndarray::Array2;
use ndarray_rand::{RandomExt, rand_distr::{StandardNormal, Uniform}};

#[derive(Debug)]
pub struct LinearLayer {
    number_of_inputs: usize,
    number_of_outputs: usize,
    weights: Array2<f64>,
    biases: Array2<f64>,
    recent_input: Array2<f64>,
    learning_rate: f64,
    delta_weights: Array2<f64>,
    delta_biases: Array2<f64>,
}

impl Step for LinearLayer {
    fn get_number_of_inputs_and_outputs(&self) -> Option<(usize,usize)> {
        Some((self.number_of_inputs, self.number_of_outputs))
    }
    fn step_type(&self) -> StepType{
        StepType::Layer
    }
    fn forward(&mut self, input_array: Array2<f64>) -> Array2<f64> {
        self.recent_input =input_array.clone();
        &self.weights.dot(&input_array) + &self.biases
    }
    fn backward(&mut self, error: Array2<f64>, previous_gradient: Array2<f64>)  -> (Array2<f64>, Array2<f64>){
        let new_error = error * previous_gradient;
        let delta_weights = new_error.dot(&self.recent_input.t());
        let output_error = self.weights.t().dot(&new_error);
        self.delta_weights = &self.delta_weights + delta_weights;
        self.delta_biases = &self.delta_biases + new_error;
        (output_error, Array2::from_elem((self.number_of_inputs, 1), 1.0))
    }
    fn finalize_batch(&mut self, batch_size: f64) {
        self.adjust_weights(&self.delta_weights / batch_size, &self.delta_biases / batch_size);
        self.delta_weights = Array2::zeros((self.number_of_outputs, self.number_of_inputs));
        self.delta_biases = Array2::zeros((self.number_of_outputs, 1));
    }
}

impl LinearLayer {
    pub fn new(inputs: usize, outputs: usize, learning_rate: f64) -> LinearLayer {
        let weights = Array2::random((outputs, inputs),StandardNormal) / f64::sqrt((outputs) as f64);
        let biases =  Array2::random((outputs, 1), Uniform::new(-0.01, 0.01));
        LinearLayer {
            number_of_inputs: inputs,
            number_of_outputs: outputs, 
            weights,
            biases,
            recent_input: Array2::zeros((inputs, 1)),
            learning_rate,
            delta_weights: Array2::zeros((outputs, inputs)),
            delta_biases: Array2::zeros((outputs, 1)),
        }
    }

    pub fn adjust_weights(&mut self, weights_delta: Array2<f64>, biases_delta: Array2<f64>) {
        self.weights = &self.weights - weights_delta * self.learning_rate;
        self.biases = &self.biases - biases_delta * self.learning_rate;
    }
}
