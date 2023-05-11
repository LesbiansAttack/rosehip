use ndarray::Array2;
use crate::step::{Step, StepType};
use crate::math::{
    sigmoid,
    d_sigmoid,
    softmax_stable,
    d_softmax_stable,
};

#[derive(Default)]
pub struct SigmoidActivation {
    recent_input: Array2<f64>,
}

impl Step for SigmoidActivation {
    fn step_type(&self) -> StepType{
        StepType::Activation
    }
    fn forward(&mut self, input_array: Array2<f64>) -> Array2<f64>{
        self.recent_input = input_array.clone();
        input_array.map(sigmoid)
    }
    fn backward(&mut self, error: Array2<f64>, previous_gradient: Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        (error, self.recent_input.map(d_sigmoid) * previous_gradient)
    }
}

#[derive(Default)]
pub struct SoftmaxActivation {    
    recent_input: Array2<f64>,
}

impl Step for SoftmaxActivation {
    fn step_type(&self) -> StepType{
        StepType::Activation
    }
    fn forward(&mut self, input_array: Array2<f64>) -> Array2<f64>{
        self.recent_input = input_array.clone();
        softmax_stable(input_array)
    }
    fn backward(&mut self, error: Array2<f64>, previous_gradient: Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        (error, d_softmax_stable(self.recent_input.clone()) * previous_gradient)
    }
}

#[derive(Default)]
pub struct PassthroughActivation {    
    recent_input: Array2<f64>,
}

impl Step for PassthroughActivation {
    fn step_type(&self) -> StepType {
        StepType::Activation
    }
    fn forward(&mut self, input_array: Array2<f64>) -> Array2<f64> {
        self.recent_input = input_array.clone();
        input_array
    }
    fn backward(&mut self, error: Array2<f64>, previous_gradient: Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        (error, previous_gradient)
    }
}