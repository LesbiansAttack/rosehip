mod linearlayer;
mod activations;

pub use linearlayer::LinearLayer;
pub use activations::{
    SigmoidActivation,
    SoftmaxActivation
};
use ndarray::Array2;


#[derive(Debug)]
pub enum StepType {
    Activation,
    Layer,
}

pub trait Step {
    fn get_number_of_inputs_and_outputs(&self) -> Option<(usize,usize)> {None}
    fn step_type(&self) -> StepType;
    fn forward(&mut self, input_array: Array2<f64>) -> Array2<f64>;
    fn backward(&mut self, error: Array2<f64>, previous_gradient: Array2<f64>) -> (Array2<f64>, Array2<f64>);
    #[allow(unused_variables)]
    fn apply_gradients(&mut self){}
}