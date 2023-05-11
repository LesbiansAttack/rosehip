use std::time::Instant;

use ndarray::Array2;
use crate::step::{
    LinearLayer, 
    Step, 
    StepType, 
    SoftmaxActivation,
    SigmoidActivation
};
use eyre::{eyre, Result};

use crate::math::d_squared_error;



#[derive(Default)]
pub struct ModelBuilder {
    steps: Vec<Box<dyn Step>>
}

impl ModelBuilder {
    pub fn add_linear_layer(mut self, inputs: usize, outputs: usize, learning_rate: f64) -> ModelBuilder {
        self.steps.push(Box::new(LinearLayer::new(inputs, outputs, learning_rate)));
        self
    }

    pub fn add_sigmoid(mut self) -> ModelBuilder {
        self.steps.push(Box::new(SigmoidActivation::default()));
        self
    }

    pub fn add_softmax(mut self) -> ModelBuilder {
        self.steps.push(Box::new(SoftmaxActivation::default()));
        self
    }

    pub fn build(self) -> Result<Model> {
        
        let mut current_outputs : usize = 0;
        let mut num_lin_layers : usize = 0;

        for (key, layer) in self.steps.iter().enumerate() {
            if matches!(layer.step_type(), StepType::Layer){
                num_lin_layers += 1;
            }

            match layer.get_number_of_inputs_and_outputs(){
                Some((inputs, outputs)) => {
                    if current_outputs == inputs || current_outputs == 0 {
                        current_outputs = outputs
                    } else {
                        return Err(eyre!("Mismatch on layer {}, expected {} inputs but got {} inputs",
                            key, current_outputs, inputs));
                    }
                },
                None => continue
            } 
        }
        
        Ok(Model {
            outputs: current_outputs,
            steps: self.steps,
            num_lin_layers: num_lin_layers,
        })
    }
}

#[derive(Default)]
pub struct Model {
    outputs: usize,
    pub steps: Vec<Box<dyn Step>>,
    pub num_lin_layers: usize, 
}

impl Model {
    pub fn forward(&mut self, inputs_array: Array2<f64>) -> Result<Array2<f64>> {
        let mut output_array = inputs_array.clone();
        for step in self.steps.iter_mut() {
            output_array = step.forward(output_array);
        }
        Ok(output_array)
    }

    pub fn forward_backward(&mut self, inputs_array: Array2<f64>, answer: &f64) -> Result<Array2<f64>> {
        let mut answer_array : Array2<f64> = Array2::zeros((self.outputs, 1));
        let mut output_array = inputs_array.clone();
        answer_array[[*answer as usize, 0]] = 1.0; 
        for step in self.steps.iter_mut() {
            output_array = step.forward(output_array);
        }

        let final_output = output_array.clone();

        let mut error = d_squared_error(&answer_array, &final_output);
        let mut previous_gradient = Array2::from_elem((self.outputs, 1), 1.0);
        for step in self.steps.iter_mut().rev() {
            (error, previous_gradient) = step.backward(error, previous_gradient);
        }

        Ok(final_output)
    }

    pub fn finalize_batch(&mut self, batch_size: usize) {
        let b = batch_size as f64;
        for step in self.steps.iter_mut() {
            step.finalize_batch(b);
        }
    }

}

