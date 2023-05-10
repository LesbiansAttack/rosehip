
use ndarray::{Array1, Array2};
use ndarray_rand::{RandomExt, rand_distr::{StandardNormal, Uniform}};

use eyre::{eyre, Result};
use rand::{
    Rng,
    SeedableRng,
    rngs::SmallRng
};
use crate::math::{
    sigmoid,
    d_sigmoid,
    softmax_stable,
    d_softmax_stable,
    squared_error,
    d_squared_error
};

#[derive(Debug)]
pub enum Step {
    Sigmoid, 
    Softmax,
    Identity,
    LinearLayer(LinearLayer)
}

impl Step {
    fn is_layer(&self) -> bool {
        matches!(self, Step::LinearLayer(_))
    }
}

#[derive(Debug)]
pub struct LinearLayer {
    number_of_inputs: usize,
    number_of_outputs: usize,
    weights: Array2<f64>,
    biases: Array2<f64>
}

impl LinearLayer {
    pub fn new(inputs: usize, outputs: usize) -> LinearLayer {
        let weights = Array2::random((outputs, inputs),StandardNormal) / f64::sqrt((outputs) as f64);
        let biases =  Array2::random((outputs, 1), Uniform::new(-0.01, 0.01));
        LinearLayer {
            number_of_inputs: inputs,
            number_of_outputs: outputs, 
            weights,
            biases
        }
    }

    pub fn adjust_weights(&mut self, weights_delta: Array2<f64>, biases_delta: Array2<f64>) {
        //println!("weights before: {}",self.weights);
        self.weights = &self.weights - weights_delta;
        self.biases = &self.biases - biases_delta;
       // println!("weights after: {}", self.weights);
    }

    pub fn calculate_outputs(&self, inputs_array: Array2<f64>) -> Array2<f64> {
        &self.weights.dot(&inputs_array) + &self.biases
    }
}

#[derive(Default)]
pub struct ModelBuilder {
    steps: Vec<Step>
}

impl ModelBuilder {
    pub fn add_linear_layer(mut self, inputs: usize, outputs: usize) -> ModelBuilder {
        match self.steps.last() {
            Some(step) if step.is_layer() => {
                self.steps.push(Step::Identity);
            },
            _ => {}
        }
        self.steps.push(Step::LinearLayer(LinearLayer::new(inputs, outputs)));
        self
    }

    pub fn add_sigmoid(mut self) -> ModelBuilder {
        self.steps.push(Step::Sigmoid);
        self
    }

    pub fn add_softmax(mut self) -> ModelBuilder {
        self.steps.push(Step::Softmax);
        self
    }

    pub fn build(mut self) -> Result<Model> {
        // validate the model
        if self.steps.last().unwrap().is_layer() {
            self.steps.push(Step::Identity);
        }
        let mut current_outputs : usize = 0;
        let mut num_lin_layers : usize = 0;

        for (key, layer) in self.steps.iter().enumerate() {
            if let Step::LinearLayer(l) = layer {
                num_lin_layers += 1;
                if current_outputs == 0 {
                    current_outputs = l.number_of_outputs;
                    continue;
                }
                if current_outputs == l.number_of_inputs {
                    current_outputs = l.number_of_outputs;
                } else {
                    return Err(eyre!("Mismatch on layer {}, expected {} inputs but got {} inputs",
                              key, current_outputs, l.number_of_inputs)
                    );
                } 
            }
        }
        
        Ok(Model {
            outputs: current_outputs,
            steps: self.steps,
            num_lin_layers: num_lin_layers,
        })
    }
}

#[derive(Debug, Default)]
pub struct Model {
    outputs: usize,
    pub steps: Vec<Step>,
    pub num_lin_layers: usize, 
}

impl Model {
    pub fn forward(&self, inputs_array: Array1<f64>) -> Result<Array2<f64>> {
        let inputs = inputs_array.len();

        let mut output_array = inputs_array.into_shape((inputs, 1)).unwrap();
        for step in &self.steps {
            match step {
                Step::Sigmoid => {
                    output_array = output_array.map(sigmoid);
                },
                Step::Softmax => {
                    output_array = softmax_stable(output_array);
                },
                Step::LinearLayer(layer) => {
                    output_array = layer.calculate_outputs(output_array);
                },
                Step::Identity => {

                }
            }
        }
        Ok(output_array)
    }

    pub fn forward_backward(&self, inputs_array: Array1<f64>, answer: &f64) -> Result<(Array2<f64>, Vec<Array2<f64>>, Vec<Array2<f64>>)> {
        let mut answer_array : Array2<f64> = Array2::zeros((self.outputs, 1));
        let inputs = inputs_array.len();
        let mut output_array = inputs_array.into_shape((inputs, 1)).unwrap();
        answer_array[[*answer as usize, 0]] = 1.0; 
        let mut outputs : Vec<Array2<f64>> = vec![];
        outputs.push(output_array.clone());
        for step in &self.steps {
            match step {
                Step::Sigmoid => {
                    output_array = output_array.map(sigmoid);
                },
                Step::Softmax => {
                    output_array = softmax_stable(output_array);
                },
                Step::LinearLayer(layer) => {
                    output_array = layer.calculate_outputs(output_array);
                    //println!("output_array: {}", output_array);
                },
                Step::Identity => {},
                _ => {}
            }
            outputs.push(output_array.clone());
        }

        let mut index = self.steps.len();
        let mut weights_deltas : Vec<Array2<f64>> = vec![];
        let mut bias_deltas : Vec<Array2<f64>> = vec![];

        let final_output = outputs.last().unwrap().clone();

        // * means element-wise multiplication
        let output_before_activation = outputs.get(index -1).unwrap();

        let mut error =
            d_squared_error(&answer_array, 
                outputs.last().unwrap()) * 
            Model::activation_derivative(
                &self.steps.last().unwrap(), 
                output_before_activation)
            .unwrap();

        let input_into_layer = outputs.get(index - 2).unwrap();
        let delta_weights = error.dot(&input_into_layer.t());

        weights_deltas.push(delta_weights);
        bias_deltas.push(error.clone());
        index -= 2;

        
        while index > 0 {
            let weights = match &self.steps[index] {
                Step::LinearLayer(layer) => layer.weights.clone(),
                _ => return Err(eyre!("Attempted to retreive Weights from Activation Function, expected Linear Layer"))
            };
            let activation_function = self.steps.get(index-1).unwrap();
            let output_before_activation = outputs.get(index - 1).unwrap();
            let input_into_layer = outputs.get(index - 2).unwrap();

            error = weights.t().dot(&error) * 
                Model::activation_derivative(
                    activation_function, 
                    output_before_activation)
                .unwrap();

            let delta_weights = error.dot(&input_into_layer.t());

            weights_deltas.push(delta_weights);
            bias_deltas.push(error.clone());
            index -= 2;
        }

        weights_deltas.reverse();
        bias_deltas.reverse();

        Ok((final_output,weights_deltas, bias_deltas))
    }

    pub fn activation_derivative(activation_function: &Step, input_vector: &Array2<f64>) -> Result<Array2<f64>> {
        match activation_function {
            Step::Sigmoid => {
                let output_vec = input_vector.iter().map(|i| d_sigmoid(&i)).collect();
                Ok(Array2::from_shape_vec((input_vector.len(), 1), output_vec).unwrap())
            },
            Step::Softmax => {
                Ok(d_softmax_stable(input_vector.clone()))
            },
            Step::Identity => {
                Ok(Array2::from_elem((input_vector.len(), 1), 1.0))
            },
            Step::LinearLayer(_) => {
                return Err(eyre!("Linear Layer Was Passed To Activation Derivative"));
            }
        }
    }


}

