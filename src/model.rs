use ndarray::{Array1, Array2};
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
    biases: Array1<f64>
}

impl LinearLayer {
    pub fn new(inputs: usize, outputs: usize) -> LinearLayer {
        let mut rng = SmallRng::from_entropy();
        let seeded_weights = vec![rng.gen_range(-1.0..1.0); inputs * outputs];
        let weights = Array2::from_shape_vec((inputs, outputs), seeded_weights)
            .expect("Error convertiong seeded Layer values into Array2"); 
        let biases = Array1::from_vec(vec![rng.gen_range(-1.0..1.0); outputs]);
        LinearLayer {
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
        for (key, layer) in self.steps.iter().enumerate() {
            if let Step::LinearLayer(l) = layer {
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
            steps: self.steps
        })
    }
}

#[derive(Debug, Default)]
pub struct Model {
    outputs: usize,
    pub steps: Vec<Step>
}

impl Model {
    pub fn forward(&self, inputs_array: Array1<f64>) -> Result<Array1<f64>> {
        let mut output_array = inputs_array;
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
                _ => {}
            }
        }
        Ok(output_array)
    }

    pub fn forward_backward(&self, inputs_array: Array1<f64>, answer: &f64) -> Result<()> {
        let mut answer_array : Array1<f64> = Array1::zeros(self.outputs);
        let mut output_array = inputs_array;
        answer_array[*answer as usize] = 1.0; 
        let mut outputs : Vec<Array1<f64>> = vec![];
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
                _ => {}
            }
            outputs.push(output_array.clone());
        }

        let probability_array = outputs.last().unwrap();
        assert_eq!(probability_array.len(), answer_array.len());
        let offset = &outputs.len();
        let output_after_activation = outputs.get(offset-1).unwrap();
        let output_before_activation = outputs.get(offset-2).unwrap();
        let input_into_layer = outputs.get(offset-3).unwrap();
        let cost = squared_error(&answer_array, probability_array);
        let error = d_squared_error(&answer_array, probability_array); // don't we need something
        let inputs = input_into_layer.len();
        println!("{:?}", &input_into_layer.t());
        let delta_weight = &error * &input_into_layer.t();
        // distance from goal 
        // squared_error * d-softmax(last-layer-output) 
        // need to get:
        // output from layer after activation
        // output from layer before activation
        // input into layer
        //let error = squared_error(answer_array, probability_array); 
        //let error = (probability_array - answer_array)* d_softmax_stable();

        Ok(())
    }
}

