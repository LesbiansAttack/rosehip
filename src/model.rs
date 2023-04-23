use ndarray::{Array1, Array2};
use eyre::{eyre, Result};
use rand::{
    Rng,
    SeedableRng,
    rngs::SmallRng
};
use crate::math::{sigmoid, d_sigmoid, softmax_stable, d_softmax_stable};

#[derive(Debug)]
pub enum Activation {
    Sigmoid, 
    Softmax,
    Layer(Layer)
}

#[derive(Debug)]
pub struct Layer {
    number_of_inputs: usize,
    number_of_outputs: usize,
    weights: Array2<f64>,
    biases: Array1<f64>
}

impl Layer {
    pub fn new(inputs: usize, outputs: usize) -> Layer {
        let mut rng = SmallRng::from_entropy();
        let seeded_weights = vec![rng.gen_range(-1.0..1.0); inputs * outputs];
        let weights = Array2::from_shape_vec((inputs, outputs), seeded_weights)
            .expect("Error convertiong seeded Layer values into Array2"); 
        let biases = Array1::from_vec(vec![rng.gen_range(-1.0..1.0); outputs]);
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

#[derive(Debug, Default)]
pub struct Model {
    built: bool,
    outputs: usize,
    pub layers: Vec<Activation>
}

impl Model {
    pub fn add_layer(mut self, inputs: usize, outputs: usize) -> Model {
        self.layers.push(Activation::Layer(Layer::new(inputs, outputs)));
        self
    }

    pub fn add_sigmoid(mut self) -> Model {
        self.layers.push(Activation::Sigmoid);
        self
    }

    pub fn add_softmax(mut self) -> Model {
        self.layers.push(Activation::Softmax);
        self
    }

    pub fn build(mut self) -> Result<Model> {
        // validate the model
        let mut current_outputs : usize = 0;
        for (key, layer) in self.layers.iter().enumerate() {
            if let Activation::Layer(l) = layer {
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
        self.outputs = current_outputs;
        Ok(self)
    }

    pub fn forward(&self, inputs_array: Array1<f64>) -> Array1<f64> {
        let mut output_array = inputs_array;
        for layer in &self.layers {
            match layer {
                Activation::Sigmoid => {
                    output_array = output_array.map(sigmoid);
                },
                Activation::Softmax => {
                    output_array = softmax_stable(output_array);
                },
                Activation::Layer(layer) => {
                    output_array = layer.calculate_outputs(output_array);
                }
            }
        }
        output_array
    }

    pub fn learn(&self, inputs_array: Array1<f64>, answer: &f64) -> Result<()> {
        let mut answer_array : Array1<f64> = Array1::zeros(self.outputs);
        let mut output_array = inputs_array;
        answer_array[*answer as usize] = 1.0; 
        let mut outputs : Vec<Array1<f64>> = vec![];
        for layer in &self.layers {
            match layer {
                Activation::Sigmoid => {
                    output_array = output_array.map(sigmoid);
                },
                Activation::Softmax => {
                    output_array = softmax_stable(output_array);
                },
                Activation::Layer(layer) => {
                    output_array = layer.calculate_outputs(output_array);
                }
            }
            outputs.push(output_array.clone());
        }

        let probability_array = outputs.last().unwrap();
        assert_eq!(probability_array.len(), answer_array.len());

        // distance from goal 
        //let error = (probability_array - answer_array)* d_softmax_stable();
        Ok(())
    }
}

