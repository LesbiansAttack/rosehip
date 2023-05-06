use rosehip::{model, dataset, math::softmax_stable};
use std::{time::Instant, vec};
use ndarray::{Zip, Array2, s, array};
use ndarray_stats::QuantileExt;
use std::env;
use rand::{seq::{IteratorRandom, index}, thread_rng};
use eyre::{eyre, Result};

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let mnist_dataset = dataset::Dataset::new_from_mnist(50_000, 10_000, 10_000);
    let mut model = model::ModelBuilder
        ::default()
        .add_linear_layer(784, 128)
        .add_sigmoid()
        .add_linear_layer(128, 10)
        .add_softmax()
        .build()
        .unwrap();

    let now = Instant::now();

    let epochs = 5000;
    let learning_rate = 0.005;
    let batch_size = 128;

    let mut rng = thread_rng();


    for i in 0..epochs {
        let mut outputs: Vec<(usize, Array2<f64>)> = vec![];
        let mut weight_deltas: Vec<Vec<Array2<f64>>> = vec![];
        let mut bias_deltas: Vec<Vec<Array2<f64>>> = vec![];

        let indexes = index::sample(&mut rng, 50_000, batch_size);

        for index in indexes {
            let data = mnist_dataset.training_data.slice(s![index, ..]);
            let label = mnist_dataset.training_labels.get([index, 0]).unwrap();
            let result = model.forward_backward(data.to_owned(), label).unwrap();
            outputs.push((*label as usize, result.0));
            weight_deltas.push(result.1);
            bias_deltas.push(result.2);
        }

        let accuracy = outputs.iter().fold(0.0, |acc, elem| {
            if elem.1.argmax() == Ok((elem.0, 0)) {
                acc + ( 1.0 / (batch_size as f64) )
            } else {
                acc
            }
        });

        let mut sums = weight_deltas[0].clone();
        for delta in weight_deltas {
            for (i, d) in delta.into_iter().enumerate() {
                sums[i] = &sums[i] + d;
            }
        }
        for s in sums.iter_mut() {
            *s /= (batch_size as f64);
        }

        let mut sums_biases = bias_deltas[0].clone();
        for delta in bias_deltas {
            for (i, d) in delta.into_iter().enumerate() {
                sums_biases[i] = &sums_biases[i] + d;
            }
        }
        for s in sums_biases.iter_mut() {
            *s /= (batch_size as f64);
        }

        for (i, (d_w,d_b)) in sums.into_iter().rev().zip(sums_biases.into_iter().rev()).enumerate(){
            let layer = match model.steps.get_mut(2*i).unwrap() {
                model::Step::LinearLayer(l) => Ok(l),
                _ => Err(eyre!("Error When Trying to Adjust Layer")),
            };
            if layer.is_ok() {
                layer.unwrap().adjust_weights(d_w * learning_rate, d_b * learning_rate);
            }
        }

        if i%50 == 0 {
            let mut val_accuracy = 0.0;
            Zip::from(mnist_dataset.test_data.rows())
                .and(mnist_dataset.test_labels.rows())
                .for_each(|data, label| { 
                    let result = model.forward(data.to_owned()).unwrap();
                    if result.argmax() == Ok((*label.get(0).unwrap() as usize, 0)) {
                        val_accuracy += 1.0 / 10_000.0;
                    }
                });

            println!("For {}th epoch: train accuracy: {} | validation accuracy : {}", i, accuracy, val_accuracy);
        }

    }


    // Zip::from(mnist_dataset.training_data.rows())
    //     .and(mnist_dataset.training_labels.rows())
    //     .for_each(|data, label| { 
    //         let result = model.forward_backward(data.to_owned(), label.get(0).unwrap()).unwrap();
    //     });
    println!("took: {:.2?}", now.elapsed());
}
