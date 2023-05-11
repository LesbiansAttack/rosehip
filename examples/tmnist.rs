use rosehip::{tmodel, dataset};
use std::{time::Instant};
use ndarray::{Zip, s};
use ndarray_stats::QuantileExt;
use std::env;
use rand::{seq::{index}, thread_rng};

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let mnist_dataset = dataset::Dataset::new_from_mnist(50_000, 10_000, 10_000);
    let mut model = tmodel::ModelBuilder
        ::default()
        .add_linear_layer(784, 128, 0.01)
        .add_sigmoid()
        .add_linear_layer(128, 10, 0.01)
        .add_softmax()
        .build()
        .unwrap();

    let now = Instant::now();

    let epochs = 5000;
    let batch_size = 64;

    let mut rng = thread_rng();

    for i in 0..epochs {

        let indexes = index::sample(&mut rng, 50_000, batch_size);

        let mut accuracy = 0.0;
        for index in indexes {
            let data = mnist_dataset.training_data.slice(s![index, ..]).to_owned().into_shape((784,1)).unwrap();
            let label = mnist_dataset.training_labels.get([index, 0]).unwrap();
            let result = model.forward_backward(data, label).unwrap();
            if result.argmax() == Ok((*label as usize, 0)){
                accuracy += 1.0 / batch_size as f64;
            }
        }
        model.finalize_batch(batch_size);

        if i%100 == 0 {
            let mut val_accuracy = 0.0;
            Zip::from(mnist_dataset.test_data.rows())
                .and(mnist_dataset.test_labels.rows())
                .for_each(|data, label| { 
                    let input = data.to_owned().into_shape((784,1)).unwrap();
                    let result = model.forward(input).unwrap();
                    if result.argmax() == Ok((*label.get(0).unwrap() as usize, 0)) {
                        val_accuracy += 1.0 / 10_000.0;
                    }
                });
            println!("For {}th epoch, training accuracy: {:.4?}, validation accuracy: {:.4?}", i, accuracy, val_accuracy);
        }

    }
    println!("took: {:.2?}", now.elapsed());
}
