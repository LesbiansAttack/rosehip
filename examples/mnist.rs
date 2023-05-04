use rosehip::{model, dataset};
use std::time::Instant;
use ndarray::Zip;

fn main() {
    let mnist_dataset = dataset::Dataset::new_from_mnist(50_000, 10_000, 10_000);
    let model = model::ModelBuilder
        ::default()
        .add_linear_layer(784, 128)
        .add_sigmoid()
        .add_linear_layer(128, 10)
        .add_softmax()
        .build()
        .unwrap();
    let now = Instant::now();
    Zip::from(mnist_dataset.training_data.rows())
        .and(mnist_dataset.training_labels.rows())
        .for_each(|data, label| { 
            model.forward_backward(data.to_owned(), label.get(0).unwrap());
        });
    println!("took: {:.2?}", now.elapsed());
}
