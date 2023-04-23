use rosehip::{model, dataset};
use ndarray::Zip;

fn main() {
    let mnist_dataset = dataset::Dataset::new_from_mnist(50_000, 10_000, 10_000);
    let model = model::Model
        ::default()
        .add_layer(784, 128)
        .add_sigmoid()
        .add_layer(128, 10)
        .add_softmax()
        .build().unwrap();
    Zip::from(mnist_dataset.training_data.rows())
        .and(mnist_dataset.training_labels.rows())
        .for_each(|data, label| { 
                  println!("{}, {}", label, data);
                  model.learn(data.to_owned(), label.get(0).unwrap());

        });
}
