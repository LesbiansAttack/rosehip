use ml_learn::{model, dataset};
use ndarray::array;

fn main() {
    let mnist_dataset = dataset::Dataset::new_from_mnist(50_000, 10_000, 10_000);
    let layer = model::Layer::new(2, 3);
    println!("{:?}", layer.calculate_outputs(array![1.0, 1.0]));
}
