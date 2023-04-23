use mnist::{Mnist, MnistBuilder};
use ndarray::{Array2};

pub const IMG_SIZE: usize = 28 * 28;

pub struct Dataset {
    pub training_data: Array2<f64>,
    pub training_labels: Array2<f64>,
    pub test_data: Array2<f64>,
    pub test_labels: Array2<f64>
}

impl Dataset {
    pub fn new_from_mnist(training_size: usize, validation_size: usize, test_size: usize) -> Dataset {
        let Mnist {
            trn_img,
            trn_lbl,
            tst_img,
            tst_lbl,
            ..
        } = MnistBuilder::new()
            .label_format_digit()
            .training_set_length(training_size as u32)
            .validation_set_length(validation_size as u32)
            .test_set_length(test_size as u32)
            .finalize();
        Dataset {
            training_data: Array2::from_shape_vec((training_size, IMG_SIZE), trn_img)
                .expect("Error converting training images to Array2 struct.")
                .map(|x| *x as f64 / 256.0),
            training_labels: Array2::from_shape_vec((training_size, 1), trn_lbl)
                .expect("Error converting training labels to Array2 struct.")
                .map(|x| *x as f64),
            test_data: Array2::from_shape_vec((test_size, IMG_SIZE), tst_img)
                .expect("Error converting test images to Array2 struct.")
                .map(|x| *x as f64 / 256.0),
            test_labels: Array2::from_shape_vec((test_size, 1), tst_lbl)
                .expect("Error converting test labels to Array2 struct.")
                .map(|x| *x as f64),

        }
    }

}

