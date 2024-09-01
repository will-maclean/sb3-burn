use burn::tensor::{backend::Backend, Bool, Float, Tensor};
use rand::Rng;

pub mod module_update;
pub mod modules;

const PI: f32 = 3.1415;

pub fn linear_decay(curr_frac: f32, start: f32, end: f32, end_frac: f32) -> f32 {
    if curr_frac > end_frac {
        end
    } else {
        start + curr_frac * (end - start) / end_frac
    }
}

pub fn mean(data: &[f32]) -> f32 {
    product(data) / (data.len() as f32)
}

pub fn product(data: &[f32]) -> f32 {
    data.iter().fold(0.0, |acc, x| acc + x)
}

pub fn generate_random_vector(lows: Vec<f32>, highs: Vec<f32>) -> Vec<f32> {
    if lows.len() != highs.len() {
        panic!("Vectors of lows and highs must have the same length");
    }

    let mut rng = rand::thread_rng();
    let mut random_vector = Vec::with_capacity(lows.len());

    for (low, high) in lows.iter().zip(highs.iter()) {
        if low > high {
            panic!("Each low value must be less than or equal to its corresponding high value");
        }
        random_vector.push(rng.gen_range(*low..=*high));
    }

    random_vector
}

pub fn vec_usize_to_one_hot<B: Backend>(
    data: Vec<usize>,
    classes: usize,
    device: &B::Device,
) -> Tensor<B, 2, Float> {
    Tensor::stack(
        data.iter()
            .map(|d| Tensor::<B, 1>::one_hot(*d, classes, device))
            .collect(),
        0,
    )
}

pub fn angle_normalise(f: f32) -> f32 {
    (f + PI) % (2.0 * PI) - PI
}

pub fn disp_tensorf<B: Backend, const D: usize>(name: &str, t: &Tensor<B, D>) {
    // println!("{name}. {t}\n");
}

pub fn disp_tensorb<B: Backend, const D: usize>(name: &str, t: &Tensor<B, D, Bool>) {
    // println!("{name}. {t}\n");
}

#[cfg(test)]
mod test {
    use assert_approx_eq::assert_approx_eq;

    use burn::{
        backend::{ndarray::NdArrayDevice, NdArray},
        tensor::Tensor,
    };

    use crate::common::utils::{generate_random_vector, linear_decay, mean, vec_usize_to_one_hot};

    #[test]
    fn test_mean() {
        let v = [0.0, 1.0, 2.0];

        assert_eq!(mean(&v), 1.0);

        let v = [];

        assert!(mean(&v).is_nan());
    }

    #[test]
    fn test_linear_decay() {
        assert_approx_eq!(linear_decay(0.0, 1.0, 0.01, 0.8), 1.0, 1e-3f32);
        assert_approx_eq!(linear_decay(0.8, 1.0, 0.01, 0.8), 0.01, 1e-3f32);
        assert_approx_eq!(linear_decay(1.0, 1.0, 0.01, 0.8), 0.01, 1e-3f32);
    }

    #[test]
    fn test_gen_rand_vec() {
        let sample = generate_random_vector(vec![0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0]);

        for s in sample {
            assert!((s >= 0.0) & (s <= 1.0));
        }
    }

    #[should_panic]
    #[test]
    fn test_gen_rand_vec_bad() {
        generate_random_vector(vec![1.0], vec![0.0]);
    }

    #[test]
    fn test_usize_to_one_hot() {
        let ins = vec![0, 1, 2];
        let classes = 4;

        let t: Tensor<NdArray, 2> = vec_usize_to_one_hot(ins, classes, &NdArrayDevice::default());

        assert_eq!(t.shape().dims[0], 3);
        assert_eq!(t.shape().dims[1], 4);
    }
}
