use std::sync::{LazyLock, Mutex};

use burn::tensor::cast::ToElement;
use burn::tensor::{backend::Backend, Distribution, Tensor};
use dyn_clone::DynClone;
use rand::{rngs::StdRng, Rng, SeedableRng};

pub static SHARED_RNG: LazyLock<Mutex<StdRng>> =
    LazyLock::new(|| Mutex::new(StdRng::seed_from_u64(1234)));

pub fn seed_spaces_rng(seed: u64) {
    *SHARED_RNG.lock().unwrap() = StdRng::seed_from_u64(seed);
}

/// Defines a space in which a action, observation, or other may exist
pub trait Space<T: Clone>: DynClone {
    /// tests whether the sample is contained within the space
    fn contains(&self, sample: &T) -> bool;

    /// randomly samples from the space
    fn sample(&mut self) -> T;

    /// returns some semantic representation of the space of
    /// the space, to be used for initialising models
    fn shape(&self) -> T;
}

/// Defines a Discrete Space.
///
/// A Discrete space is a space on `usize` where samples
/// are drawn uniformly from `[0, n)`.
#[derive(Debug, Clone)]
pub struct Discrete {
    /// The upper bound on the space
    n: usize,
}

impl From<usize> for Discrete {
    fn from(value: usize) -> Self {
        Self { n: value }
    }
}

impl Space<usize> for Discrete {
    fn contains(&self, sample: &usize) -> bool {
        *sample < self.n
    }

    fn sample(&mut self) -> usize {
        let mut rng = SHARED_RNG.lock().unwrap();
        rng.random_range(0..self.n)
    }

    fn shape(&self) -> usize {
        self.n
    }
}

/// Defines a `BoxSpace<T>`.
///
/// A `BoxSpace` is an n-dimensional container on
/// some generic `T`, where `T` is classically some
/// form of number. Current implementations are
/// for `Vec<f32>`, but it is also possible to use
/// e.g. `Tensor<B, D>`.
#[derive(Debug, Clone)]
pub struct BoxSpace<T> {
    /// The lower bound on the space
    low: T,

    /// The upper bound on the space
    high: T,
}

impl From<(Vec<f32>, Vec<f32>)> for BoxSpace<Vec<f32>> {
    fn from(value: (Vec<f32>, Vec<f32>)) -> Self {
        Self {
            low: value.0,
            high: value.1,
        }
    }
}

impl Space<Vec<f32>> for BoxSpace<Vec<f32>> {
    fn contains(&self, sample: &Vec<f32>) -> bool {
        if sample.len() != self.low.len() {
            return false;
        }

        sample
            .iter()
            .zip(self.low.iter())
            .zip(self.high.iter())
            .all(|((&s, &l), &h)| l <= s && s <= h)
    }

    fn sample(&mut self) -> Vec<f32> {
        let mut rng = SHARED_RNG.lock().unwrap();
        (0..self.low.len())
            .map(|i| rng.random_range(self.low[i]..=self.high[i]))
            .collect()
    }

    fn shape(&self) -> Vec<f32> {
        self.low.clone()
    }
}

impl BoxSpace<Vec<f32>> {
    pub fn low(&self) -> &Vec<f32> {
        &self.low
    }

    pub fn high(&self) -> &Vec<f32> {
        &self.high
    }
}

impl<B: Backend, const D: usize> From<(Tensor<B, D>, Tensor<B, D>)> for BoxSpace<Tensor<B, D>> {
    fn from(value: (Tensor<B, D>, Tensor<B, D>)) -> Self {
        Self {
            low: value.0,
            high: value.1,
        }
    }
}

impl<B: Backend, const D: usize> Space<Tensor<B, D>> for BoxSpace<Tensor<B, D>> {
    fn contains(&self, sample: &Tensor<B, D>) -> bool {
        if sample.shape() != self.low.shape() {
            return false;
        }

        sample
            .clone()
            .greater_equal(self.low.clone())
            .all()
            .into_scalar()
            .to_bool()
            & sample
                .clone()
                .lower_equal(self.low.clone())
                .all()
                .into_scalar()
                .to_bool()
    }

    fn sample(&mut self) -> Tensor<B, D> {
        let shape = self.low.shape();
        let sample: Tensor<B, D> =
            Tensor::random(shape, Distribution::Uniform(0.0, 1.0), &self.low.device());
        let range = self.high.clone().sub(self.low.clone());
        let sample = sample.mul(range).add(self.low.clone());

        sample
    }

    fn shape(&self) -> Tensor<B, D> {
        self.low.clone()
    }
}

#[cfg(test)]
mod test {
    use crate::common::spaces::{BoxSpace, Discrete, Space};

    #[test]
    fn test_discrete_space() {
        let mut space = Discrete::from(2);

        assert_eq!(space.shape(), 2);
        assert!(space.contains(&0));
        assert!(space.contains(&1));
        assert!(!space.contains(&2));

        let sample = space.sample();
        assert!((sample == 0) | (sample == 1))
    }

    #[test]
    fn test_box_f32_space() {
        let low = vec![0.0, -0.1, 0.1];
        let high = vec![1.0, 1.1, 0.9];

        let mut space = BoxSpace::from((low, high));

        assert_eq!(space.shape().len(), 3);

        assert!(space.contains(&vec![0.0, 1.1, 0.3]));
        assert!(!space.contains(&vec![30.0, 1.1, 0.3]));

        let sample = space.sample();
        assert!(sample.len() == 3);
    }
}
