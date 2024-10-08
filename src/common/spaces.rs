use burn::tensor::{backend::Backend, Distribution, Tensor};
use dyn_clone::DynClone;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::cell::RefCell;
use std::rc::Rc;

/// Defines a space in which a action, observation, or other may exist
pub trait Space<T: Clone>: DynClone {
    /// tests whether the sample is contained within the space
    fn contains(&self, sample: &T) -> bool;

    /// randomly samples from the space
    fn sample(&mut self) -> T;

    /// seeds the rng for the space
    fn seed(&mut self, seed: u64);

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

    /// rng used for sampling
    rng: Rc<RefCell<StdRng>>,
}

impl From<usize> for Discrete {
    fn from(value: usize) -> Self {
        Self {
            n: value,
            rng: Rc::new(RefCell::new(StdRng::from_entropy())),
        }
    }
}

impl Space<usize> for Discrete {
    fn contains(&self, sample: &usize) -> bool {
        *sample < self.n
    }

    fn sample(&mut self) -> usize {
        self.rng.borrow_mut().gen_range(0..self.n)
    }

    fn seed(&mut self, seed: u64) {
        let mut rng = self.rng.borrow_mut();
        *rng = StdRng::seed_from_u64(seed);
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

    /// rng used for sampling
    rng: Rc<RefCell<StdRng>>,
}

impl From<(Vec<f32>, Vec<f32>)> for BoxSpace<Vec<f32>> {
    fn from(value: (Vec<f32>, Vec<f32>)) -> Self {
        Self {
            low: value.0,
            high: value.1,
            rng: Rc::new(RefCell::new(StdRng::from_entropy())),
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
        (0..self.low.len())
            .map(|_| {
                let low_bound = self.low.iter().cloned().fold(f32::INFINITY, f32::min);
                let high_bound = self.high.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                self.rng.borrow_mut().gen_range(low_bound..high_bound)
            })
            .collect()
    }

    fn seed(&mut self, seed: u64) {
        let mut rng = self.rng.borrow_mut();
        *rng = StdRng::seed_from_u64(seed);
    }

    fn shape(&self) -> Vec<f32> {
        self.low.clone()
    }
}

impl<B: Backend, const D: usize> From<(Tensor<B, D>, Tensor<B, D>)> for BoxSpace<Tensor<B, D>> {
    fn from(value: (Tensor<B, D>, Tensor<B, D>)) -> Self {
        Self {
            low: value.0,
            high: value.1,
            rng: Rc::new(RefCell::new(StdRng::from_entropy())),
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
            & sample
                .clone()
                .lower_equal(self.low.clone())
                .all()
                .into_scalar()
    }

    fn sample(&mut self) -> Tensor<B, D> {
        let shape = self.low.shape();
        let sample: Tensor<B, D> =
            Tensor::random(shape, Distribution::Uniform(0.0, 1.0), &self.low.device());
        let range = self.high.clone().sub(self.low.clone());
        let sample = sample.mul(range).add(self.low.clone());

        sample
    }

    fn seed(&mut self, seed: u64) {
        let mut rng = self.rng.borrow_mut();
        *rng = StdRng::seed_from_u64(seed);
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
