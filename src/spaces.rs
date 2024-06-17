use dyn_clone::DynClone;
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Defines a space in which a action, observation, or other may exist
pub trait Space<T: Clone>: DynClone {
    /// tests whether the sample is contained within the space
    fn contains(&self, sample: &T) -> bool;

    /// randomly samples from the space
    fn sample(&mut self) -> T;

    /// seeds the rng for the space
    fn seed(&mut self, seed: [u8; 32]);

    /// returns some semantic representation of the space of
    /// the space, to be used for initialising models
    fn shape(&self) -> T;
}

#[derive(Debug, Clone)]
pub struct Discrete {
    n: usize,
    rng: StdRng,
}

impl From<usize> for Discrete {
    fn from(value: usize) -> Self {
        Self {
            n: value,
            rng: StdRng::from_entropy(),
        }
    }
}

impl Space<usize> for Discrete {
    fn contains(&self, sample: &usize) -> bool {
        *sample < self.n
    }

    fn sample(&mut self) -> usize {
        self.rng.gen_range(0..self.n)
    }

    fn seed(&mut self, seed: [u8; 32]) {
        self.rng = StdRng::from_seed(seed);
    }

    fn shape(&self) -> usize {
        self.n
    }
}

#[derive(Debug, Clone)]
pub struct BoxSpace<T> {
    low: T,
    high: T,
    rng: StdRng,
}

impl From<(Vec<f32>, Vec<f32>)> for BoxSpace<Vec<f32>> {
    fn from(value: (Vec<f32>, Vec<f32>)) -> Self {
        Self {
            low: value.0,
            high: value.1,
            rng: StdRng::from_entropy(),
        }
    }
}

impl Space<Vec<f32>> for BoxSpace<Vec<f32>> {
    fn contains(&self, sample: &Vec<f32>) -> bool {
        if sample.len() != self.low.len(){
            return false;
        }

        sample.iter()
            .zip(self.low.iter())
            .zip(self.high.iter())
            .all(|((&s, &l), &h)| l <= s && s <= h)
    }

    fn sample(&mut self) -> Vec<f32> {
        (0..self.low.len())
            .map(|_| {
                let low_bound = self.low.iter().cloned().fold(f32::INFINITY, f32::min);
                let high_bound = self.high.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                self.rng.gen_range(low_bound..high_bound)
            })
            .collect()
    }

    fn seed(&mut self, seed: [u8; 32]) {
        self.rng = StdRng::from_seed(seed);
    }

    fn shape(&self) -> Vec<f32> {
        self.low.clone()
    }
}
