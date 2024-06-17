use rand::{rngs::StdRng, SeedableRng};
use dyn_clone::DynClone;


/// Defines a space in which a action, observation, or other may exist
pub trait Space<T: Clone>: DynClone {

    /// tests whether the sample is contained within the space
    fn contains(&self, sample: &T) -> bool;

    /// randomly samples from the space
    fn sample(&mut self) -> T;

    /// seeds the rng for the space 
    fn seed(&mut self);

    /// returns some semantic representation of the space of
    /// the space, to be used for initialising models
    fn shape(&self) -> T;
}

#[derive(Debug, Clone)]
pub struct Discrete{
    n: usize,
    rng: StdRng,
}

impl From<usize> for Discrete {
    fn from(value: usize) -> Self {
        Self {
            n: value,
            rng: StdRng::from_entropy()
        }
    }
}

impl Space<usize> for Discrete {
    fn contains(&self, sample: &usize) -> bool {
        todo!()
    }

    fn sample(&mut self) -> usize {
        todo!()
    }

    fn seed(&mut self) {
        todo!()
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
            rng: StdRng::from_entropy()
        }
    }
}

impl Space<Vec<f32>> for BoxSpace<Vec<f32>> {
    fn contains(&self, sample: &Vec<f32>) -> bool {
        todo!()
    }

    fn sample(&mut self) -> Vec<f32> {
        todo!()
    }

    fn seed(&mut self) {
        todo!()
    }

    fn shape(&self) -> Vec<f32> {
        self.low.clone()
    }
}