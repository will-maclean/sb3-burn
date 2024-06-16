use rand::{rngs::StdRng, SeedableRng};
use dyn_clone::DynClone;


pub trait Space<T: Clone>: DynClone {
    fn contains(&self, sample: &T) -> bool;
    fn sample(&mut self) -> T;
    fn seed(&mut self);
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
}