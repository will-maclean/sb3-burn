use burn::prelude::*;
use rand::{thread_rng, Rng};

#[derive(Clone, Debug)]
pub enum SpaceSample {
    Discrete { space: Space, idx: i32 },
    Continuous { space: Space, data: Vec<f32> },
}

impl SpaceSample {
    pub fn to_tensor<B: Backend>(&self) -> Tensor<B, 1> {
        match self {
            SpaceSample::Discrete { space: _, idx } => {
                let shape = Shape::new([1]);
                let data: Data<f32, 1> = Data::new(vec![(*idx) as f32], shape);
                Tensor::<B, 1>::from_data(data.convert(), &Default::default())
            }
            SpaceSample::Continuous { space: _, data } => {
                let shape = Shape::new([data.len()]);
                let data: Data<f32, 1> = Data::new(data.clone(), shape);

                Tensor::<B, 1>::from_data(data.convert(), &Default::default())
            }
        }
    }

    pub fn to_train_tensor<B: Backend>(&self) -> Tensor<B, 1> {
        match self {
            SpaceSample::Discrete { space, idx } => {
                Tensor::<B, 1>::one_hot(*idx as usize, space.size(), &Default::default())
            }
            SpaceSample::Continuous { space: _, data } => {
                let shape = Shape::new([data.len()]);
                let data: Data<f32, 1> = Data::new(data.clone(), shape);

                Tensor::<B, 1>::from_data(data.convert(), &Default::default())
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum Space {
    Discrete { size: usize },
    Continuous { lows: Vec<f32>, highs: Vec<f32> },
}

impl Space {
    pub fn sample(&self) -> SpaceSample {
        let mut rng = thread_rng();

        match self {
            Space::Discrete { size } => SpaceSample::Discrete {
                space: self.clone(),
                idx: rng.gen_range(0..*size) as i32,
            },
            Space::Continuous { lows, highs } => {
                let random_floats: Vec<f32> = (0..lows.len())
                    .map(|_| {
                        let low_bound = lows.iter().cloned().fold(f32::INFINITY, f32::min);
                        let high_bound = highs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        rng.gen_range(low_bound..high_bound)
                    })
                    .collect();

                SpaceSample::Continuous {
                    space: self.clone(),
                    data: random_floats,
                }
            }
        }
    }

    pub fn size(&self) -> usize {
        match self {
            Space::Discrete { size } => *size,
            Space::Continuous { lows, highs: _ } => lows.len(),
        }
    }
}

pub type ActionSpace = Space;
pub type ObsSpace = Space;
pub type Action = SpaceSample;
pub type Obs = SpaceSample;
pub type ObsT<B, const D: usize, K = Float> = Tensor<B, D, K>;
pub type ActionT<B, const D: usize, K = Float> = Tensor<B, D, K>;

#[cfg(test)]
mod test {
    use burn::{
        backend::NdArray,
        tensor::{Data, Shape, Tensor},
    };

    use super::{Obs, ObsSpace};

    #[test]
    fn test_to_train_tensor() {
        type Backend = NdArray;
        let space = ObsSpace::Discrete { size: 2 };
        let obs = Obs::Discrete { space, idx: 0 };
        let obs_t = obs.to_train_tensor::<Backend>();

        let shape = Shape::new([2]);
        let data: Data<f32, 1> = Data::new(vec![1.0, 0.0], shape);
        let expected = Tensor::<Backend, 1>::from_data(data.convert(), &Default::default());

        assert!(obs_t.equal(expected).all().into_scalar());
    }
}
