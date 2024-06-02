use burn::{nn, tensor::Tensor};
use burn::module::Module;
use burn::tensor::backend::Backend;

use crate::spaces::{Space, SpaceSample};


pub trait Policy<B: Backend>: Module<B> {
    fn act(&self, state: &SpaceSample) -> SpaceSample;
    fn predict(&self, state: Tensor<B, 2>) -> Tensor<B, 2>;
}

#[derive(Debug, Module)]
pub struct DQNNet<B: Backend>{
    l1: nn::Linear<B>,
    l2: nn::Linear<B>,
}

impl<B: Backend> DQNNet<B> {
    pub fn init(&self, device: &B::Device, observation_space: Space, action_space: Space, hidden_size: usize) -> Self {
        match action_space {
            Space::Continuous { lows: _, highs: _ } => panic!("Continuous actions are not supported by DQN"),
            Space::Discrete { size: action_size } => {
                let input_size = observation_space.size();

                Self {
                    l1: nn::LinearConfig::new(input_size, hidden_size).init(device),
                    l2: nn::LinearConfig::new(hidden_size, action_size).init(device),
                }
            },
        }
    }

    pub fn forward(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.l1.forward(state);
        self.l2.forward(x)
    }
}

impl<B: Backend> Policy<B> for DQNNet<B> {    
    fn act(&self, state: &SpaceSample) -> SpaceSample {
        todo!()
    }
    
    fn predict(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
        todo!()
    }
}