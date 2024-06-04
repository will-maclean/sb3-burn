use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::{tensor::Tensor};

use crate::spaces::{Space, SpaceSample};

pub trait Policy<B: Backend>: Module<B> + Clone {
    fn act(&self, state: &SpaceSample, action_space: Space) -> SpaceSample;
    fn predict(&self, state: Tensor<B, 2>) -> Tensor<B, 2>;
}
