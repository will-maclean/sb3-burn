use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::spaces::{Action, ActionSpace, Obs, ObsT};

pub trait Policy<B: Backend>: Module<B> + Clone {
    fn act(&self, state: &Obs, action_space: ActionSpace) -> Action;
    fn predict(&self, state: ObsT<B, 2>) -> Tensor<B, 2>;
    fn update(&mut self, from: &Self, tau: Option<f32>);
}
