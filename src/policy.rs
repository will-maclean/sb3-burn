use burn::module::Module;
use burn::tensor::backend::Backend;

use crate::{
    algorithm::OfflineAlgParams, buffer::ReplayBuffer, logger::LogItem, spaces::Space
};

pub trait Agent<B: Backend, O: Clone, A: Clone> {
    fn act(&self,
        global_step: usize,
        global_frac: f32,
        obs: &O,
        inference_device: &<B as Backend>::Device,
    ) -> (A, LogItem);

    fn train_step(
        &mut self,
        global_step: usize,
        replay_buffer: ReplayBuffer<O, A>,
        offline_params: &OfflineAlgParams,
        train_device: &B::Device,
    ) -> (Option<f32>, LogItem);

    fn eval(
        &mut self,
        n_eps: usize,
    );

    fn observation_space(&self) -> Box<dyn Space<O>>;
    fn action_space(&self) -> Box<dyn Space<A>>;
}

pub trait Policy<B: Backend>: Module<B> + Clone {
    fn update(&mut self, from: &Self, tau: Option<f32>);
}
