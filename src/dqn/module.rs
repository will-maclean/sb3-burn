use std::fmt::Debug;

use burn::{
    module::Module,
    nn,
    tensor::{activation::relu, backend::Backend, Tensor},
};

use crate::{
    policy::Policy,
    utils::module_update::{update_conv2d, update_linear},
};

pub trait DQNNet<B: Backend, const D: usize>: Policy<B> {
    fn forward(&self, obs: Tensor<B, D>) -> Tensor<B, 2>;
}

#[derive(Module, Debug)]
pub struct LinearAdvDQNNet<B: Backend> {
    l1: nn::Linear<B>,
    l2: nn::Linear<B>,
    l3: nn::Linear<B>,
    adv: nn::Linear<B>,
}

impl<B: Backend> LinearAdvDQNNet<B> {
    pub fn init(device: &B::Device, obs_size: usize, act_size: usize, hidden_size: usize) -> Self {
        Self {
            l1: nn::LinearConfig::new(obs_size, hidden_size).init(device),
            l2: nn::LinearConfig::new(hidden_size, hidden_size).init(device),
            l3: nn::LinearConfig::new(hidden_size, act_size).init(device),
            adv: nn::LinearConfig::new(hidden_size, 1).init(device),
        }
    }
}

impl<B: Backend> DQNNet<B, 2> for LinearAdvDQNNet<B> {
    fn forward(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = relu(self.l1.forward(state));
        let x = relu(self.l2.forward(x));
        self.adv.forward(x.clone()) - self.l3.forward(x)
    }
}

impl<B: Backend> Policy<B> for LinearAdvDQNNet<B> {
    fn update(&mut self, from: &Self, tau: Option<f32>) {
        self.l1 = update_linear(&from.l1, self.l1.clone(), tau);
        self.l2 = update_linear(&from.l2, self.l2.clone(), tau);
        self.l3 = update_linear(&from.l3, self.l3.clone(), tau);
        self.adv = update_linear(&from.adv, self.adv.clone(), tau);
    }
}

#[derive(Module, Debug)]
pub struct LinearDQNNet<B: Backend> {
    l1: nn::Linear<B>,
    l2: nn::Linear<B>,
    l3: nn::Linear<B>,
}

impl<B: Backend> LinearDQNNet<B> {
    pub fn init(device: &B::Device, obs_size: usize, act_size: usize, hidden_size: usize) -> Self {
        Self {
            l1: nn::LinearConfig::new(obs_size, hidden_size).init(device),
            l2: nn::LinearConfig::new(hidden_size, hidden_size).init(device),
            l3: nn::LinearConfig::new(hidden_size, act_size).init(device),
        }
    }
}

impl<B: Backend> DQNNet<B, 2> for LinearDQNNet<B> {
    fn forward(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = relu(self.l1.forward(state));
        let x = relu(self.l2.forward(x));
        self.l3.forward(x)
    }
}

impl<B: Backend> Policy<B> for LinearDQNNet<B> {
    fn update(&mut self, from: &Self, tau: Option<f32>) {
        self.l1 = update_linear(&from.l1, self.l1.clone(), tau);
        self.l2 = update_linear(&from.l2, self.l2.clone(), tau);
        self.l3 = update_linear(&from.l3, self.l3.clone(), tau);
    }
}

#[derive(Module, Debug)]
pub struct ConvDQNNet<B: Backend> {
    c1: nn::conv::Conv2d<B>,
    c2: nn::conv::Conv2d<B>,
    l1: nn::Linear<B>,
    l2: nn::Linear<B>,
}

impl<B: Backend> ConvDQNNet<B> {
    pub fn init(
        device: &B::Device,
        obs_shape: Tensor<B, 3>,
        act_size: usize,
        hidden_size: usize,
    ) -> Self {
        Self {
            c1: nn::conv::Conv2dConfig::new([obs_shape.shape().dims[0], 4], [3, 3]).init(device),
            c2: nn::conv::Conv2dConfig::new([4, 8], [3, 3]).init(device),
            l1: nn::LinearConfig::new(64, hidden_size).init(device),
            l2: nn::LinearConfig::new(hidden_size, act_size).init(device),
        }
    }
}

impl<B: Backend> DQNNet<B, 4> for ConvDQNNet<B> {
    fn forward(&self, state: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = relu(self.c1.forward(state));
        let x = relu(self.c2.forward(x));
        let x = x.flatten::<2>(1, 3);
        let x = relu(self.l1.forward(x));
        self.l2.forward(x)
    }
}

impl<B: Backend> Policy<B> for ConvDQNNet<B> {
    fn update(&mut self, from: &Self, tau: Option<f32>) {
        self.l1 = update_linear(&from.l1, self.l1.clone(), tau);
        self.l2 = update_linear(&from.l2, self.l2.clone(), tau);
        self.c1 = update_conv2d(&from.c1, self.c1.clone(), tau);
        self.c2 = update_conv2d(&from.c2, self.c2.clone(), tau);
    }
}
