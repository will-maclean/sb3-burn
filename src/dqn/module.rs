use burn::{module::Module, nn, tensor::{activation::relu, backend::Backend, Tensor}};

use crate::{policy::Policy, utils::module_update::update_linear};

#[derive(Module, Debug)]
pub struct DQNNet<B: Backend> {
    l1: nn::Linear<B>,
    l2: nn::Linear<B>,
    l3: nn::Linear<B>,
    adv: nn::Linear<B>,
}

impl<B: Backend> DQNNet<B> {
    pub fn init(
        device: &B::Device,
        obs_size: usize,
        act_size: usize,
        hidden_size: usize,
    ) -> Self {
        Self {
            l1: nn::LinearConfig::new(obs_size, hidden_size).init(device),
            l2: nn::LinearConfig::new(hidden_size, hidden_size).init(device),
            l3: nn::LinearConfig::new(hidden_size, act_size).init(device),
            adv: nn::LinearConfig::new(hidden_size, 1).init(device),
        }
    }

    pub fn forward(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = relu(self.l1.forward(state));
        let x = relu(self.l2.forward(x));
        self.adv.forward(x.clone()) - self.l3.forward(x)
    }
}

impl<B: Backend> Policy<B> for DQNNet<B> {
    fn update(&mut self, from: &Self, tau: Option<f32>) {
        self.l1 = update_linear(&from.l1, self.l1.clone(), tau);
        self.l2 = update_linear(&from.l2, self.l2.clone(), tau);
        self.l3 = update_linear(&from.l3, self.l3.clone(), tau);
        self.adv = update_linear(&from.adv, self.adv.clone(), tau);
    }
}