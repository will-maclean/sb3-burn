use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    prelude::Backend,
    tensor::{activation::relu, Tensor},
};

use crate::common::{
    agent::Policy,
    distributions::{distribution::BaseDistribution, normal::Normal},
    utils::modules::MLP,
};

#[derive(Debug, Module)]
pub struct PiModel<B: Backend> {
    mlp: MLP<B>,
    loc_head: Linear<B>,
    scale_head: Linear<B>,
    // dist: SquashedDiagGaussianDistribution<B>,
    n_actions: usize,
}

impl<B: Backend> PiModel<B> {
    pub fn new(obs_size: usize, n_actions: usize, device: &B::Device) -> Self {
        Self {
            mlp: MLP::new(&[obs_size, 4].to_vec(), device),
            scale_head: LinearConfig::new(4, n_actions).init(device),
            loc_head: LinearConfig::new(4, n_actions).init(device),
            // dist: SquashedDiagGaussianDistribution::new(256, n_actions, device, 1e-6),
            n_actions,
        }
    }
}

impl<B: Backend> PiModel<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let latent = relu(self.mlp.forward(obs.clone()));
        let loc = self.loc_head.forward(latent.clone());
        let log_scale = self.scale_head.forward(latent);
        let log_scale = log_scale.tanh();

        let min_log_scale = -20.0;
        let max_log_scale = 2.0;

        let log_scale = min_log_scale + 0.5 * (max_log_scale - min_log_scale) * (log_scale + 1.0);

        (loc, log_scale)
    }
    pub fn act(&mut self, obs: &Tensor<B, 1>, deterministic: bool) -> Tensor<B, 1> {
        let (loc, log_scale) = self.forward(obs.clone().unsqueeze_dim(0));

        if deterministic {
            loc.tanh().squeeze(0)
        } else {
            let scale = log_scale.exp();
            let dist = Normal::new(loc, scale);
            let x_t = dist.rsample();
            let action = x_t.tanh();

            action.squeeze(0)
        }

        // self.dist.actions_from_obs(latent, deterministic).squeeze(0)
    }

    pub fn act_log_prob(&mut self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let (loc, log_scale) = self.forward(obs);
        let scale = log_scale.exp();
        let dist = Normal::new(loc, scale);
        let x_t = dist.rsample();
        let action = x_t.clone().tanh();
        let log_prob = dist.log_prob(x_t);

        (action, log_prob)

        // let latent = self.mlp.forward(obs.clone());
        // let actions = self.dist.actions_from_obs(latent, false);
        // let log_prob = self.dist.log_prob(actions.clone());
        //
        // (actions, log_prob)
    }
}

#[derive(Debug, Module)]
pub struct QModel<B: Backend> {
    mlp: MLP<B>,
}

impl<B: Backend> QModel<B> {
    pub fn new(obs_size: usize, n_actions: usize, device: &B::Device) -> Self {
        Self {
            mlp: MLP::new(&[obs_size + n_actions, 4, 1].to_vec(), device),
        }
    }
}

impl<B: Backend> Policy<B> for QModel<B> {
    fn update(&mut self, from: &Self, tau: Option<f32>) {
        self.mlp.update(&from.mlp, tau)
    }
}

impl<B: Backend> QModel<B> {
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.mlp.forward(x)
    }

    pub fn q_from_actions(&self, obs: Tensor<B, 2>, actions: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = Tensor::cat(Vec::from([obs, actions]), 1);

        self.forward(x)
    }
}

#[derive(Debug, Module)]
pub struct QModelSet<B: Backend> {
    qs: Vec<QModel<B>>,
}

impl<B: Backend> QModelSet<B> {
    pub fn new(obs_size: usize, n_actions: usize, device: &B::Device, n_critics: usize) -> Self {
        let mut qs = Vec::new();

        for _ in 0..n_critics {
            qs.push(QModel::new(obs_size, n_actions, device));
        }

        Self { qs }
    }
    pub fn q_from_actions(&self, obs: Tensor<B, 2>, actions: Tensor<B, 2>) -> Vec<Tensor<B, 2>> {
        self.qs
            .iter()
            .map(|q| q.q_from_actions(obs.clone(), actions.clone()))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.qs.len()
    }
}

impl<B: Backend> Policy<B> for QModelSet<B> {
    fn update(&mut self, from: &Self, tau: Option<f32>) {
        for i in 0..self.qs.len() {
            self.qs[i].update(&from.qs[i], tau);
        }
    }
}
