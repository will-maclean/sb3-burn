use burn::{
    module::Module, nn::{Linear, LinearConfig}, prelude::Backend, tensor::Tensor
};

use crate::common::{
    agent::Policy,
    distributions::{distribution::BaseDistribution, normal::Normal},
    utils::modules::MLP,
};

#[derive(Debug, Module)]
pub struct PiModel<B: Backend> {
    mlp: MLP<B>,
    means: Linear<B>,
    log_stds: Linear<B>,
}

impl<B: Backend> PiModel<B> {
    pub fn new(obs_size: usize, n_actions: usize, device: &B::Device) -> Self {
        Self {
            mlp: MLP::new(&[obs_size, 256, 256].to_vec(), device),
            means: LinearConfig::new(256, n_actions).init(device),
            log_stds: LinearConfig::new(256, n_actions).init(device),
        }
    }
}

impl<B: Backend> PiModel<B> {

    pub fn get_dist_params(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let latent = self.mlp.forward(obs);

        let means = self.means.forward(latent.clone());

        let log_stds = self.log_stds.forward(latent); // extract from model output
        let log_stds = log_stds.clamp(-20, 2); // clamp for safety
        let stds = log_stds.exp(); // exponentiate to convert to std

        (means, stds)
    }

    pub fn act(&self, obs: &Tensor<B, 1>, deterministic: bool) -> Tensor<B, 1> {
        let (means, stds) = self.get_dist_params(obs.clone().unsqueeze_dim(0));

        if deterministic {
            means.squeeze(0)
        } else {
            let dist = Normal::new(means.squeeze(0), stds.squeeze(0));
    
            dist.rsample()
        }
    }

    pub fn act_log_prob(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let (means, stds) = self.get_dist_params(obs);

        let dist = Normal::new(means, stds);

        let actions = dist.rsample();
        let log_prob = dist.log_prob(actions.clone());

        (actions, log_prob)
    }
}

#[derive(Debug, Module)]
pub struct QModel<B: Backend> {
    mlp: MLP<B>,
}

impl<B: Backend> QModel<B> {
    pub fn new(obs_size: usize, n_actions: usize, device: &B::Device) -> Self {
        Self {
            mlp: MLP::new(
                &[obs_size + n_actions, 256, 256, n_actions].to_vec(),
                device,
            ),
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
