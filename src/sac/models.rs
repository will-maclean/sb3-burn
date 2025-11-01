use burn::{module::Module, prelude::Backend, tensor::Tensor};

use crate::common::{
    agent::Policy,
    distributions::action_distribution::{ActionDistribution, SquashedDiagGaussianDistribution},
    utils::modules::MLP,
};

#[derive(Debug, Module)]
pub struct PiModel<B: Backend> {
    mlp: MLP<B>,
    dist: SquashedDiagGaussianDistribution<B>,
    n_actions: usize,
}

impl<B: Backend> PiModel<B> {
    pub fn new(obs_size: usize, n_actions: usize, hidden_size: usize, device: &B::Device) -> Self {
        Self {
            mlp: MLP::new(
                &[obs_size, hidden_size, hidden_size].to_vec(),
                device,
                Some(0.0),
            ),
            dist: SquashedDiagGaussianDistribution::new(hidden_size, n_actions, device, 1e-6),
            n_actions,
        }
    }
}

impl<B: Backend> PiModel<B> {
    pub fn act(&mut self, obs: &Tensor<B, 1>, deterministic: bool) -> Tensor<B, 1> {
        let latent = self.mlp.forward(obs.clone().unsqueeze());
        self.dist
            .actions_from_obs(latent, deterministic)
            .squeeze_dim(0)
    }

    pub fn act_log_prob(&mut self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let latent = self.mlp.forward(obs.clone().unsqueeze());
        self.dist.actions_from_obs_with_log_probs(latent, false)
    }
}

#[derive(Debug, Module)]
pub struct QModel<B: Backend> {
    mlp: MLP<B>,
}

impl<B: Backend> QModel<B> {
    pub fn new(obs_size: usize, n_actions: usize, hidden_size: usize, device: &B::Device) -> Self {
        let mlp = MLP::new(
            &[obs_size + n_actions, hidden_size, hidden_size, 1].to_vec(),
            device,
            Some(0.0),
        );

        Self { mlp: mlp }
    }
}

impl<B: Backend> Policy<B> for QModel<B> {
    fn update(&mut self, from: &Self, tau: Option<f32>) {
        self.mlp.update(&from.mlp, tau);
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
    pub fn new(
        obs_size: usize,
        n_actions: usize,
        hidden_size: usize,
        device: &B::Device,
        n_critics: usize,
    ) -> Self {
        let mut qs = Vec::new();

        for _ in 0..n_critics {
            qs.push(QModel::new(obs_size, n_actions, hidden_size, device));
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
