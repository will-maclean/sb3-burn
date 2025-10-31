use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    prelude::Backend,
    tensor::Tensor,
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
    pub fn new(obs_size: usize, n_actions: usize, hidden_size: usize, device: &B::Device) -> Self {
        let loc_head: Linear<B> = LinearConfig::new(hidden_size, n_actions).init(device);

        let scale_head: Linear<B> = LinearConfig::new(hidden_size, n_actions).init(device);

        Self {
            mlp: MLP::new(&[obs_size, hidden_size, hidden_size].to_vec(), device),
            scale_head,
            loc_head,
            // dist: SquashedDiagGaussianDistribution::new(hidden_size, n_actions, device, 1e-6),
            n_actions,
        }
    }
}

impl<B: Backend> PiModel<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let latent = self.mlp.forward(obs.clone());
        let loc = self.loc_head.forward(latent.clone());
        let log_scale = self.scale_head.forward(latent);
        let log_scale = log_scale.clamp(-20.0, 2.0);

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

        // let latent = self.mlp.forward(obs.clone().unsqueeze());
        // self.dist.actions_from_obs(latent, deterministic).squeeze(0)
    }

    pub fn act_log_prob(&mut self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let (loc, log_scale) = self.forward(obs);
        let scale = log_scale.exp();
        let dist = Normal::new(loc, scale);
        let x_t = dist.rsample();
        let action = x_t.clone().tanh();
        let log_prob = dist.log_prob(x_t.clone());

        let log_prob: Tensor<B, 2> = log_prob
            - action
                .clone()
                .powi_scalar(2)
                .neg()
                .add_scalar(1.0)
                .add_scalar(1e-6)
                .log();

        // let log_prob = log_prob
        //     - (2.0 * (2.0 as f32).ln() - x_t.clone() - softplus(-2.0 * x_t, 1.0)).sum_dim(1);

        (action, log_prob)

        // let latent = self.mlp.forward(obs.clone());
        // let (actions, log_prob) = self.dist.actions_from_obs_with_log_probs(latent, false);

        // (actions, log_prob)
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
