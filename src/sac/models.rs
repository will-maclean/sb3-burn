use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    prelude::Backend,
    tensor::{activation::softplus, Tensor},
};

use crate::common::{
    agent::Policy,
    distributions::{distribution::BaseDistribution, normal::Normal},
    utils::{modules::MLP, set_linear_bias},
};

#[derive(Debug, Module)]
pub struct PiModel<B: Backend> {
    mlp: MLP<B>,
    loc_head: Linear<B>,
    scale_head: Linear<B>,
    // dist: SquashedDiagGaussianDistribution<B>,
    n_actions: usize,
}

const LOG_STD_MIN: f32 = -5.0;
const LOG_STD_MAX: f32 = 2.0;

impl<B: Backend> PiModel<B> {
    pub fn new(obs_size: usize, n_actions: usize, hidden_size: usize, device: &B::Device) -> Self {
        let loc_head: Linear<B> =
            set_linear_bias(LinearConfig::new(hidden_size, n_actions).init(device), 0.0);

        let scale_head: Linear<B> =
            set_linear_bias(LinearConfig::new(hidden_size, n_actions).init(device), 0.0);

        Self {
            mlp: MLP::new(
                &[obs_size, hidden_size, hidden_size].to_vec(),
                device,
                Some(0.0),
            ),
            scale_head,
            loc_head,
            // dist: SquashedDiagGaussianDistribution::new(hidden_size, n_actions, device, 1e-6),
            n_actions,
        }
    }
}

impl<B: Backend> PiModel<B> {
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

    fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let latent = self.mlp.forward(obs.clone());
        let loc = self.loc_head.forward(latent.clone());
        let log_scale = self
            .scale_head
            .forward(latent)
            .clamp(LOG_STD_MIN, LOG_STD_MAX);

        (loc, log_scale)
    }

    pub fn act_log_prob(&mut self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let (loc, log_scale) = self.forward(obs);
        let scale = log_scale.exp();
        let dist = Normal::new(loc, scale);
        let x_t = dist.rsample();
        let action = x_t.clone().tanh();
        let log_prob = dist.log_prob(x_t.clone());

        // This is the log prob calculation I find in most places, that I was using
        // previously, not working!
        //
        // let log_prob = log_prob
        //     - ((1 - action.clone().powi_scalar(2) + 1e-6) as Tensor<B, 2>)
        //         .log()
        //         .sum_dim(1);

        // Found this one online in some other implementations
        let log_prob = log_prob
            - (2.0 * (2.0 as f32).ln() - action.clone() - softplus(-2.0 * action.clone(), 1.0))
                .sum_dim(1);

        (action, log_prob)
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
