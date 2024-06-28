use burn::{module::Module, tensor::{backend::Backend, Tensor}};

use crate::common::{distributions::action_distribution::{ActionDistribution, DiagGaussianDistribution}, utils::modules::MLP};

use super::utils::ActionLogProb;


#[derive(Debug, Default, Clone)]
pub struct QVals<B: Backend> {
    pub q: Vec<Tensor<B, 2>>,
}

pub trait SACNet<B, O, A, AD> : Module<B>
where
    B: Backend,
    O: Clone,
    AD: ActionDistribution<B>
{
    fn q_vals(&self, obs: Vec<O>, acts: Vec<A>, device: &B::Device) -> QVals<B>;
    fn pi(&mut self, obs: Vec<O>, device: &B::Device) -> ActionLogProb<B, A>;
    fn update_targets(&mut self, tau: Option<f32>);
}

#[derive(Debug, Module)]
pub struct LinearSACNet<B: Backend>{
    feature_extractor: MLP<B>,
    q_nets: Vec<MLP<B>>,
    target_q_nets: Vec<MLP<B>>,
    pi: MLP<B>,
    dist: DiagGaussianDistribution<B>,
}


impl<B: Backend> LinearSACNet<B>{
    fn init(action_dim: usize, n_q_nets: usize, feature_extractor_layers: Vec<usize>, q_net_layers: Vec<usize>, pi_net_layers: Vec<usize>, device: &B::Device) -> Self{
        let latent_dim = feature_extractor_layers[feature_extractor_layers.len() - 1];

        let mut q_nets = Vec::new();
        let mut target_q_nets = Vec::new();

        for _ in 0..n_q_nets{
            let new_q = MLP::new(&q_net_layers, device);

            q_nets.push(new_q.clone());
            target_q_nets.push(new_q.no_grad());
        }

        Self { 
            feature_extractor: MLP::new(&feature_extractor_layers, device), 
            q_nets, 
            target_q_nets, 
            pi: MLP::new(&pi_net_layers, device), 
            dist:  DiagGaussianDistribution::new(latent_dim, action_dim, -3.0, device)
        }
    }
}

impl <B: Backend> SACNet<B, Tensor<B, 1>, Tensor<B, 1>, DiagGaussianDistribution<B>> for LinearSACNet<B>{
    fn q_vals(&self, obs: Vec<Tensor<B, 1>>, acts: Vec<Tensor<B, 1>>, device: &B::Device) -> QVals<B>{
        let obs = Tensor::stack(obs, 0);
        let acts = Tensor::stack(acts, 0);
        let latent = self.feature_extractor.forward(obs);

        let q_in = Tensor::cat(vec![latent, acts], 1);

        let mut q_vals = QVals::default();

        for q in &self.q_nets{
            q_vals.q.push(q.forward(q_in.clone()));
        }

        q_vals
    }
    fn pi(&mut self, obs: Vec<Tensor<B, 1>>, device: &B::Device) -> ActionLogProb<B, Tensor<B, 1>>{
        let x = Tensor::stack(obs, 0);
        let x = self.pi.forward(x);

        let pi = self.dist.actions_from_obs(x);
        let log_pi = self.dist.log_prob(pi.clone());

        let mut actions = Vec::new();

        //FIXME: is there a faster way of doing this?
        for i in 0..pi.shape().dims[0] {
            actions.push(
                pi.clone().slice([i..i+1]).squeeze(0)
            )
        }

        ActionLogProb{
            pi: actions,
            log_pi,
        }

    }
    fn update_targets(&mut self, tau: Option<f32>){}
}