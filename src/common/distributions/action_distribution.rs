use burn::{module::{Module, Param}, nn::{Linear, LinearConfig}, tensor::{backend::Backend, Shape, Tensor}};

use crate::common::{agent::Policy, utils::module_update::update_linear};

use super::{distribution::BaseDistribution, normal::Normal};

pub trait ActionDistribution<B> : Policy<B>
where
    B: Backend,
    // SD: BaseDistribution<B, 1>
{
    /// takes in a batched input and returns the
    /// batched log prob
    fn log_prob(&self, sample: Tensor<B, 2>) -> Tensor<B, 2>;

    /// returns the entropy of the distribution
    fn entropy(&self) -> Tensor<B, 2>;

    /// returns the mode of the distribution
    fn mode(&self) -> Tensor<B, 2>;

    /// returns an unbatched sample from the distribution
    fn sample(&mut self) -> Tensor<B, 2>;


    fn get_actions(&mut self, deterministic: bool) -> Tensor<B, 2>{
        if deterministic {
            self.mode()
        } else {
            self.sample()
        }
    }

    fn actions_from_obs(&mut self, obs: Tensor<B, 2>) -> Tensor<B, 2>;
}

/// Continuous actions are usually considered to be independent,
/// so we can sum components of the ``log_prob`` or the entropy.
/// 
/// # Shapes
/// t: (batch, n_actions) or (batch)
/// return: (batch) for (batch, n_actions) input, or (1) for (batch) input
// fn sum_independent_dims<B: Backend>(t: Tensor<B, 1>) -> Tensor<B, 1>{
//     t.sum()
// }

// fn sum_independent_dims_batched<B: Backend>(t: Tensor<B, 2>) -> Tensor<B, 1>{
//     t.sum_dim(1).squeeze(1)
// }

#[derive(Debug, Module)]
pub struct DiagGaussianDistribution<B: Backend>{
    means: Linear<B>,
    log_std: Param<Tensor<B, 1>>,
    dist: Normal<B, 2>,
}

impl<B: Backend> DiagGaussianDistribution<B>{
    pub fn new(latent_dim: usize, action_dim: usize, log_std_init: f32, device: &B::Device) -> Self {
        // create the distribution with dummy values for now
        let loc: Tensor<B, 2> = Tensor::ones(Shape::new([action_dim]), &Default::default()).unsqueeze_dim(0);
        let std: Tensor<B, 2> = Tensor::ones(Shape::new([action_dim]), &Default::default()).mul_scalar(log_std_init).unsqueeze_dim(0);
        let dist: Normal<B, 2> = Normal::new(loc, std);

        Self { 
            means: LinearConfig::new(latent_dim, action_dim).init(device),
            log_std: Param::from_tensor(Tensor::ones(Shape::new([action_dim]), device).mul_scalar(log_std_init)),
            dist: dist.no_grad()
        }
    }
}

impl<B: Backend>ActionDistribution<B> for DiagGaussianDistribution<B>{
    fn log_prob(&self, sample: Tensor<B, 2>) -> Tensor<B, 2> {
        self.dist.log_prob(sample)
    }

    fn entropy(&self) -> Tensor<B, 2> {
        self.dist.entropy()
    }

    fn mode(&self) -> Tensor<B, 2> {
        self.dist.mode()
    }

    fn sample(&mut self) -> Tensor<B, 2> {
        self.dist.rsample()
    }
    
    fn actions_from_obs(&mut self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let scale = self.log_std.val().clone().exp().unsqueeze_dim(0).repeat(0, obs.shape().dims[0]);
        let mean = self.means.forward(obs);
        self.dist = Normal::new(mean, scale).no_grad();

        self.sample()
    }
}

impl<B: Backend> Policy<B> for DiagGaussianDistribution<B>{
    fn update(&mut self, from: &Self, tau: Option<f32>) {
        self.means = update_linear(&from.means, self.means.clone(), tau);
        //TODO: update self.log_std
    }
}


// #[derive(Clone, Debug)]
// pub struct StateDependentNoiseDistribution<B: Backend>{
//     means: Linear<B>,
//     log_std: Param<Tensor<B, 1>>,
// }

// impl<B: Backend> StateDependentNoiseDistribution<B>{
//     fn sample_noise(&mut self, latent_sde: Tensor<B, 1>) -> Tensor<B, 1>{
//         todo!()
//     }
// }

#[cfg(test)]
mod test {
    use burn::{backend::{Autodiff, NdArray}, tensor::{Distribution, Shape, Tensor}};

    use crate::common::distributions::action_distribution::ActionDistribution;

    use super::DiagGaussianDistribution;

    #[test]
    fn test_diag_gaussian_dist(){
        type Backend = Autodiff<NdArray>;
        let latent_size = 10;
        let action_size = 3;
        let log_std_init = 0.4;
        let mut dist: DiagGaussianDistribution<Backend> = DiagGaussianDistribution::new(
            latent_size,
            action_size,
            log_std_init,
            &Default::default(),
        );

        // create some dummy obs
        let batch_size = 6;
        let dummy_obs: Tensor<Backend, 2> = Tensor::random(
            Shape::new([batch_size, latent_size]), 
            Distribution::Normal(0.0, 1.0), 
            &Default::default(),
        );

        let action_sample = dist.actions_from_obs(dummy_obs);
        let log_prob = dist.log_prob(action_sample);

        // build a dummy loss function on the log prob and 
        // make sure we can do a backwards pass
        let dummy_loss = log_prob.sub_scalar(0.1).powi_scalar(2).mean();
        let _grads = dummy_loss.backward();

        // make sure no panic
        dist.mode();
        dist.entropy();
    }
}