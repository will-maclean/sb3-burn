use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Shape, Tensor},
};

use crate::common::{
    agent::Policy,
    utils::{disp_tensorf, module_update::update_linear},
};

use super::{distribution::BaseDistribution, normal::Normal};

pub trait ActionDistribution<B>: Policy<B>
where
    B: Backend,
{
    /// takes in a batched input and returns the
    /// batched log prob
    fn log_prob(&self, sample: Tensor<B, 2>) -> Tensor<B, 2>;

    /// returns the entropy of the distribution
    fn entropy(&self) -> Tensor<B, 2>;

    /// returns the mode of the distribution
    fn mode(&mut self) -> Tensor<B, 2>;

    /// returns an unbatched sample from the distribution
    fn sample(&mut self) -> Tensor<B, 2>;

    fn get_actions(&mut self, deterministic: bool) -> Tensor<B, 2> {
        if deterministic {
            self.mode()
        } else {
            self.sample()
        }
    }

    fn actions_from_obs(&mut self, obs: Tensor<B, 2>, deterministic: bool) -> Tensor<B, 2>;
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

// fn sum_independent_dims_batched<B: Backend>(t: Tensor<B, 1>) -> Tensor<B, 1>{
//     t.sum_dim(1).squeeze(1)
// }

#[derive(Debug, Module)]
pub struct DiagGaussianDistribution<B: Backend> {
    means: Linear<B>,
    log_std: Linear<B>,
    dist: Normal<B, 2>,
}

impl<B: Backend> DiagGaussianDistribution<B> {
    pub fn new(latent_dim: usize, action_dim: usize, device: &B::Device) -> Self {
        // create the distribution with dummy values for now
        let loc: Tensor<B, 2> =
            Tensor::<B, 1>::ones(Shape::new::<1>([action_dim]), &Default::default())
                .unsqueeze_dim(0);
        let std: Tensor<B, 2> =
            Tensor::<B, 1>::ones(Shape::new([action_dim]), &Default::default()).unsqueeze_dim(0);
        let dist: Normal<B, 2> = Normal::new(loc, std);

        Self {
            means: LinearConfig::new(latent_dim, action_dim).init(device),
            log_std: LinearConfig::new(latent_dim, action_dim).init(device),
            dist,
        }
    }
}

impl<B: Backend> ActionDistribution<B> for DiagGaussianDistribution<B> {
    fn log_prob(&self, sample: Tensor<B, 2>) -> Tensor<B, 2> {
        let log_prob = self.dist.log_prob(sample);

        // TODO: add sum_independent_dims when multi-dim actions are supported
        log_prob
    }

    fn entropy(&self) -> Tensor<B, 2> {
        self.dist.entropy()
    }

    fn mode(&mut self) -> Tensor<B, 2> {
        self.dist.mode()
    }

    fn sample(&mut self) -> Tensor<B, 2> {
        self.dist.rsample()
    }

    fn actions_from_obs(&mut self, obs: Tensor<B, 2>, deterministic: bool) -> Tensor<B, 2> {
        let scale: Tensor<B, 2> = self.log_std.forward(obs.clone()).clamp(-20, 2).exp();

        let loc = self.means.forward(obs);

        self.dist = Normal::new(loc.clone(), scale);

        if deterministic {
            loc
        } else {
            self.dist.rsample()
        }
    }
}

impl<B: Backend> Policy<B> for DiagGaussianDistribution<B> {
    fn update(&mut self, from: &Self, tau: Option<f32>) {
        self.means = update_linear(&from.means, self.means.clone(), tau);
        //TODO: update self.log_std
    }
}

#[derive(Debug, Module)]
pub struct SquashedDiagGaussianDistribution<B: Backend> {
    diag_gaus_dist: DiagGaussianDistribution<B>,
    epsilon: f32,
}

impl<B: Backend> SquashedDiagGaussianDistribution<B> {
    pub fn new(latent_dim: usize, action_dim: usize, device: &B::Device, epsilon: f32) -> Self {
        Self {
            diag_gaus_dist: DiagGaussianDistribution::new(latent_dim, action_dim, device),
            epsilon,
        }
    }
}

impl<B: Backend> ActionDistribution<B> for SquashedDiagGaussianDistribution<B> {
    fn log_prob(&self, sample: Tensor<B, 2>) -> Tensor<B, 2> {
        // disp_tensorf("actions in", &sample);

        let actions = tanh_bijector_inverse(sample.clone(), self.epsilon);

        // disp_tensorf("actions post atanh", &actions);

        let log_prob = self.diag_gaus_dist.log_prob(actions).sum_dim(1);

        // disp_tensorf("first log prob", &log_prob);

        // Squash correction (from original SAC implementation)
        // this comes from the fact that tanh is bijective and differentiable
        let out = log_prob
            - sample
                .powi_scalar(2)
                .mul_scalar(-1)
                .add_scalar(1.0 + self.epsilon)
                .log()
                .sum_dim(1);

        // disp_tensorf("second log prob", &out);

        out
    }

    fn entropy(&self) -> Tensor<B, 2> {
        todo!()
    }

    fn mode(&mut self) -> Tensor<B, 2> {
        self.diag_gaus_dist.mode().tanh()
    }

    fn sample(&mut self) -> Tensor<B, 2> {
        self.diag_gaus_dist.sample().tanh()
    }

    fn actions_from_obs(&mut self, obs: Tensor<B, 2>, deterministic: bool) -> Tensor<B, 2> {
        self.diag_gaus_dist
            .actions_from_obs(obs, deterministic)
            .tanh()
    }
}

impl<B: Backend> Policy<B> for SquashedDiagGaussianDistribution<B> {
    fn update(&mut self, from: &Self, tau: Option<f32>) {
        self.diag_gaus_dist.update(&from.diag_gaus_dist, tau)
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

fn tanh_bijector_inverse<B: Backend>(sample: Tensor<B, 2>, eps: f32) -> Tensor<B, 2> {
    let sample = sample.clamp(-1.0 + eps, 1.0 - eps);

    tanh_bijector_atanh(sample)
}

fn tanh_bijector_atanh<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    (x.clone().log1p() - (-x).log1p()).mul_scalar(0.5)
}

#[cfg(test)]
mod test {
    use burn::{
        backend::{Autodiff, NdArray},
        tensor::{Distribution, ElementConversion, Shape, Tensor},
    };

    use crate::common::distributions::action_distribution::{
        ActionDistribution, SquashedDiagGaussianDistribution,
    };

    use super::DiagGaussianDistribution;

    #[test]
    fn test_diag_gaussian_dist() {
        type Backend = Autodiff<NdArray>;
        let latent_size = 10;
        let action_size = 3;
        let mut dist: DiagGaussianDistribution<Backend> =
            DiagGaussianDistribution::new(latent_size, action_size, &Default::default());

        // create some dummy obs
        let dummy_obs: Tensor<Backend, 2> = Tensor::<Backend, 1>::random(
            Shape::new([latent_size]),
            Distribution::Normal(0.0, 1.0),
            &Default::default(),
        )
        .unsqueeze_dim(0);

        let action_sample = dist.actions_from_obs(dummy_obs, false);
        let log_prob = dist.log_prob(action_sample);

        // build a dummy loss function on the log prob and
        // make sure we can do a backwards pass
        let dummy_loss = log_prob.sub_scalar(0.1).powi_scalar(1).mean();
        let _grads = dummy_loss.backward();

        // make sure no panic
        dist.mode();
        dist.entropy();
    }

    #[test]
    fn test_squashed_gaussian() {
        type Backend = NdArray;

        let n_features: usize = 3;
        let n_actions: usize = 2;

        let mut dist: SquashedDiagGaussianDistribution<Backend> =
            SquashedDiagGaussianDistribution::new(n_features, n_actions, &Default::default(), 1e-6);

        let actions = dist.sample();

        assert!(actions
            .abs()
            .max()
            .lower_equal_elem(1.0)
            .all()
            .into_scalar()
            .elem::<bool>())
    }
}
