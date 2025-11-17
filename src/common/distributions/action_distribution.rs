use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Shape, Tensor},
};

use crate::common::{
    agent::Policy,
    utils::module_update::update_linear,
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
    fn actions_from_obs_with_log_probs(
        &mut self,
        obs: Tensor<B, 2>,
        deterministic: bool,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let action = self.actions_from_obs(obs, deterministic);
        let log_prob = self.log_prob(action.clone());

        (action, log_prob)
    }
}

#[derive(Debug, Module)]
pub struct DiagGaussianDistribution<B: Backend> {
    means: Linear<B>,
    log_std: Linear<B>,
    dist: Normal<B, 2>,
    min_log_std: f32,
    max_log_std: f32,
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
            min_log_std: -5.0,
            max_log_std: 2.0,
        }
    }
}

impl<B: Backend> ActionDistribution<B> for DiagGaussianDistribution<B> {
    fn log_prob(&self, sample: Tensor<B, 2>) -> Tensor<B, 2> {
        // (B, N)
        let log_prob = self.dist.log_prob(sample);

        // (B, 1)
        log_prob.sum_dim(1)
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
        let loc = self.means.forward(obs.clone());

        // Soft clamp the log std
        let log_std: Tensor<B, 2> = self.log_std.forward(obs).tanh();
        let log_std = self.min_log_std
            + 0.5 * (self.max_log_std - self.min_log_std) * (log_std + 1.0);
        let scale = log_std.exp();

        // if loc.shape()[0] == 1 {
        //     println!("loc={loc}");
        //     println!("scale={scale}");
        // }

        self.dist = Normal::new(loc.clone(), scale);

        if deterministic {
            self.dist.mean()
        } else {
            self.dist.rsample()
        }
    }
}

impl<B: Backend> Policy<B> for DiagGaussianDistribution<B> {
    fn update(&mut self, from: &Self, tau: Option<f32>) {
        self.means = update_linear(&from.means, self.means.clone(), tau);
        self.log_std = update_linear(&from.log_std, self.log_std.clone(), tau);
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

    fn log_prob_correction(&self, ln_u: Tensor<B, 2>, a: Tensor<B, 2>) -> Tensor<B, 2> {
        // ln_u: (B, 1)
        // a: (B, N)

        // (B, 1)
        let correction = ((1.0 - a.powi_scalar(2.0) + self.epsilon) as Tensor<B, 2>)
            .log()
            .sum_dim(1);

        // (B, 1)
        ln_u - correction
    }
}

impl<B: Backend> ActionDistribution<B> for SquashedDiagGaussianDistribution<B> {
    fn log_prob(&self, a: Tensor<B, 2>) -> Tensor<B, 2> {
        // (B, N)
        let u = tanh_bijector_inverse(a.clone());

        // (B, 1)
        let ln_u = self.diag_gaus_dist.log_prob(u);

        // (B, 1)
        self.log_prob_correction(ln_u, a)
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
        let u = self.diag_gaus_dist.actions_from_obs(obs, deterministic);

        let a = u.clone().tanh();

        // if a.shape()[0] == 1 {
        //     println!("Gaussian Action: {u}. Tanh Action: {a}");
        // }

        a
    }

    fn actions_from_obs_with_log_probs(
        &mut self,
        obs: Tensor<B, 2>,
        deterministic: bool,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // we can calc the log probs with u directly, rather than
        // having to do the tanh bijector stuff to map the squashed actions back onto
        // (an approximation of) the sampled gaussian actions
        let (u, ln_u) = self
            .diag_gaus_dist
            .actions_from_obs_with_log_probs(obs, deterministic);

        let a = u.clone().tanh();

        (a.clone(), self.log_prob_correction(ln_u, a))
    }
}

impl<B: Backend> Policy<B> for SquashedDiagGaussianDistribution<B> {
    fn update(&mut self, from: &Self, tau: Option<f32>) {
        self.diag_gaus_dist.update(&from.diag_gaus_dist, tau)
    }
}

fn tanh_bijector_inverse<B: Backend>(sample: Tensor<B, 2>) -> Tensor<B, 2> {
    let eps = f32::EPSILON;
    let sample = sample.clamp(-1.0 + eps, 1.0 - eps);

    tanh_bijector_atanh(sample)
}

fn tanh_bijector_atanh<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    0.5 * (x.clone().log1p() - (-x).log1p())
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
