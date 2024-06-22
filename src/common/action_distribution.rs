use burn::{module::{Module, Param}, nn::{Linear, LinearConfig}, tensor::{backend::Backend, Shape, Tensor}};

pub trait ActionDistribution<B, A> : Module<B>
where
    B: Backend,
    A: Clone
{
    fn log_prob(&self, a: &A);
    fn entroy(&self);
    fn mode(&self);
    fn sample(&self);
    fn get_actions(&self, deterministic: bool){
        if deterministic {
            self.mode()
        } else {
            self.sample()
        }
    }
}

/// Continuous actions are usually considered to be independent,
/// so we can sum components of the ``log_prob`` or the entropy.
/// 
/// # Shapes
/// t: (batch, n_actions) or (batch)
/// return: (batch) for (batch, n_actions) input, or (1) for (batch) input
fn sum_independent_dims<B: Backend>(t: Tensor<B, 1>) -> Tensor<B, 1>{
    t.sum()
}

fn sum_independent_dims_batched<B: Backend>(t: Tensor<B, 2>) -> Tensor<B, 1>{
    t.sum_dim(1).squeeze(1)
}

#[derive(Module, Debug)]
pub struct DiagGaussianDistribution<B: Backend>{
    means: Linear<B>,
    log_std: Param<Tensor<B, 1>>,
}

impl<B: Backend> DiagGaussianDistribution<B>{
    pub fn new(latent_dim: usize, action_dim: usize, log_std_init: f32, device: &B::Device) -> Self {
        Self { 
            means: LinearConfig::new(latent_dim, action_dim).init(device),
            log_std: Param::from_tensor(Tensor::ones(Shape::new([action_dim]), device).mul_scalar(log_std_init))
        }
    }
}

impl<B: Backend> ActionDistribution<B, Vec<f32>> for DiagGaussianDistribution<B>{
    fn log_prob(&self, a: &Vec<f32>) {
        todo!()
    }

    fn entroy(&self) {
        todo!()
    }

    fn mode(&self) {
        todo!()
    }

    fn sample(&self) {
        todo!()
    }
}

#[derive(Module, Debug)]
pub struct StateDependentNoiseDistribution<B: Backend>{
    means: Linear<B>,
    log_std: Param<Tensor<B, 1>>,
}

impl<B: Backend> StateDependentNoiseDistribution<B>{
    fn sample_noise(&mut self, latent_sde: Tensor<B, 1>) -> Tensor<B, 1>{
        todo!()
    }
}