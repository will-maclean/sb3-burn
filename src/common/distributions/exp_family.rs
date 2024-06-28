
use burn::tensor::{backend::Backend, Tensor};

use super::distribution::BaseDistribution;

pub struct NaturalParams<B: Backend, const D: usize> {
    pub params: Vec<Tensor<B, D>>
}

pub trait ExpFamily<B: Backend, const D: usize> : BaseDistribution<B, D> {
    fn log_normaliser(&self, natural_params: NaturalParams<B, D>) -> Tensor<B, D>;
    fn mean_carrier_measure(&self) -> f32;
    fn natural_params(&self) -> NaturalParams<B, D>;
}

/// Method to compute the entropy using Bregman divergence of the log normalizer
pub fn exp_family_entropy<B: Backend, const D: usize, EF: ExpFamily<B, D>>(ef: &EF) -> f32{
    let result = -ef.mean_carrier_measure();

    // FIXME: Not sure how the below would be implemented
    //
    // nparams = [p.detach().requires_grad_() for p in self._natural_params]
    // lg_normal = self._log_normalizer(*nparams)
    // gradients = torch.autograd.grad(lg_normal.sum(), nparams, create_graph=True)
    // result += lg_normal
    // for np, g in zip(nparams, gradients):
    //     result -= (np * g).reshape(self._batch_shape + (-1,)).sum(-1)
    result
}