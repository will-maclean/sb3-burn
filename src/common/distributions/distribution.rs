use std::fmt::Debug;

use burn::tensor::{backend::Backend, Tensor};

pub trait BaseDistribution<B: Backend, const D: usize>: Clone + Debug {
    fn mean(&self) -> Tensor<B, D>;
    fn mode(&self) -> Tensor<B, D>;
    fn variance(&self) -> Tensor<B, D>;
    fn stdev(&self) -> Tensor<B, D>;
    fn sample(&mut self) -> Tensor<B, D>;
    fn rsample(&self) -> Tensor<B, D>;
    fn log_prob(&self, value: Tensor<B, D>) -> Tensor<B, D>;
    fn cdf(&self, value: Tensor<B, D>) -> Tensor<B, D>;
    fn icdf(&self, value: Tensor<B, D>) -> Tensor<B, D>;
    fn entropy(&self) -> Tensor<B, D>;
    fn perplexity(&self) -> Tensor<B, D> {
        self.entropy().exp()
    }

    // Some methods have been left out from the torch Distribution
    // class:
    // - set_default_validate_args
    // - expand
    // - batch_shape
    // - event_shape
    // - arg_constraints
    // - support
    // - enumerate_support
    // - _extended_shape
    // - _validate_sample
    // - _get_checked_instance
    //
    // These can be added in if/when required
}
