use burn::{optim::SimpleOptimizer, tensor::backend::AutodiffBackend};
use core::fmt::Debug;

use crate::env::base::EnvObservation;

use super::{agent::Agent, algorithm::OfflineTrainer};

/// Base trait for any callback.
///
/// Callbacks in stable-baselines3 do not take in specific parameters - rather, they
/// take in the local and global variable namespaces. I'm not sure how this would be
/// replicated in rust and, even if it could be, I'm not sure if that's the nicest
/// way to do it. Instead, we'll pass in a specific set of params. We can update the
/// API as required until things settle down a bit
/// The Callback trait, which handles flexible functionality to be called at various
/// times during training. The Callback is a useful too for implementing unique
/// functionality without needing to modify the core training loop.
pub trait Callback<
    A: Agent<B, OS, AS>,
    O: SimpleOptimizer<B::InnerBackend>,
    B: AutodiffBackend,
    OS: Clone + Debug,
    AS: Clone + Debug,
>
{
    fn on_training_start(&self, trainer: &OfflineTrainer<A, O, B, OS, AS>);
    fn on_step(
        &self,
        trainer: &OfflineTrainer<A, O, B, OS, AS>,
        step: usize,
        env_obs: EnvObservation<OS>,
        loss: Option<f32>,
    );
    fn on_training_end(&self, trainer: &OfflineTrainer<A, O, B, OS, AS>);
}

// A stub callback that does nothing.
pub struct EmptyCallback {}

impl<
        A: Agent<B, OS, AS>,
        O: SimpleOptimizer<B::InnerBackend>,
        B: AutodiffBackend,
        OS: Clone + Debug,
        AS: Clone + Debug,
    > Callback<A, O, B, OS, AS> for EmptyCallback
{
    fn on_training_start(&self, _trainer: &OfflineTrainer<A, O, B, OS, AS>) {}

    fn on_step(
        &self,
        _trainer: &OfflineTrainer<A, O, B, OS, AS>,
        _step: usize,
        _env_obs: EnvObservation<OS>,
        _loss: Option<f32>,
    ) {
    }

    fn on_training_end(&self, _trainer: &OfflineTrainer<A, O, B, OS, AS>) {}
}
