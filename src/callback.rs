use burn::{optim::SimpleOptimizer, tensor::backend::AutodiffBackend};

use crate::{algorithm::OfflineTrainer, env::base::EnvObservation};

// Callbacks in stable-baselines3 do not take in specific parameters - rather, they
// take in the local and global variable namespaces. I'm not sure how this would be
// replicated in rust and, even if it could be, I'm not sure if that's the nicest
// way to do it. Instead, we'll pass in a specific set of params. We can update the
// API as required until things settle down a bit

// The Callback trait, which handles flexible functionality to be called at various
// times during training. The Callback is a useful too for implementing unique
// functionality without needing to modify the core training loop.
pub trait Callback<O: SimpleOptimizer<B::InnerBackend>, B: AutodiffBackend> {
    fn on_training_start(&self, trainer: &OfflineTrainer<O, B>);
    fn on_step(
        &self,
        trainer: &OfflineTrainer<O, B>,
        step: usize,
        env_obs: EnvObservation,
        loss: Option<f32>,
    );
    fn on_training_end(&self, trainer: &OfflineTrainer<O, B>);
}

// A stub callback that does nothing.
pub struct EmptyCallback {}

impl<O: SimpleOptimizer<B::InnerBackend>, B: AutodiffBackend> Callback<O, B> for EmptyCallback {
    fn on_training_start(&self, _trainer: &OfflineTrainer<O, B>) {}

    fn on_step(
        &self,
        _trainer: &OfflineTrainer<O, B>,
        _step: usize,
        _env_obs: EnvObservation,
        _loss: Option<f32>,
    ) {
    }

    fn on_training_end(&self, _trainer: &OfflineTrainer<O, B>) {}
}
