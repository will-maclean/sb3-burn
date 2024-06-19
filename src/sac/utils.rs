use burn::{optim::{adaptor::OptimizerAdaptor, Adam}, prelude::*, tensor::backend::AutodiffBackend};

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub enum EntCoefSetup {
    Static(f32),
    Trainable(f32)
}

pub enum EntCoef<B: AutodiffBackend> {
    Static(f32),
    Trainable(Tensor<B, 1>, OptimizerAdaptor<Adam<B::InnerBackend>, Tensor<B, 1>, B>)
}

impl<B: AutodiffBackend> EntCoef<B> {
    pub fn to_float(&self) -> f32 {
        match self {
            EntCoef::Static(v) => *v,
            EntCoef::Trainable(t, _) => t.clone().exp().into_scalar().elem(),
        }
    }
} 

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub enum ActionNoise{
    None,
}

#[derive(Config)]
pub struct SACConfig {
    #[config(default = 0.05)]
    pub tau: f32,
    #[config(default = 0.99)]
    pub gamma: f32,
    pub action_noise: ActionNoise,
    #[config(default = 1)]
    pub target_update_interval: usize,
    pub ent_coef: EntCoefSetup,
    pub target_entropy: Option<f32>,
    #[config(default = false)]
    pub use_sde: bool,
    #[config(default = 1)]
    pub sde_sample_freq: usize,
    #[config(default = false)]
    pub use_sde_at_warmup: bool,
}

pub trait ActionDistribution{

}

pub struct ActionLogProb<B: Backend, A> {
    pub pi: Vec<A>,
    pub log_pi: Tensor<B, 2>,
}