use crate::spaces::{ActionSpace, Obs, ObsSpace, SpaceSample};

use async_trait::async_trait;

#[derive(Debug)]
pub enum EnvError {
    InitError(String),
    FatalEnvError(String),
}

#[derive(Clone, Debug)]
pub struct EnvObservation {
    pub obs: Obs,
    pub reward: f32,
    pub done: bool,
}

#[async_trait]
pub trait Env {
    async fn step(&mut self, action: &SpaceSample) -> Result<EnvObservation, EnvError>;
    async fn reset(&mut self) -> Result<Obs, EnvError>;
    fn action_space(&self) -> ActionSpace;
    fn observation_space(&self) -> ObsSpace;
    fn render(&self);
    fn renderable(&self) -> bool;
}
