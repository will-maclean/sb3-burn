use crate::spaces::{Space, SpaceSample};

pub struct EnvObservation {
    pub obs: SpaceSample,
    pub reward: f32,
    pub done: bool
}

pub trait Env {
    fn step(&self, action: &SpaceSample) -> EnvObservation;
    fn reset(&self) -> SpaceSample;
    fn action_space(&self) -> Space;
    fn observation_space(&self) -> Space;
}
