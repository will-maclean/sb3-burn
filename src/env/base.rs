use crate::spaces::{ActionSpace, Obs, ObsSpace, SpaceSample};

#[derive(Clone, Debug)]
pub struct EnvObservation {
    pub obs: Obs,
    pub reward: f32,
    pub done: bool,
}

pub trait Env {
    fn step(&mut self, action: &SpaceSample) -> EnvObservation;
    fn reset(&mut self) -> Obs;
    fn action_space(&self) -> ActionSpace;
    fn observation_space(&self) -> ObsSpace;
    fn render(&self);
    fn renderable(&self) -> bool;
}
