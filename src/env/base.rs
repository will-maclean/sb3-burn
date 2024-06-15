use std::collections::HashMap;

use crate::{
    logger::LogData,
    spaces::{ActionSpace, Obs, ObsSpace, SpaceSample},
};

#[derive(Debug, Clone)]
pub enum InfoData {
    String(String),
    Float(f32),
    Int(i32),
    Obs(SpaceSample),
    InfoDict(Info),
}

pub type ResetOptions = HashMap<String, LogData>;
pub type Info = HashMap<String, InfoData>;

#[derive(Clone, Debug)]
pub struct EnvObservation {
    pub obs: Obs,
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
    pub info: Info,
}

#[derive(Clone, Debug, Copy)]
pub struct RewardRange {
    pub low: f32,
    pub high: f32,
}

pub trait Env {
    fn step(&mut self, action: &SpaceSample) -> EnvObservation;
    fn reset(&mut self, seed: Option<[u8; 32]>, options: Option<ResetOptions>) -> Obs;
    fn action_space(&self) -> ActionSpace;
    fn observation_space(&self) -> ObsSpace;
    fn reward_range(&self) -> RewardRange;
    fn render(&self);
    fn renderable(&self) -> bool;
    fn close(&mut self);
    fn unwrapped(&self) -> &dyn Env;
}
