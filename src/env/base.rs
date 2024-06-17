use std::collections::HashMap;

use crate::{logger::LogData, spaces::Space};

#[derive(Debug, Clone)]
pub enum InfoData<O> {
    String(String),
    Float(f32),
    Int(i32),
    Obs(O),
    InfoDict(Info<O>),
}

pub type ResetOptions = HashMap<String, LogData>;
pub type Info<O> = HashMap<String, InfoData<O>>;

#[derive(Clone, Debug)]
pub struct EnvObservation<O> {
    pub obs: O,
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
    pub info: Info<O>,
}

#[derive(Clone, Debug, Copy)]
pub struct RewardRange {
    pub low: f32,
    pub high: f32,
}

pub trait Env<O, A> {
    fn step(&mut self, action: &A) -> EnvObservation<O>;
    fn reset(&mut self, seed: Option<[u8; 32]>, options: Option<ResetOptions>) -> O;
    fn action_space(&self) -> Box<dyn Space<A>>;
    fn observation_space(&self) -> Box<dyn Space<O>>;
    fn reward_range(&self) -> RewardRange;
    fn render(&self);
    fn renderable(&self) -> bool;
    fn close(&mut self);
    fn unwrapped(&self) -> &dyn Env<O, A>;
}
