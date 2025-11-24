use crate::{
    common::spaces::Space,
    env::base::{EnvObservation, Info, ResetOptions, RewardRange},
};

#[derive(Clone, Debug)]
pub struct VecEnvObservation<O> {
    pub obs: Vec<O>,
    pub reward: Vec<f32>,
    pub terminated: Vec<bool>,
    pub truncated: Vec<bool>,
    pub info: Vec<Info<O>>,
    pub truncated_obs: Vec<Option<O>>,
}

impl<O> VecEnvObservation<O> {
    pub fn new(mut obs_vec: Vec<EnvObservation<O>>, truncated_obs: Vec<Option<O>>) -> Self {
        let mut obs = Vec::with_capacity(obs_vec.len());
        let mut reward = Vec::with_capacity(obs_vec.len());
        let mut terminated = Vec::with_capacity(obs_vec.len());
        let mut truncated = Vec::with_capacity(obs_vec.len());
        let mut info = Vec::with_capacity(obs_vec.len());

        for _ in 0..obs_vec.len() {
            let o = obs_vec.pop().unwrap();

            obs.push(o.obs);
            reward.push(o.reward);
            terminated.push(o.terminated);
            truncated.push(o.truncated);
            info.push(o.info);
        }

        Self {
            obs,
            reward,
            terminated,
            truncated,
            info,
            truncated_obs,
        }
    }

    pub fn len(&self) -> usize {
        self.obs.len()
    }
}

pub trait VecEnv<O, A> {
    fn step(&mut self, action: Vec<A>) -> VecEnvObservation<O> {
        self.step_async(action);
        self.step_wait()
    }
    fn step_async(&mut self, action: Vec<A>);
    fn step_wait(&mut self) -> VecEnvObservation<O>;
    fn reset(&mut self, seed: Option<u64>, options: Option<ResetOptions>) -> Vec<O>;
    fn action_space(&self) -> Box<dyn Space<A>>;
    fn observation_space(&self) -> Box<dyn Space<O>>;
    fn reward_range(&self) -> RewardRange;
    fn render(&self);
    fn renderable(&self) -> bool;
    fn close(&mut self);
}
