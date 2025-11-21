use std::mem;

use crate::{
    common::vec_env::base_env::VecEnv,
    env::base::{Env, EnvObservation, ResetOptions},
};

pub struct DummyVecEnv<O, A> {
    envs: Vec<Box<dyn Env<O, A>>>,
    cached_obs: Vec<EnvObservation<O>>,
}

impl<O: Clone, A: Clone> VecEnv<O, A> for DummyVecEnv<O, A> {
    async fn step_async(&mut self, action: Vec<A>) {
        if action.len() != self.envs.len() {
            panic!(
                "Wrong amount of actions! Got {}, expecting {}",
                action.len(),
                self.envs.len()
            );
        }

        let mut new_obs = Vec::with_capacity(self.envs.len());

        for (i, a) in action.iter().enumerate() {
            new_obs.push(self.envs[i].step(&a));
        }
    }

    async fn step_wait(&mut self) -> Vec<EnvObservation<O>> {
        mem::replace(&mut self.cached_obs, Vec::with_capacity(self.envs.len()))
    }

    async fn reset(&mut self, seed: Option<u64>, options: Option<ResetOptions>) -> Vec<O> {
        self.envs
            .iter_mut()
            .map(|e| e.reset(seed, options.clone()))
            .collect()
    }

    fn action_space(&self) -> Box<dyn crate::common::spaces::Space<A>> {
        self.envs[0].action_space()
    }

    fn observation_space(&self) -> Box<dyn crate::common::spaces::Space<O>> {
        self.envs[0].observation_space()
    }

    fn reward_range(&self) -> crate::env::base::RewardRange {
        self.envs[0].reward_range()
    }

    fn render(&self) {
        self.envs[0].render()
    }

    fn renderable(&self) -> bool {
        self.envs[0].renderable()
    }

    fn close(&mut self) {
        self.envs.iter_mut().for_each(|e| e.close());
    }
}
