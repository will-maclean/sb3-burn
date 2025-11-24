use std::mem;

use crate::{
    common::vec_env::base_env::{VecEnv, VecEnvObservation},
    env::base::{Env, ResetOptions},
};

pub struct DummyVecEnv<O, A> {
    envs: Vec<Box<dyn Env<O, A>>>,
    cached_obs: Option<VecEnvObservation<O>>,
}

impl<O, A> DummyVecEnv<O, A> {
    pub fn new(envs: Vec<Box<dyn Env<O, A>>>) -> Self {
        Self {
            envs,
            cached_obs: None,
        }
    }
}

impl<O: Clone, A: Clone> VecEnv<O, A> for DummyVecEnv<O, A> {
    fn step_async(&mut self, action: Vec<A>) {
        if action.len() != self.envs.len() {
            panic!(
                "Wrong amount of actions! Got {}, expecting {}",
                action.len(),
                self.envs.len()
            );
        }

        let mut new_obs = Vec::with_capacity(self.envs.len());
        let mut truncated_obs = Vec::with_capacity(self.envs.len());

        for (i, a) in action.iter().enumerate() {
            let mut obs = self.envs[i].step(&a);

            if obs.truncated {
                // store the termination
                truncated_obs.push(Some(obs.obs.clone()));
            } else {
                truncated_obs.push(None);
            }

            if obs.truncated || obs.terminated {
                obs.obs = self.envs[i].reset(None, None);
            }

            new_obs.push(obs);
        }

        self.cached_obs = Some(VecEnvObservation::new(new_obs, truncated_obs));
    }

    fn step_wait(&mut self) -> VecEnvObservation<O> {
        let obs = mem::replace(&mut self.cached_obs, None);
        obs.unwrap()
    }

    fn reset(&mut self, seed: Option<u64>, options: Option<ResetOptions>) -> Vec<O> {
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

#[cfg(test)]
mod tests {
    use crate::{
        common::vec_env::{base_env::VecEnv, dummy_vec_env::DummyVecEnv},
        env::{base::Env, classic_control::cartpole::CartpoleEnv},
    };

    #[test]
    fn test_dummy_basic() {
        let envs: Vec<Box<dyn Env<Vec<f32>, usize>>> = vec![
            Box::new(CartpoleEnv::default()),
            Box::new(CartpoleEnv::default()),
            Box::new(CartpoleEnv::default()),
        ];

        let mut vec_env = DummyVecEnv::new(envs);
        let mut action_space = vec_env.action_space();

        let reset_obs = vec_env.reset(None, None);

        assert_eq!(reset_obs.len(), 3);

        for _ in 0..10 {
            let act = (0..3).map(|_| action_space.sample()).collect();
            let obs = vec_env.step(act);
            assert_eq!(obs.len(), 3);
        }

        vec_env.close();
    }
}
