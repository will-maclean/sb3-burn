use std::fmt::Debug;

use crate::spaces::Space;

use super::base::{Env, EnvObservation, InfoData, RewardRange};

pub struct TimeLimitWrapper<O: Clone + Debug, A: Clone + Debug> {
    env: Box<dyn Env<O, A>>,
    max_steps: usize,
    curr_steps: usize,
}

impl<O: Clone + Debug, A: Clone + Debug> TimeLimitWrapper<O, A> {
    pub fn new(env: Box<dyn Env<O, A>>, max_steps: usize) -> Self {
        Self {
            env,
            max_steps,
            curr_steps: 0,
        }
    }
}

impl<O: Clone + Debug, A: Clone + Debug> Env<O, A> for TimeLimitWrapper<O, A> {
    fn step(&mut self, action: &A) -> EnvObservation<O> {
        let mut step_result = self.env.step(action);

        self.curr_steps += 1;
        step_result.truncated |= self.curr_steps >= self.max_steps;

        step_result
    }

    fn reset(&mut self, seed: Option<[u8; 32]>, options: Option<super::base::ResetOptions>) -> O {
        self.curr_steps = 0;

        self.env.reset(seed, options)
    }

    fn action_space(&self) -> Box<dyn Space<A>> {
        self.env.action_space()
    }

    fn observation_space(&self) -> Box<dyn Space<O>> {
        self.env.observation_space()
    }

    fn reward_range(&self) -> super::base::RewardRange {
        self.env.reward_range()
    }

    fn render(&self) {
        self.env.render()
    }

    fn renderable(&self) -> bool {
        self.env.renderable()
    }

    fn close(&mut self) {
        self.env.close()
    }

    fn unwrapped(&self) -> &dyn Env<O, A> {
        &*self.env
    }
}

pub struct AutoResetWrapper<O: Clone + Debug, A: Clone + Debug> {
    env: Box<dyn Env<O, A>>,
}

impl<O: Clone + Debug, A: Clone + Debug> AutoResetWrapper<O, A> {
    pub fn new(env: Box<dyn Env<O, A>>) -> Self {
        Self { env }
    }
}

impl<O: Clone + Debug, A: Clone + Debug> Env<O, A> for AutoResetWrapper<O, A> {
    fn step(&mut self, action: &A) -> EnvObservation<O> {
        let mut step_result = self.env.step(action);

        if step_result.truncated | step_result.terminated {
            if step_result
                .info
                .contains_key(&"final_observation".to_string())
            {
                panic!("info dict cannot contain key \"final_observation\"");
            }

            if step_result.info.contains_key(&"final_info".to_string()) {
                panic!("info dict cannot contain key \"final_info\"");
            }

            step_result.info.insert(
                "final_observation".to_string(),
                InfoData::Obs(step_result.obs.clone()),
            );
            step_result.info.insert(
                "final_info".to_string(),
                InfoData::InfoDict(step_result.info.clone()),
            );

            let new_obs = self.reset(None, None);
            step_result.obs = new_obs;
        }

        step_result
    }

    fn reset(&mut self, seed: Option<[u8; 32]>, options: Option<super::base::ResetOptions>) -> O {
        self.env.reset(seed, options)
    }

    fn action_space(&self) -> Box<dyn Space<A>> {
        self.env.action_space()
    }

    fn observation_space(&self) -> Box<dyn Space<O>> {
        self.env.observation_space()
    }

    fn reward_range(&self) -> RewardRange {
        self.env.reward_range()
    }

    fn render(&self) {
        self.env.render()
    }

    fn renderable(&self) -> bool {
        self.env.renderable()
    }

    fn close(&mut self) {
        self.env.close()
    }

    fn unwrapped(&self) -> &dyn Env<O, A> {
        &*self.env
    }
}

#[cfg(test)]
mod test {
    use crate::env::{base::Env, classic_control::cartpole::CartpoleEnv};

    use super::TimeLimitWrapper;

    #[test]
    fn test_time_limit_wrapper() {
        let truncate_steps = 5;
        let unwrapped_env = CartpoleEnv::new(200);
        let mut wrapped_env = TimeLimitWrapper::new(Box::new(unwrapped_env), truncate_steps);

        let mut ep_len = 0;
        let mut done = false;

        wrapped_env.reset(None, None);
        while !done {
            let step_result = wrapped_env.step(&wrapped_env.action_space().sample());
            done = step_result.truncated | step_result.terminated;
            ep_len += 1;
        }

        assert_eq!(truncate_steps, ep_len);
    }
}
