use super::base::{Env, InfoData};

pub struct TimeLimitWrapper {
    env: Box<dyn Env>,
    max_steps: usize,
    curr_steps: usize,
}

impl TimeLimitWrapper {
    pub fn new(env: Box<dyn Env>, max_steps: usize) -> Self {
        Self {
            env,
            max_steps,
            curr_steps: 0,
        }
    }
}

impl Env for TimeLimitWrapper {
    fn step(&mut self, action: &crate::spaces::SpaceSample) -> super::base::EnvObservation {
        let mut step_result = self.env.step(action);

        self.curr_steps += 1;
        step_result.truncated |= self.curr_steps >= self.max_steps;

        step_result
    }

    fn reset(
        &mut self,
        seed: Option<[u8; 32]>,
        options: Option<super::base::ResetOptions>,
    ) -> crate::spaces::Obs {
        self.curr_steps = 0;

        self.env.reset(seed, options)
    }

    fn action_space(&self) -> crate::spaces::ActionSpace {
        self.env.action_space()
    }

    fn observation_space(&self) -> crate::spaces::ObsSpace {
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

    fn unwrapped(&self) -> &dyn Env {
        &*self.env
    }
}

pub struct AutoResetWrapper {
    env: Box<dyn Env>,
}

impl AutoResetWrapper {
    pub fn new(env: Box<dyn Env>) -> Self {
        Self { env }
    }
}

impl Env for AutoResetWrapper {
    fn step(&mut self, action: &crate::spaces::SpaceSample) -> super::base::EnvObservation {
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

    fn reset(
        &mut self,
        seed: Option<[u8; 32]>,
        options: Option<super::base::ResetOptions>,
    ) -> crate::spaces::Obs {
        self.env.reset(seed, options)
    }

    fn action_space(&self) -> crate::spaces::ActionSpace {
        self.env.action_space()
    }

    fn observation_space(&self) -> crate::spaces::ObsSpace {
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

    fn unwrapped(&self) -> &dyn Env {
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
