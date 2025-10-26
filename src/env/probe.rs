use rand::{rngs::ThreadRng, Rng};

use crate::common::spaces::{BoxSpace, Discrete, Space};

use super::base::{Env, EnvObservation, ResetOptions, RewardRange};

// One action, zero observation, one timestep long, +1 reward every timestep: This
// isolates the value network. If my agent can't learn that the value of the only
// observation it ever sees it 1, there's a problem with the value loss calculation
// or the optimizer.
#[derive(Debug, Default, Clone, Copy)]
pub struct ProbeEnvValueTest {}

impl Env<Vec<f32>, usize> for ProbeEnvValueTest {
    fn step(&mut self, _action: &usize) -> EnvObservation<Vec<f32>> {
        EnvObservation {
            obs: self.observation_space().sample(),
            reward: 1.0,
            terminated: true,
            truncated: false,
            info: Default::default(),
        }
    }

    fn reset(&mut self, _seed: Option<u64>, _options: Option<ResetOptions>) -> Vec<f32> {
        self.observation_space().sample()
    }

    fn render(&self) {}

    fn renderable(&self) -> bool {
        false
    }

    fn reward_range(&self) -> RewardRange {
        RewardRange {
            low: 1.0,
            high: 1.0,
        }
    }

    fn close(&mut self) {}

    fn unwrapped(&self) -> &dyn Env<Vec<f32>, usize> {
        self
    }

    fn action_space(&self) -> Box<dyn Space<usize>> {
        Box::new(Discrete::from(1))
    }

    fn observation_space(&self) -> Box<dyn Space<Vec<f32>>> {
        Box::new(BoxSpace::from((vec![0.0], vec![1.0])))
    }
}

// One action, random +1/0 observation, one timestep long, obs-dependent +1/0
// reward every time: If my agent can learn the value in (1.) but not this
// one - meaning it can learn a constant reward but not a predictable one! - it
// must be that backpropagation through my network is broken.
#[derive(Debug, Default, Clone)]
pub struct ProbeEnvBackpropTest {
    last_obs: usize,
    rng: ThreadRng,
    needs_reset: bool,
}

impl ProbeEnvBackpropTest {
    fn gen_obs(&mut self) -> usize {
        if self.rng.random_bool(0.5) {
            1
        } else {
            0
        }
    }
}

impl Env<usize, usize> for ProbeEnvBackpropTest {
    fn step(&mut self, _action: &usize) -> EnvObservation<usize> {
        if self.needs_reset {
            panic!("Env needs a reset");
        }

        let reward = self.last_obs as f32;
        self.needs_reset = true;

        EnvObservation {
            obs: 1,
            reward,
            terminated: true,
            truncated: false,
            info: Default::default(),
        }
    }

    fn reset(&mut self, _seed: Option<u64>, _options: Option<ResetOptions>) -> usize {
        self.last_obs = self.gen_obs();
        self.needs_reset = false;
        self.last_obs
    }

    fn action_space(&self) -> Box<dyn Space<usize>> {
        Box::new(Discrete::from(1))
    }

    fn observation_space(&self) -> Box<dyn Space<usize>> {
        Box::new(Discrete::from(2))
    }

    fn render(&self) {}

    fn renderable(&self) -> bool {
        false
    }

    fn reward_range(&self) -> super::base::RewardRange {
        RewardRange {
            low: 0.0,
            high: 1.0,
        }
    }

    fn close(&mut self) {}

    fn unwrapped(&self) -> &dyn Env<usize, usize> {
        self
    }
}

// One action, zero-then-one observation, two timesteps long, +1
// reward at the end: If my agent can learn the value in (2.)
// but not this one, it must be that my reward discounting is broken.
#[derive(Debug, Clone, Default)]
pub struct ProbeEnvDiscountingTest {
    done_next: bool,
}

impl Env<usize, usize> for ProbeEnvDiscountingTest {
    fn step(&mut self, _action: &usize) -> EnvObservation<usize> {
        if self.done_next {
            EnvObservation {
                obs: 1,
                reward: 1.0,
                terminated: true,
                truncated: false,
                info: Default::default(),
            }
        } else {
            self.done_next = true;

            EnvObservation {
                obs: 1,
                reward: 0.0,
                terminated: false,
                truncated: false,
                info: Default::default(),
            }
        }
    }

    fn reset(&mut self, _seed: Option<u64>, _options: Option<ResetOptions>) -> usize {
        self.done_next = false;

        0
    }

    fn action_space(&self) -> Box<dyn Space<usize>> {
        Box::new(Discrete::from(1))
    }

    fn observation_space(&self) -> Box<dyn Space<usize>> {
        Box::new(Discrete::from(2))
    }

    fn render(&self) {}

    fn renderable(&self) -> bool {
        false
    }

    fn reward_range(&self) -> super::base::RewardRange {
        RewardRange {
            low: 0.0,
            high: 1.0,
        }
    }

    fn close(&mut self) {}

    fn unwrapped(&self) -> &dyn Env<usize, usize> {
        self
    }
}

// Two actions, zero observation, one timestep long, action-dependent
// +1/-1 reward: The first env to exercise the policy! If my agent can't
// learn to pick the better action, there's something wrong with either
// my advantage calculations, my policy loss or my policy update. That's
// three things, but it's easy to work out by hand the expected values
// for each one and check that the values produced by your actual code
// line up with them.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProbeEnvActionTest {}

impl Env<usize, usize> for ProbeEnvActionTest {
    fn step(&mut self, action: &usize) -> EnvObservation<usize> {
        let reward = (*action == 1) as i32 as f32;

        EnvObservation {
            obs: self.observation_space().sample(),
            reward,
            terminated: true,
            truncated: false,
            info: Default::default(),
        }
    }

    fn reset(&mut self, _seed: Option<u64>, _options: Option<ResetOptions>) -> usize {
        0
    }

    fn action_space(&self) -> Box<dyn Space<usize>> {
        Box::new(Discrete::from(2))
    }

    fn observation_space(&self) -> Box<dyn Space<usize>> {
        Box::new(Discrete::from(1))
    }

    fn render(&self) {}

    fn renderable(&self) -> bool {
        false
    }

    fn reward_range(&self) -> super::base::RewardRange {
        RewardRange {
            low: 0.0,
            high: 1.0,
        }
    }

    fn close(&mut self) {}

    fn unwrapped(&self) -> &dyn Env<usize, usize> {
        self
    }
}

// Two actions, random +1/-1 observation, one timestep long, action-and-obs
// dependent +1/-1 reward: Now we've got a dependence on both obs and action.
// The policy and value networks interact here, so there's a couple of
// things to verify: that the policy network learns to pick the right action
// in each of the two states, and that the value network learns that the value
//  of each state is +1. If everything's worked up until now, then if - for
// example - the value network fails to learn here, it likely means your batching
//  process is feeding the value network stale experience.
#[derive(Debug, Clone, Default)]
pub struct ProbeEnvStateActionTest {
    rng: ThreadRng,
    obs: usize,
}

impl ProbeEnvStateActionTest {
    fn gen_obs(&mut self) -> usize {
        if self.rng.random_bool(0.5) {
            1
        } else {
            0
        }
    }
}

impl Env<usize, usize> for ProbeEnvStateActionTest {
    fn step(&mut self, action: &usize) -> EnvObservation<usize> {
        let reward = (*action == self.obs) as i32 as f32;

        EnvObservation {
            obs: self.observation_space().sample(),
            reward,
            terminated: true,
            truncated: false,
            info: Default::default(),
        }
    }

    fn reset(&mut self, _seed: Option<u64>, _options: Option<ResetOptions>) -> usize {
        self.obs = self.gen_obs();
        self.obs
    }

    fn action_space(&self) -> Box<dyn Space<usize>> {
        Box::new(Discrete::from(2))
    }

    fn observation_space(&self) -> Box<dyn Space<usize>> {
        Box::new(Discrete::from(2))
    }

    fn render(&self) {}

    fn renderable(&self) -> bool {
        false
    }

    fn reward_range(&self) -> super::base::RewardRange {
        RewardRange {
            low: 0.0,
            high: 1.0,
        }
    }

    fn close(&mut self) {}

    fn unwrapped(&self) -> &dyn Env<usize, usize> {
        self
    }
}

#[derive(Debug, Clone)]
pub struct ProbeEnvContinuousActions {
    state: f32,
    rng: ThreadRng,
}

impl Default for ProbeEnvContinuousActions {
    fn default() -> Self {
        Self {
            state: 0.0,
            rng: Default::default(),
        }
    }
}

impl Env<Vec<f32>, Vec<f32>> for ProbeEnvContinuousActions {
    fn step(&mut self, action: &Vec<f32>) -> EnvObservation<Vec<f32>> {
        assert!(action.len() == 1);

        let a: f32 = (action[0]).clamp(0.0, 1.0);

        let reward = 1.0 - (a - self.state).abs();

        EnvObservation {
            obs: [0.0].to_vec(),
            reward,
            terminated: true,
            truncated: false,
            info: Default::default(),
        }
    }

    fn reset(&mut self, _seed: Option<u64>, _options: Option<ResetOptions>) -> Vec<f32> {
        self.state = self.rng.random::<f32>();

        [self.state].to_vec()
    }

    fn action_space(&self) -> Box<dyn Space<Vec<f32>>> {
        Box::new(BoxSpace::from(([0.0].to_vec(), [1.0].to_vec())))
    }

    fn observation_space(&self) -> Box<dyn Space<Vec<f32>>> {
        Box::new(BoxSpace::from(([0.0].to_vec(), [1.0].to_vec())))
    }

    fn reward_range(&self) -> RewardRange {
        RewardRange {
            low: 0.0,
            high: 1.0,
        }
    }

    fn render(&self) {}

    fn renderable(&self) -> bool {
        false
    }

    fn close(&mut self) {}

    fn unwrapped(&self) -> &dyn Env<Vec<f32>, Vec<f32>> {
        self
    }
}

#[cfg(test)]
mod test {
    use crate::env::{base::Env, probe::ProbeEnvContinuousActions};

    use super::{
        ProbeEnvActionTest, ProbeEnvBackpropTest, ProbeEnvDiscountingTest, ProbeEnvStateActionTest,
        ProbeEnvValueTest,
    };

    #[test]
    fn test_probe_env_value_test() {
        let mut env = ProbeEnvValueTest::default();

        let mut done = false;
        while !done {
            let res = env.step(&env.action_space().sample());
            done = res.truncated | res.terminated;
        }
    }

    #[test]
    fn test_probe_env_backprop_test() {
        let mut env = ProbeEnvBackpropTest::default();

        let mut done = false;
        while !done {
            let res = env.step(&env.action_space().sample());
            done = res.truncated | res.terminated;
        }
    }

    #[test]
    fn test_probe_env_action_test() {
        let mut env = ProbeEnvActionTest::default();

        let mut done = false;
        while !done {
            let res = env.step(&env.action_space().sample());
            done = res.truncated | res.terminated;
        }
    }

    #[test]
    fn test_probe_env_state_action_test() {
        let mut env = ProbeEnvStateActionTest::default();

        let mut done = false;
        while !done {
            let res = env.step(&env.action_space().sample());
            done = res.truncated | res.terminated;
        }
    }

    #[test]
    fn test_probe_env_discounting_test() {
        let mut env = ProbeEnvDiscountingTest::default();

        let mut done = false;
        while !done {
            let res = env.step(&env.action_space().sample());
            done = res.truncated | res.terminated;
        }
    }

    #[test]
    fn test_probe_env_cont_actions_optimal_action() {
        let mut env = ProbeEnvContinuousActions::default();

        // should be two step system, where first step
        // returns state=s, reward=0, done = false,
        // second step returns done = true, (no state), reward = 1 - |s - a|
        let state = env.reset(None, None);

        assert_eq!(state.len(), 1);

        let optimal_action = state; // optimal is a = s

        let step = env.step(&optimal_action);

        assert_eq!(step.terminated, true);
        assert_eq!(step.truncated, false);
        assert_approx_eq::assert_approx_eq!(step.reward, 1.0)
    }

    #[test]
    fn test_probe_env_cont_actions_bad_action() {
        let mut env = ProbeEnvContinuousActions::default();

        // should be two step system, where first step
        // returns state=s, reward=0, done = false,
        // second step returns done = true, (no state), reward = 1 - |s - a|
        let state = env.reset(None, None);

        assert_eq!(state.len(), 1);

        let bad_action = vec![(state[0] + 1.0).clamp(0.0, 1.0)]; // optimal is a = s
        let expected_reward = 1.0 - (state[0] - bad_action[0]).abs();

        let step = env.step(&bad_action);

        assert_eq!(step.terminated, true);
        assert_eq!(step.truncated, false);
        assert_approx_eq::assert_approx_eq!(step.reward, expected_reward);
    }
}
