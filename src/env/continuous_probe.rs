use rand::Rng;

use crate::{
    common::spaces::{BoxSpace, Space, SHARED_RNG},
    env::base::{Env, EnvObservation, ResetOptions, RewardRange},
};

// One step environment, reward is always 1, just check the
// value function converges
#[derive(Debug, Clone)]
pub struct ProbeEnvContinuousActions1 {
    obs_space: BoxSpace<Vec<f32>>,
    action_space: BoxSpace<Vec<f32>>,
}

impl Default for ProbeEnvContinuousActions1 {
    fn default() -> Self {
        Self {
            obs_space: BoxSpace::from((vec![-1.0], vec![1.0])),
            action_space: BoxSpace::from((vec![-1.0], vec![1.0])),
        }
    }
}

impl Env<Vec<f32>, Vec<f32>> for ProbeEnvContinuousActions1 {
    fn step(&mut self, _action: &Vec<f32>) -> EnvObservation<Vec<f32>> {
        EnvObservation {
            obs: vec![0.0],
            reward: 1.0,
            terminated: true,
            truncated: false,
            info: Default::default(),
        }
    }

    fn reset(&mut self, _seed: Option<u64>, _options: Option<ResetOptions>) -> Vec<f32> {
        vec![0.0]
    }

    fn action_space(&self) -> Box<dyn Space<Vec<f32>>> {
        Box::new(self.action_space.clone())
    }

    fn observation_space(&self) -> Box<dyn Space<Vec<f32>>> {
        Box::new(self.obs_space.clone())
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

// Simple one step env, r = a
// action dependent
#[derive(Debug, Clone)]
pub struct ProbeEnvContinuousActions2 {
    state: f32,
}

impl Default for ProbeEnvContinuousActions2 {
    fn default() -> Self {
        Self { state: 0.0 }
    }
}

impl Env<Vec<f32>, Vec<f32>> for ProbeEnvContinuousActions2 {
    fn step(&mut self, action: &Vec<f32>) -> EnvObservation<Vec<f32>> {
        assert!(action.len() == 1);

        let a: f32 = (action[0]).clamp(-1.0, 1.0);

        let reward = a;

        EnvObservation {
            obs: [0.0].to_vec(),
            reward,
            terminated: true,
            truncated: false,
            info: Default::default(),
        }
    }

    fn reset(&mut self, _seed: Option<u64>, _options: Option<ResetOptions>) -> Vec<f32> {
        let mut rng = SHARED_RNG.lock().unwrap();
        self.state = (2.0 * rng.random::<f32>()) - 1.0;

        [self.state].to_vec()
    }

    fn action_space(&self) -> Box<dyn Space<Vec<f32>>> {
        Box::new(BoxSpace::from(([-1.0].to_vec(), [1.0].to_vec())))
    }

    fn observation_space(&self) -> Box<dyn Space<Vec<f32>>> {
        Box::new(BoxSpace::from(([-1.0].to_vec(), [1.0].to_vec())))
    }

    fn reward_range(&self) -> RewardRange {
        RewardRange {
            low: 0.0,
            high: 2.0,
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

// Simple one step env, r = -(s-a)^2
// obs and action dependent
#[derive(Debug, Clone)]
pub struct ProbeEnvContinuousActions3 {
    state: f32,
}

impl Default for ProbeEnvContinuousActions3 {
    fn default() -> Self {
        Self { state: 0.0 }
    }
}

impl Env<Vec<f32>, Vec<f32>> for ProbeEnvContinuousActions3 {
    fn step(&mut self, action: &Vec<f32>) -> EnvObservation<Vec<f32>> {
        assert!(action.len() == 1);

        let a: f32 = (action[0]).clamp(-1.0, 1.0);

        let reward = -(a - self.state).powi(2);

        EnvObservation {
            obs: [0.0].to_vec(),
            reward,
            terminated: true,
            truncated: false,
            info: Default::default(),
        }
    }

    fn reset(&mut self, _seed: Option<u64>, _options: Option<ResetOptions>) -> Vec<f32> {
        let mut rng = SHARED_RNG.lock().unwrap();
        self.state = (2.0 * rng.random::<f32>()) - 1.0;

        [self.state].to_vec()
    }

    fn action_space(&self) -> Box<dyn Space<Vec<f32>>> {
        Box::new(BoxSpace::from(([-1.0].to_vec(), [1.0].to_vec())))
    }

    fn observation_space(&self) -> Box<dyn Space<Vec<f32>>> {
        Box::new(BoxSpace::from(([-1.0].to_vec(), [1.0].to_vec())))
    }

    fn reward_range(&self) -> RewardRange {
        RewardRange {
            low: 0.0,
            high: 2.0,
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
// Very simple 1D probe env
//
// Model has to update position to a target
// on a 1D line. Given its current position
// and target position (no velocity/momentum)
// Reward is calculated from difference between state and target
// state
pub struct ProbeEnvContinuousActions4 {
    goal: f32,
    curr: f32,
    max_bound: f32,
    reward_scale: f32,
    tolerance: f32,
}

impl Default for ProbeEnvContinuousActions4 {
    fn default() -> Self {
        Self {
            goal: 0.0,
            curr: 0.0,
            max_bound: 10.0,
            reward_scale: 20.0,
            tolerance: 0.1,
        }
    }
}

impl Env<Vec<f32>, Vec<f32>> for ProbeEnvContinuousActions4 {
    fn step(&mut self, action: &Vec<f32>) -> EnvObservation<Vec<f32>> {
        self.curr += action[0].clamp(-1.0, 1.0);

        if self.curr.abs() > self.max_bound {
            // done, failure
            let reward = -(self.goal - self.curr).powi(2) / self.reward_scale;
            EnvObservation {
                obs: vec![self.goal, self.curr],
                reward: reward,
                terminated: true,
                truncated: false,
                info: Default::default(),
            }
        } else if (self.curr - self.goal).abs() < self.tolerance {
            // done, success
            EnvObservation {
                obs: vec![self.goal, self.curr],
                reward: 1.0,
                terminated: true,
                truncated: false,
                info: Default::default(),
            }
        } else {
            let reward = -(self.goal - self.curr).powi(2) / self.reward_scale;
            EnvObservation {
                obs: vec![self.goal, self.curr],
                reward: reward,
                terminated: false,
                truncated: false,
                info: Default::default(),
            }
        }
    }

    fn reset(&mut self, _seed: Option<u64>, _options: Option<ResetOptions>) -> Vec<f32> {
        let mut rng = SHARED_RNG.lock().unwrap();
        self.goal = rng.random::<f32>() * 5.0;
        self.curr = 0.0;

        vec![self.goal, self.curr]
    }

    fn action_space(&self) -> Box<dyn Space<Vec<f32>>> {
        Box::new(BoxSpace::from((vec![-1.0], vec![1.0])))
    }

    fn observation_space(&self) -> Box<dyn Space<Vec<f32>>> {
        Box::new(BoxSpace::from((
            vec![-self.max_bound, -self.max_bound],
            vec![self.max_bound, self.max_bound],
        )))
    }

    fn reward_range(&self) -> RewardRange {
        RewardRange {
            low: -10.0,
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
    use crate::env::base::Env;

    use super::{ProbeEnvContinuousActions1, ProbeEnvContinuousActions2};

    #[test]
    fn test_probe_env1_cont_actions() {
        let mut env = ProbeEnvContinuousActions1::default();

        // one step system
        // s=0, r=1, terminated = true, truncated = false
        let state = env.reset(None, None);

        assert_eq!(
            state.len(),
            1,
            "probe env continusous 1 state len should be 1, got {}",
            state.len()
        );
        assert_approx_eq::assert_approx_eq!(state[0], 0.0);

        // action doesn't matter
        let a = vec![0.5];

        let step = env.step(&a);

        assert_eq!(step.terminated, true);
        assert_eq!(step.truncated, false);
        assert_approx_eq::assert_approx_eq!(step.reward, 1.0)
    }

    #[test]
    fn test_probe_env2_cont_actions_optimal_action() {
        let mut env = ProbeEnvContinuousActions2::default();

        // one step system
        // done = true, reward = (s - a)^2
        let state = env.reset(None, None);

        assert_eq!(state.len(), 1);

        let optimal_action = state; // optimal is a = s

        let step = env.step(&optimal_action);

        assert_eq!(step.terminated, true);
        assert_eq!(step.truncated, false);
        assert_approx_eq::assert_approx_eq!(step.reward, 0.0)
    }

    #[test]
    fn test_probe_env2_cont_actions_bad_action() {
        let mut env = ProbeEnvContinuousActions2::default();

        // should be two step system, where first step
        // returns state=s, reward=0, done = false,
        // second step returns done = true, (no state), reward = (s - a)^2
        let state = env.reset(None, None);

        assert_eq!(state.len(), 1);

        let bad_action = vec![(state[0] + 1.0).clamp(0.0, 1.0)]; // optimal is a = s
        let expected_reward = (state[0] - bad_action[0]).powi(2);

        let step = env.step(&bad_action);

        assert_eq!(step.terminated, true);
        assert_eq!(step.truncated, false);
        assert_approx_eq::assert_approx_eq!(step.reward, expected_reward);
    }
}
