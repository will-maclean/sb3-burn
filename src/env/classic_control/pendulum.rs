use core::f32;

use crate::{
    common::{
        spaces::{BoxSpace, Space},
        utils::angle_normalise,
    },
    env::{
        base::{Env, EnvObservation, ResetOptions, RewardRange},
        wrappers::TimeLimitWrapper,
    },
};

const DEFAULT_X: f32 = f32::consts::PI;
const DEFAULT_Y: f32 = 1.0;

pub struct PendulumEnv {
    // constant
    max_speed: f32,
    max_torque: f32,
    dt: f32,
    g: f32,
    m: f32,
    l: f32,
    obs_space: BoxSpace<Vec<f32>>,
    action_space: BoxSpace<Vec<f32>>,

    // stateful
    state: Vec<f32>,
    last_u: f32,
}

impl PendulumEnv {
    fn _get_obs(&self) -> Vec<f32> {
        let theta = self.state[0];
        let theta_dot = self.state[1];

        vec![theta.cos(), theta.sin(), theta_dot]
    }
}

impl Default for PendulumEnv {
    fn default() -> Self {
        let high = vec![1.0, 1.0, 8.0];
        let low = vec![-1.0, -1.0, -8.0];
        Self {
            max_speed: 8.0,
            max_torque: 2.0,
            dt: 0.05,
            g: 10.0,
            m: 1.0,
            l: 1.0,
            obs_space: BoxSpace::from((low, high)),
            action_space: BoxSpace::from((vec![-2.0], vec![2.0])),
            state: vec![0.0, 0.0],
            last_u: 0.0,
        }
    }
}

impl Env<Vec<f32>, Vec<f32>> for PendulumEnv {
    fn step(&mut self, action: &Vec<f32>) -> EnvObservation<Vec<f32>> {
        assert!(action.len() == 1);

        let th = self.state[0];
        let th_dot = self.state[1];

        let u = action[0];
        let u = u.clamp(-self.max_torque, self.max_torque);
        self.last_u = u;

        let costs = angle_normalise(th).powi(2) + 0.1 * th_dot.powi(2) + 0.001 * u.powi(2);

        let new_th_dot = th_dot
            + (3.0 * self.g / (2.0 * self.l) * th.sin() + 3.0 / (self.m * self.l.powi(2)) * u)
                * self.dt;
        let new_th_dot = new_th_dot.clamp(-self.max_speed, self.max_speed);
        let new_th = th + new_th_dot * self.dt;

        self.state = vec![new_th, new_th_dot];

        EnvObservation {
            obs: self._get_obs(),
            reward: -costs,
            terminated: false,
            truncated: false,
            info: Default::default(),
        }
    }

    fn reset(&mut self, seed: Option<u64>, options: Option<ResetOptions>) -> Vec<f32> {
        let (x, y) = if let Some(_options) = options {
            println!("Warning - passing options to PendulumEnv, but options are not implemented");
            // let x: f32 = options.get("x_init").unwrap().into();
            // let y: f32 = options.get("y_init").unwrap().into();

            // self.obs_space = BoxSpace::from((vec![x, y], vec![-x, -y]));
            (DEFAULT_X, DEFAULT_Y)
        } else {
            (DEFAULT_X, DEFAULT_Y)
        };

        if let Some(seed) = seed {
            self.action_space.seed(seed);
            self.obs_space.seed(seed);
        }

        self.last_u = 0.0;
        self.state = BoxSpace::from(([-x, -y].to_vec(), [x, y].to_vec())).sample();

        self._get_obs()
    }

    fn action_space(&self) -> Box<dyn Space<Vec<f32>>> {
        Box::new(self.action_space.clone())
    }

    fn observation_space(&self) -> Box<dyn Space<Vec<f32>>> {
        Box::new(self.obs_space.clone())
    }

    fn reward_range(&self) -> RewardRange {
        // Is this correct?
        RewardRange {
            low: f32::MIN,
            high: 0.0,
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

pub fn make_pendulum(max_steps: Option<usize>) -> Box<dyn Env<Vec<f32>, Vec<f32>>> {
    let env = make_pendulum_eval(max_steps);

    env
}

pub fn make_pendulum_eval(max_steps: Option<usize>) -> Box<dyn Env<Vec<f32>, Vec<f32>>> {
    let max_steps = match max_steps {
        Some(s) => s,
        None => 200, // 200 is default for Pendulum
    };

    let env = PendulumEnv::default();
    let env = TimeLimitWrapper::new(Box::new(env), max_steps);

    Box::new(env)
}

#[cfg(test)]
mod test {
    use super::make_pendulum;

    #[test]
    fn test_make_env() {
        let mut env = make_pendulum(None);

        let mut done = false;
        let mut steps = 0;

        while !done {
            let res = env.step(&env.action_space().sample());
            done = res.terminated | res.truncated;
            steps += 1;
        }

        assert_eq!(steps, 200)
    }

    #[test]
    fn test_make_env_custom_steps() {
        let custom_steps = 10;
        let mut env = make_pendulum(Some(custom_steps));

        let mut done = false;
        let mut steps = 0;

        while !done {
            let res = env.step(&env.action_space().sample());
            done = res.terminated | res.truncated;
            steps += 1;
        }

        assert_eq!(steps, custom_steps)
    }
}
