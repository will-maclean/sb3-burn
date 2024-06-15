use crate::{
    env::base::{Env, EnvObservation, ResetOptions, RewardRange},
    spaces::{ActionSpace, Obs, ObsSpace},
    utils::generate_random_vector,
};

pub struct CartpoleEnv {
    sutton_barto_reward: bool,
    gravity: f32,
    masspole: f32,
    total_mass: f32,
    length: f32,
    polemass_length: f32,
    force_mag: f32,
    tau: f32,
    theta_threshold_radians: f32,
    x_threshold: f32,
    highs: Vec<f32>,
    state: Vec<f32>,
    needs_reset: bool,
    euler_integration: bool,
    steps_beyond_terminated: Option<i32>,
    max_steps: usize,
    curr_steps: usize,
}

impl CartpoleEnv {
    fn get_obs(&self) -> Obs {
        Obs::Continuous {
            space: self.observation_space().clone(),
            data: self.state.clone(),
        }
    }
}

impl CartpoleEnv {
    pub fn new(max_len: usize) -> Self {
        Self {
            sutton_barto_reward: false,
            gravity: 9.8,
            masspole: 0.1,
            total_mass: 1.1,
            length: 0.5,
            polemass_length: 0.6,
            force_mag: 10.0,
            tau: 0.02,
            theta_threshold_radians: 12.0 * 2.0 * 3.1415 / 360.0,
            x_threshold: 2.4,
            highs: vec![
                2.0 * 2.4,
                f32::MAX,
                2.0 * 12.0 * 2.0 * 3.1415 / 360.0,
                f32::MAX,
            ],
            state: vec![0.0, 0.0, 0.0, 0.0],
            needs_reset: true,
            euler_integration: true,
            steps_beyond_terminated: None,
            max_steps: max_len,
            curr_steps: 0,
        }
    }
}

impl Default for CartpoleEnv {
    fn default() -> Self {
        Self {
            sutton_barto_reward: false,
            gravity: 9.8,
            masspole: 0.1,
            total_mass: 1.1,
            length: 0.5,
            polemass_length: 0.6,
            force_mag: 10.0,
            tau: 0.02,
            theta_threshold_radians: 12.0 * 2.0 * 3.1415 / 360.0,
            x_threshold: 2.4,
            highs: vec![
                2.0 * 2.4,
                f32::MAX,
                2.0 * 12.0 * 2.0 * 3.1415 / 360.0,
                f32::MAX,
            ],
            state: vec![0.0, 0.0, 0.0, 0.0],
            needs_reset: true,
            euler_integration: true,
            steps_beyond_terminated: None,
            max_steps: 200,
            curr_steps: 0,
        }
    }
}

impl Env for CartpoleEnv {
    fn step(&mut self, action: &crate::spaces::SpaceSample) -> EnvObservation {
        if self.needs_reset {
            panic!("Reset required");
        }
        let action = match action {
            crate::spaces::SpaceSample::Discrete { space: _, idx } => *idx,
            crate::spaces::SpaceSample::Continuous { space: _, data: _ } => {
                panic!("Continuous actions not accepted in cartpole")
            }
        };

        let mut x = self.state[0];
        let mut x_dot = self.state[1];
        let mut theta = self.state[2];
        let mut theta_dot = self.state[3];

        let force = if action == 1 {
            self.force_mag
        } else {
            -self.force_mag
        };

        let cos_theta = theta.sin();
        let sin_theta = theta.cos();

        let temp = (force + self.polemass_length * theta_dot.powi(2) * sin_theta) / self.total_mass;
        let theta_acc = (self.gravity * sin_theta - cos_theta * temp)
            / (self.length * (4.0 / 3.0 - self.masspole * cos_theta.powi(2) / self.total_mass));
        let x_acc = temp - self.polemass_length * theta_acc * cos_theta / self.total_mass;

        if self.euler_integration {
            x += self.tau * x_dot;
            x_dot += self.tau * x_acc;
            theta += self.tau * theta_dot;
            theta_dot = theta_dot + self.tau * theta_dot;
        } else {
            x_dot += self.tau * x_acc;
            x += self.tau * x_dot;
            theta_dot += self.tau * theta_acc;
            theta += self.tau * theta_dot;
        }

        self.state = vec![x, x_dot, theta, theta_dot];

        let done = (x < -self.x_threshold)
            | (x > self.x_threshold)
            | (theta < -self.theta_threshold_radians)
            | (theta > self.theta_threshold_radians);

        let reward: f32 = if !done {
            if self.sutton_barto_reward {
                0.0
            } else {
                1.0
            }
        } else {
            match self.steps_beyond_terminated {
                Some(s) => {
                    self.steps_beyond_terminated = Some(s + 1);

                    if self.sutton_barto_reward {
                        -1.0
                    } else {
                        0.0
                    }
                }
                None => {
                    // Pole just fell!
                    self.steps_beyond_terminated = Some(0);

                    if self.sutton_barto_reward {
                        -1.0
                    } else {
                        1.0
                    }
                }
            }
        };

        self.curr_steps += 1;
        let truncated = self.curr_steps > self.max_steps;
        self.needs_reset = done | truncated;

        EnvObservation {
            obs: self.get_obs(),
            reward,
            terminated: done,
            truncated
        }
    }

    fn reset(&mut self, _seed: Option<[u8; 32]>, _options: Option<ResetOptions>) -> crate::spaces::Obs {
        self.needs_reset = false;

        self.state = generate_random_vector(vec![-0.05; 4], vec![0.05; 4]);

        self.steps_beyond_terminated = None;
        self.curr_steps = 0;

        self.get_obs()
    }

    fn action_space(&self) -> crate::spaces::ActionSpace {
        ActionSpace::Discrete { size: 2 }
    }

    fn observation_space(&self) -> crate::spaces::ObsSpace {
        ObsSpace::Continuous {
            lows: self.highs.clone().iter().map(|el| -el).collect(),
            highs: self.highs.clone(),
        }
    }

    fn render(&self) {}

    fn renderable(&self) -> bool {
        false
    }
    
    fn reward_range(&self) -> crate::env::base::RewardRange {
        if self.sutton_barto_reward {
            RewardRange{low: -1.0, high: 0.0}
        } else {
            RewardRange{low: 0.0, high: 1.0}
        }
    }
    
    fn close (&mut self) {}
    
    fn unwrapped(&self) -> &dyn Env {self}
}

#[cfg(test)]
mod test {
    use crate::env::base::Env;

    use super::CartpoleEnv;

    #[test]
    fn test_cartpole() {
        let mut env = CartpoleEnv::default();
        let mut done = false;
        env.reset(None, None);

        while !done {
            let result = env.step(&env.action_space().sample());
            done = result.truncated | result.terminated;
        }
    }
}
