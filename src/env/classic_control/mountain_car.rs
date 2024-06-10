use crate::{
    env::base::{Env, EnvObservation},
    spaces::{ActionSpace, Obs, ObsSpace},
    utils::generate_random_vector,
};

pub struct MountainCarEnv {
    min_position: f32,
    max_position: f32,
    max_speed: f32,
    goal_position: f32,
    goal_velocity: f32,
    force: f32,
    gravity: f32,
    state: Vec<f32>,
    lows: Vec<f32>,
    highs: Vec<f32>,
    needs_reset: bool,
    curr_steps: i32,
    max_steps: i32,
}

impl MountainCarEnv {
    fn get_obs(&self) -> Obs {
        Obs::Continuous {
            space: self.observation_space().clone(),
            data: self.state.clone(),
        }
    }
}

impl Default for MountainCarEnv {
    fn default() -> Self {
        Self {
            min_position: -1.2,
            max_position: 0.6,
            max_speed: 0.07,
            goal_position: 0.5,
            goal_velocity: 0.0,
            force: 0.001,
            gravity: 0.0025,
            state: vec![0.0, 0.0],
            lows: vec![-1.2, -0.07],
            highs: vec![0.6, 0.07],
            needs_reset: true,
            curr_steps: 0,
            max_steps: 200,
        }
    }
}

impl Env for MountainCarEnv {
    fn step(&mut self, action: &crate::spaces::SpaceSample) -> EnvObservation {
        if self.needs_reset {
            panic!("Reset required");
        }
        let action = match action {
            crate::spaces::SpaceSample::Discrete { space: _, idx } => *idx as f32,
            crate::spaces::SpaceSample::Continuous { space: _, data: _ } => {
                panic!("Continuous actions not accepted in cartpole")
            }
        };

        let mut p = self.state[0];
        let mut v = self.state[1];

        v += (action - 1.0) * self.force + (3.0 * p).cos() * (-self.gravity);
        v = v.clamp(-self.max_speed, self.max_speed);
        p += v;
        p = p.clamp(self.min_position, self.max_position);

        if (p <= self.min_position) & (v >= self.goal_velocity) {
            v = 0.0;
        }

        let mut done = (p >= self.goal_position) & (v >= self.goal_velocity);

        self.curr_steps += 1;
        done |= self.curr_steps > self.max_steps;

        self.needs_reset = done;
        let reward = -1.0;
        self.state = vec![p, v];

        EnvObservation {
            obs: self.get_obs(),
            reward,
            done,
        }
    }

    fn reset(&mut self) -> crate::spaces::Obs {
        self.needs_reset = false;

        self.state = generate_random_vector(vec![-0.6, 0.0], vec![-0.4, 0.0]);

        self.curr_steps = 0;

        self.get_obs()
    }

    fn action_space(&self) -> crate::spaces::ActionSpace {
        ActionSpace::Discrete { size: 3 }
    }

    fn observation_space(&self) -> crate::spaces::ObsSpace {
        ObsSpace::Continuous {
            lows: self.lows.clone(),
            highs: self.highs.clone(),
        }
    }

    fn render(&self) {}

    fn renderable(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod test {
    use crate::env::base::Env;

    use super::MountainCarEnv;

    #[test]
    fn test_mountaincar() {
        let mut env = MountainCarEnv::default();
        let mut done = false;
        env.reset();

        while !done {
            let result = env.step(&env.action_space().sample());
            done = result.done;
        }
    }
}
