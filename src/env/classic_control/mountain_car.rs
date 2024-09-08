use crate::{
    common::{
        spaces::{BoxSpace, Discrete, Space},
        utils::generate_random_vector,
    },
    env::base::{Env, EnvObservation, ResetOptions, RewardRange},
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

impl Env<Vec<f32>, usize> for MountainCarEnv {
    fn step(&mut self, action: &usize) -> EnvObservation<Vec<f32>> {
        if self.needs_reset {
            panic!("Reset required");
        }
        let action = *action as f32;

        let mut p = self.state[0];
        let mut v = self.state[1];

        v += (action - 1.0) * self.force + (3.0 * p).cos() * (-self.gravity);
        v = v.clamp(-self.max_speed, self.max_speed);
        p += v;
        p = p.clamp(self.min_position, self.max_position);

        if (p <= self.min_position) & (v >= self.goal_velocity) {
            v = 0.0;
        }

        let terminated = (p >= self.goal_position) & (v >= self.goal_velocity);

        self.curr_steps += 1;
        let truncated = self.curr_steps > self.max_steps;

        self.needs_reset = terminated | truncated;
        let reward = -1.0;
        self.state = vec![p, v];

        EnvObservation {
            obs: self.state.clone(),
            reward,
            terminated,
            truncated,
            info: Default::default(),
        }
    }

    fn reset(&mut self, _seed: Option<u64>, _options: Option<ResetOptions>) -> Vec<f32> {
        self.needs_reset = false;

        self.state = generate_random_vector(vec![-0.6, 0.0], vec![-0.4, 0.0]);

        self.curr_steps = 0;

        self.state.clone()
    }

    fn action_space(&self) -> Box<dyn Space<usize>> {
        Box::new(Discrete::from(3))
    }

    fn observation_space(&self) -> Box<dyn Space<Vec<f32>>> {
        Box::new(BoxSpace::from((self.lows.clone(), self.highs.clone())))
    }

    fn render(&self) {}

    fn renderable(&self) -> bool {
        false
    }

    fn reward_range(&self) -> crate::env::base::RewardRange {
        RewardRange {
            low: -1.0,
            high: -1.0,
        }
    }

    fn close(&mut self) {}

    fn unwrapped(&self) -> &dyn Env<Vec<f32>, usize> {
        self
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
        env.reset(None, None);

        while !done {
            let result = env.step(&env.action_space().sample());
            done = result.truncated | result.terminated;
        }
    }
}
