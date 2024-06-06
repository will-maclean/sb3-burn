use crate::spaces::{ActionSpace, Obs, ObsSpace, SpaceSample};

use super::base::{Env, EnvObservation};

pub struct ProbeEnvValueTest {}

// One action, zero observation, one timestep long, +1 reward every timestep: This
// isolates the value network. If my agent can't learn that the value of the only
// observation it ever sees it 1, there's a problem with the value loss calculation
// or the optimizer.
impl ProbeEnvValueTest {}

impl Env for ProbeEnvValueTest {
    fn step(&mut self, _action: &SpaceSample) -> EnvObservation {
        EnvObservation {
            obs: self.observation_space().sample(),
            reward: 1.0,
            done: true,
        }
    }

    fn reset(&mut self) -> Obs {
        self.action_space().sample()
    }

    fn action_space(&self) -> ActionSpace {
        ActionSpace::Discrete { size: 1 }
    }

    fn observation_space(&self) -> ObsSpace {
        ObsSpace::Continuous {
            lows: vec![0.0],
            highs: vec![1.0],
        }
    }

    fn render(&self) {}

    fn renderable(&self) -> bool {
        false
    }
}
