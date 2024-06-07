use rand::{rngs::ThreadRng, Rng};

use crate::spaces::{ActionSpace, Obs, ObsSpace, SpaceSample};

use super::base::{Env, EnvObservation};

// One action, zero observation, one timestep long, +1 reward every timestep: This
// isolates the value network. If my agent can't learn that the value of the only
// observation it ever sees it 1, there's a problem with the value loss calculation
// or the optimizer.
#[derive(Debug, Default, Clone, Copy)]
pub struct ProbeEnvValueTest {}

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

// One action, random +1/0 observation, one timestep long, obs-dependent +1/0
// reward every time: If my agent can learn the value in (1.) but not this
// one - meaning it can learn a constant reward but not a predictable one! - it
// must be that backpropagation through my network is broken.
#[derive(Debug, Default, Clone)]
pub struct ProbeEnvBackpropTest {
    last_obs: i32,
    rng: ThreadRng,
}

impl ProbeEnvBackpropTest {
    fn gen_obs(&mut self) -> i32 {
        if self.rng.gen_bool(0.5) {
            1
        } else {
            0
        }
    }
}

impl Env for ProbeEnvBackpropTest {
    fn step(&mut self, _action: &SpaceSample) -> EnvObservation {
        let reward = self.last_obs as f32;
        let done = true;
        self.last_obs = self.gen_obs();

        EnvObservation {
            obs: Obs::Discrete {
                space: self.observation_space().clone(),
                idx: self.last_obs,
            },
            reward,
            done,
        }
    }

    fn reset(&mut self) -> Obs {
        self.last_obs = self.gen_obs();

        Obs::Discrete {
            space: self.observation_space().clone(),
            idx: self.last_obs,
        }
    }

    fn action_space(&self) -> ActionSpace {
        ActionSpace::Discrete { size: 1 }
    }

    fn observation_space(&self) -> ObsSpace {
        ObsSpace::Discrete { size: 2 }
    }

    fn render(&self) {}

    fn renderable(&self) -> bool {
        false
    }
}

// One action, zero-then-one observation, two timesteps long, +1
// reward at the end: If my agent can learn the value in (2.)
// but not this one, it must be that my reward discounting is broken.
#[derive(Debug, Clone)]
pub struct ProbeEnvDiscountingTest {
    done_next: bool,
}

impl Default for ProbeEnvDiscountingTest {
    fn default() -> Self {
        Self { done_next: false }
    }
}

impl Env for ProbeEnvDiscountingTest {
    fn step(&mut self, _action: &SpaceSample) -> EnvObservation {
        if self.done_next {
            EnvObservation {
                obs: Obs::Discrete {
                    space: self.observation_space().clone(),
                    idx: 1,
                },
                reward: 1.0,
                done: true,
            }
        } else {
            self.done_next = true;

            EnvObservation {
                obs: Obs::Discrete {
                    space: self.observation_space().clone(),
                    idx: 1,
                },
                reward: 0.0,
                done: false,
            }
        }
    }

    fn reset(&mut self) -> Obs {
        self.done_next = false;
        Obs::Discrete {
            space: self.observation_space().clone(),
            idx: 0,
        }
    }

    fn action_space(&self) -> ActionSpace {
        ActionSpace::Discrete { size: 1 }
    }

    fn observation_space(&self) -> ObsSpace {
        ObsSpace::Discrete { size: 2 }
    }

    fn render(&self) {}

    fn renderable(&self) -> bool {
        false
    }
}
