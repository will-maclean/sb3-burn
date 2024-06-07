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

// Two actions, zero observation, one timestep long, action-dependent
// +1/-1 reward: The first env to exercise the policy! If my agent can't
// learn to pick the better action, there's something wrong with either
// my advantage calculations, my policy loss or my policy update. That's
// three things, but it's easy to work out by hand the expected values
// for each one and check that the values produced by your actual code
// line up with them.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProbeEnvActionTest {}

impl Env for ProbeEnvActionTest {
    fn step(&mut self, action: &SpaceSample) -> EnvObservation {
        match action {
            SpaceSample::Continuous { space: _, data: _ } => {
                panic!("Only discrete actions are accepted here")
            }
            SpaceSample::Discrete { space: _, idx } => {
                let reward = (*idx == 1) as i32 as f32;

                EnvObservation {
                    obs: self.observation_space().sample(),
                    reward,
                    done: true,
                }
            }
        }
    }

    fn reset(&mut self) -> Obs {
        Obs::Discrete {
            space: self.observation_space().clone(),
            idx: 0,
        }
    }

    fn action_space(&self) -> ActionSpace {
        ActionSpace::Discrete { size: 2 }
    }

    fn observation_space(&self) -> ObsSpace {
        ObsSpace::Discrete { size: 1 }
    }

    fn render(&self) {}

    fn renderable(&self) -> bool {
        false
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
    obs: i32,
}

impl ProbeEnvStateActionTest {
    fn gen_obs(&mut self) -> i32 {
        if self.rng.gen_bool(0.5) {
            1
        } else {
            0
        }
    }
}

impl Env for ProbeEnvStateActionTest {
    fn step(&mut self, action: &SpaceSample) -> EnvObservation {
        match action {
            SpaceSample::Continuous { space: _, data: _ } => {
                panic!("Only discrete actions are accepted here")
            }
            SpaceSample::Discrete { space: _, idx } => {
                let reward = (*idx == self.obs) as i32 as f32;

                EnvObservation {
                    obs: self.observation_space().sample(),
                    reward,
                    done: true,
                }
            }
        }
    }

    fn reset(&mut self) -> Obs {
        self.obs = self.gen_obs();
        Obs::Discrete {
            space: self.observation_space().clone(),
            idx: self.obs,
        }
    }

    fn action_space(&self) -> ActionSpace {
        ActionSpace::Discrete { size: 2 }
    }

    fn observation_space(&self) -> ObsSpace {
        ObsSpace::Discrete { size: 2 }
    }

    fn render(&self) {}

    fn renderable(&self) -> bool {
        false
    }
}
