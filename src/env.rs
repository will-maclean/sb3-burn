use core::panic;

use crate::spaces::{Action, ActionSpace, Obs, ObsSpace, SpaceSample};
use ndarray::{
    indices_of,
    prelude::{Array, Dim},
};
use rand::{thread_rng, Rng};

#[derive(Clone, Debug)]
pub struct EnvObservation {
    pub obs: Obs,
    pub reward: f32,
    pub done: bool,
}

pub trait Env {
    fn step(&mut self, action: &SpaceSample) -> EnvObservation;
    fn reset(&mut self) -> Obs;
    fn action_space(&self) -> ActionSpace;
    fn observation_space(&self) -> ObsSpace;
}

#[derive(Clone, Debug, Copy)]
struct Pos {
    x: usize,
    y: usize,
}

impl Pos {
    fn to_tuple(self) -> (usize, usize) {
        (self.x, self.y)
    }
}

impl From<(usize, usize)> for Pos {
    fn from(value: (usize, usize)) -> Self {
        Self {
            x: value.0,
            y: value.1,
        }
    }
}

pub struct GridWorldEnv {
    // stores the actual map. Uses:
    // 0 = empty
    // 1 = hole
    // 2 = target
    // 3 = player
    field: Array<f32, Dim<[usize; 2]>>,
    dim: usize,
    maxlen: usize,
    curr_len: usize,
    pos: Pos,
    needs_reset: bool,

    // Continuous
    // shape -> width * height = dim * dim (always square)
    observation_space: ObsSpace,

    // discrete
    // 0 -> left
    // 1 -> up
    // 2 -> right
    // 3 -> down
    action_space: ActionSpace,

    obstacle_prob: f32,
}

impl Default for GridWorldEnv {
    fn default() -> Self {
        Self {
            field: Array::<f32, _>::zeros((4, 4)),
            dim: 4,
            pos: Pos { x: 0, y: 0 },
            observation_space: ObsSpace::Continuous {
                lows: vec![0.0; 16],
                highs: vec![3.0; 16],
            },
            action_space: ActionSpace::Discrete { size: 4 },
            obstacle_prob: 0.1,
            maxlen: 20,
            curr_len: 0,
            needs_reset: true,
        }
    }
}

impl GridWorldEnv {
    pub fn new(dim: usize, maxlen: usize, obstacle_prob: f32) -> Self {
        Self {
            field: Array::<f32, _>::zeros((dim, dim)),
            dim,
            pos: Pos { x: 0, y: 0 },
            observation_space: ObsSpace::Continuous {
                lows: vec![0.0; dim * dim],
                highs: vec![3.0; dim * dim],
            },
            action_space: ActionSpace::Discrete { size: 4 },
            obstacle_prob,
            maxlen,
            curr_len: 0,
            needs_reset: true,
        }
    }
}

impl Env for GridWorldEnv {
    fn step(&mut self, action: &Action) -> EnvObservation {
        if self.needs_reset {
            panic!("Need to reset the environment before using it!");
        }

        // first, check if we've got a valid action
        let a: i32;
        match action {
            Action::Discrete { space: _, idx } => a = *idx,
            Action::Continuous { space: _, data: _ } => {
                panic!("Continuous actions are not supported!")
            }
        }

        let mut dead = false;
        let mut win = false;

        // now try to make the action
        let can_move = match a {
            0 => self.pos.x > 0,
            1 => self.pos.y > 0,
            2 => self.pos.x < self.dim - 1,
            3 => self.pos.y < self.dim - 1,
            _ => panic!("Unknown action"),
        };

        if can_move {
            let curr_pos = self.pos.to_tuple();
            let new_pos = match a {
                0 => (curr_pos.0 - 1, curr_pos.1),
                1 => (curr_pos.0, curr_pos.1 - 1),
                2 => (curr_pos.0 + 1, curr_pos.1),
                3 => (curr_pos.0, curr_pos.1 + 1),
                _ => panic!("Unknown action"),
            };
            // start by clearing current pos
            self.field[curr_pos] = 0.0;
            // check what's to the left
            if self.field[new_pos] == 1.0 {
                // it's a hole, so we lose
                dead = true;
            } else if self.field[new_pos] == 2.0 {
                // it's the target, so we win
                win = true;
            } else {
                // it's not a hole, so continue
                self.field[new_pos] = 3.0;
                self.pos = Pos::from(new_pos);
            }
        }

        let mut reward = 0.0;
        if dead {
            reward = -1.0;
        } else if win {
            reward = 1.0;
        }

        self.curr_len += 1;
        let done = win | dead | (self.curr_len >= self.maxlen);

        if done {
            self.needs_reset = true;
        }

        EnvObservation {
            obs: Obs::Continuous {
                space: self.observation_space.clone(),
                data: self
                    .field
                    .to_shape((self.dim * self.dim,))
                    .unwrap()
                    .to_vec(),
            },
            reward,
            done,
        }
    }

    fn reset(&mut self) -> Obs {
        self.field = Array::<f32, _>::zeros((self.dim, self.dim));
        self.curr_len = 0;

        // reset the obstacles
        for i in indices_of(&self.field) {
            if rand::random::<f32>() < self.obstacle_prob {
                self.field[i] = 1.0;
            }
        }

        // set the goal position
        let mut rng = thread_rng();
        let goal_pos = (rng.gen_range(0..self.dim), rng.gen_range(0..self.dim));
        self.field[goal_pos] = 2.0;

        // set the player position
        let mut player_pos = (rng.gen_range(0..self.dim), rng.gen_range(0..self.dim));

        while player_pos == goal_pos {
            player_pos = (rng.gen_range(0..self.dim), rng.gen_range(0..self.dim));
        }

        let player_pos = player_pos;

        self.pos = Pos {
            x: player_pos.0,
            y: player_pos.1,
        };
        self.field[player_pos] = 3.0;

        self.needs_reset = false;

        Obs::Continuous {
            space: self.observation_space.clone(),
            data: self
                .field
                .to_shape((self.dim * self.dim,))
                .unwrap()
                .to_vec(),
        }
    }

    fn action_space(&self) -> ActionSpace {
        self.action_space.clone()
    }

    fn observation_space(&self) -> ObsSpace {
        self.observation_space.clone()
    }
}

mod tests {
    use super::{Env, GridWorldEnv};

    #[test]
    fn test_gridworld_default() {
        let gridworld = GridWorldEnv::default();

        assert_eq!(gridworld.dim, 4);
        assert_eq!(gridworld.obstacle_prob, 0.1);
        assert_eq!(gridworld.maxlen, 20);
    }

    #[test]
    fn test_gridworld_non_default() {
        let gridworld = GridWorldEnv::new(5, 21, 0.2);

        assert_eq!(gridworld.dim, 5);
        assert_eq!(gridworld.obstacle_prob, 0.2);
        assert_eq!(gridworld.maxlen, 21);
    }

    #[test]
    fn test_gridworld_reset() {
        let mut gridworld = GridWorldEnv::default();

        let obs = gridworld.reset();

        // check there is only one goal and one player
        match obs {
            crate::spaces::SpaceSample::Discrete { space: _, idx: _ } => {
                panic!("GridWorldEnv should return a continuous space sample")
            }
            crate::spaces::SpaceSample::Continuous { space: _, data } => {
                let mut n_players = 0;
                let mut n_goals = 0;

                for item in data.into_iter() {
                    if item == 2.0 {
                        // goal
                        n_goals += 1;
                    } else if item == 3.0 {
                        // player
                        n_players += 1;
                    }
                }

                assert_eq!(n_goals, 1);
                assert_eq!(n_players, 1);
            }
        }
    }

    #[test]
    #[should_panic]
    fn test_gridworld_step_without_reset_errors() {
        let mut gridworld = GridWorldEnv::default();

        let action = gridworld.action_space.sample();
        gridworld.step(&action);
    }

    #[test]
    fn test_gridworld_steps() {
        let mut gridworld = GridWorldEnv::default();

        gridworld.reset();
        gridworld.step(&gridworld.action_space.sample());
    }
}
