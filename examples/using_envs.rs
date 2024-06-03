extern crate sb3_burn;

use sb3_burn::env::{GridWorldEnv, Env};

fn main() {
    // create a GridWorld Env
    let mut gridworld = GridWorldEnv::default();

    let mut done = false;
    let mut reward = 0.0;
    while !done {
        let obs = gridworld.step(&gridworld.action_space().sample());

        done = obs.done;
        reward = obs.reward;
    }

    println!("Finished random action episode. Reward: {reward}");
}