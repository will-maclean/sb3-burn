extern crate sb3_burn;

use sb3_burn::env::{base::Env, classic_control::cartpole::CartpoleEnv};

fn main() {
    let mut env = CartpoleEnv::default();

    let mut done = false;
    let mut reward = 0.0;
    env.reset(None, None);

    while !done {
        let a = env.action_space().sample();
        let obs = env.step(&a);

        println!("action: {:?}", a);
        println!("obs: {:?}", obs);

        done = obs.truncated | obs.terminated;
        reward += obs.reward;
    }

    println!("Finished random action episode. Reward: {reward}");
}
