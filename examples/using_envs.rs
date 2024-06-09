extern crate sb3_burn;

use sb3_burn::env::{base::Env, cartpole::CartpoleEnv};

fn main() {
    // create a GridWorld Env
    let mut env = CartpoleEnv::default();

    let mut done = false;
    let mut reward = 0.0;
    env.reset();

    while !done {
        let a = env.action_space().sample();
        let obs = env.step(&a);

        println!("action: {:?}", a);
        println!("obs: {:?}", obs);

        done = obs.done;
        reward += obs.reward;
    }

    println!("Finished random action episode. Reward: {reward}");
}
