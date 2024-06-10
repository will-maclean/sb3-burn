extern crate sb3_burn;

use sb3_burn::env::{base::Env, gymsock::GymSockEnv};

async fn main() -> Result<(), Box<dyn Error>> {
    // Define the parameters for the Gym environment
    let params = RunningParams {
        env_name: "CartPole-v1".to_string(),
        env_type: "simple".to_string(),
        render: false,
    };

    // Create a GymSockEnv instance
    let mut gym_env = GymSockEnv::new("127.0.0.1:65432", params).await?;

    let mut done = false;
    let mut reward = 0.0;
    gym_env.reset().await?;

    while !done {
        // Sample an action from the action space
        let action = gym_env.action_space().sample();

        // Step through the environment
        let obs = gym_env.step(&action).await?;

        done = obs.done;
        reward = obs.reward;
    }

    println!("Finished random action episode. Reward: {reward}");

    Ok(())
