mod env;
mod spaces;
use env::{base::{Env, EnvError}, gymsock::GymSockEnv, gymsock::RunningParams};

#[tokio::main]
async fn main() -> Result<(), EnvError> {
    // Define the parameters for the Gym environment
    let params = RunningParams {
        env_name: "CartPole-v1".to_string(),
        env_type: "simple".to_string(),
        render: false,
    };

    // Create a GymSockEnv instance
    let mut gym_env = GymSockEnv::new("127.0.0.1:65432", params).await?;

    let mut t = 0;
    let mut obs;
    let mut done = false;
    let mut reward = 0.0;
    gym_env.reset().await?;

    while !done {
        t += 1;
        // Sample an action from the action space
        let action = gym_env.action_space().sample();

        // Step through the environment
        let res = gym_env.step(&action).await?;

        obs = res.obs;
        done = res.done;
        reward = res.reward;

        println!("{}: Reward: {} Done: {}", t, reward, done);

        if done {
            gym_env.reset().await?;
        }
    }

    println!("Finished random action episode. Reward: {reward}");

    Ok(())
}
