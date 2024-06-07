use burn::{config::Config, tensor::backend::Backend};

use crate::{env::base::Env, policy::Policy, utils::mean};

pub struct EvalResult {
    pub mean_len: f32,
    pub mean_reward: f32,
}

#[derive(Config)]
pub struct EvalConfig {
    #[config(default = 10)]
    n_eval_episodes: usize,
    #[config(default = false)]
    print_obs: bool,
    #[config(default = false)]
    print_action: bool,
    #[config(default = false)]
    print_reward: bool,
    #[config(default = false)]
    print_done: bool,
    #[config(default = false)]
    print_prediction: bool,
}

pub fn evaluate_policy<B: Backend, P: Policy<B>>(
    policy: &P,
    env: &mut dyn Env,
    cfg: &EvalConfig,
) -> EvalResult {
    let mut episode_rewards = Vec::new();
    let mut episode_lengths = Vec::new();
    let mut completed_episodes = 0;

    let mut state = env.reset();
    let mut running_reward = 0.0;
    let mut ep_len = 0.0;

    println!("Starting evaluation");

    while completed_episodes < cfg.n_eval_episodes {
        if cfg.print_obs {
            println!("state: {:?}", state);
        }

        let action = policy.act(&state, env.action_space().clone());

        if cfg.print_prediction {
            let pred = policy.predict(state.to_train_tensor().unsqueeze_dim(0));

            println!("Prediction: {pred}");
        }

        if cfg.print_action {
            println!("action: {:?}", action);
        }

        let step_sample = env.step(&action);

        let (next_obs, reward, done) = (step_sample.obs, step_sample.reward, step_sample.done);
        running_reward += reward;
        ep_len += 1.0;

        if cfg.print_reward {
            println!("reward: {:?}", reward);
        }

        if cfg.print_done {
            println!("done: {:?}", done);
        }

        if done {
            episode_rewards.push(running_reward);
            episode_lengths.push(ep_len);
            completed_episodes += 1;

            running_reward = 0.0;
            ep_len = 0.0;

            state = env.reset();
        } else {
            state = next_obs;
        }
    }

    EvalResult {
        mean_len: mean(&episode_lengths),
        mean_reward: mean(&episode_rewards),
    }
}
