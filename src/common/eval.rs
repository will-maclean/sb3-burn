use burn::{config::Config, tensor::backend::Backend};
use core::fmt::Debug;

use crate::{common::utils::mean, env::base::Env};

use super::agent::Agent;

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

pub fn evaluate_policy<B: Backend, A: Agent<B, OS, AS>, OS: Clone + Debug, AS: Clone + Debug>(
    agent: &mut A,
    env: &mut dyn Env<OS, AS>,
    cfg: &EvalConfig,
    device: &B::Device,
) -> EvalResult {
    let mut episode_rewards = Vec::new();
    let mut episode_lengths = Vec::new();
    let mut completed_episodes = 0;

    let mut state = env.reset(None, None);
    let mut running_reward = 0.0;
    let mut ep_len = 0.0;

    println!("Starting evaluation");

    while completed_episodes < cfg.n_eval_episodes {
        if cfg.print_obs {
            println!("state: {:?}", state);
        }

        let (action, _) = agent.act(0, 1.0, &state, true, device);

        if cfg.print_action {
            println!("action: {:?}", action);
        }

        let step_sample = env.step(&action);

        let done = step_sample.terminated | step_sample.truncated;
        running_reward += step_sample.reward;
        ep_len += 1.0;

        if cfg.print_reward {
            println!("reward: {:?}", step_sample.reward);
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

            state = env.reset(None, None);
        } else {
            state = step_sample.obs;
        }
    }

    EvalResult {
        mean_len: mean(&episode_lengths),
        mean_reward: mean(&episode_rewards),
    }
}
