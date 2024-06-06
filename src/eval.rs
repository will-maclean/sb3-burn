use std::rc::Rc;

use burn::{config::Config, tensor::backend::Backend};

use crate::{env::base::Env, policy::Policy, utils::mean};

pub struct EvalResult {
    pub mean_len: f32,
    pub mean_reward: f32,
}

#[derive(Config)]
pub struct EvalConfig{
    #[config(default = 10)]
    n_eval_episodes: usize,
}

pub fn evaluate_policy<B: Backend, P: Policy<B>> (
    policy: &P,
    mut env: Box<dyn Env>,
    cfg: &EvalConfig,
) -> EvalResult {

    let mut episode_rewards = Vec::new();
    let mut episode_lengths = Vec::new();
    let mut completed_episodes = 0;

    let mut state = env.reset();
    let mut running_reward = 0.0;
    let mut ep_len = 0.0;

    while completed_episodes < cfg.n_eval_episodes {
        let action = policy.act(&state, env.action_space().clone());

        let step_sample = env.step(&action);

        let (next_obs, reward, done) = (step_sample.obs, step_sample.reward, step_sample.done);
        running_reward += reward;
        ep_len += 1.0;

        if done {
            episode_rewards.push(running_reward);
            episode_lengths.push(ep_len);
            completed_episodes += 1;

            running_reward = 0.0;
            ep_len = 0.0;

        } else {
            state = next_obs;
        }
    }

    EvalResult {  
        mean_len: mean(&episode_lengths),
        mean_reward: mean(&episode_rewards)
    }
}