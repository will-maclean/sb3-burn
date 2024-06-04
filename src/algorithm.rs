use burn::config::Config;
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{GradientsParams, Optimizer, SimpleOptimizer};
use burn::tensor::ElementConversion;
use burn::tensor::{backend::AutodiffBackend, Int, Tensor};

use crate::logger::{LogData, Logger};
use crate::utils::{linear_decay, mean};
use crate::{buffer::ReplayBuffer, env::Env, policy::Policy, spaces::SpaceSample};

use crate::dqn::{dqn_act, dqn_train_step, DQNConfig, DQNNet};

#[derive(Config)]
pub struct OfflineAlgParams {
    #[config(default = 10000)]
    pub n_steps: usize,
    #[config(default = 16)]
    pub batch_size: usize,
    #[config(default = 100)]
    pub warmup_steps: usize,
    #[config(default = 0.99)]
    pub gamma: f32,
    #[config(default = 1e-4)]
    pub lr: f64,
}

// I think this current layout will do for now, will likely need to be refactored at some point
pub enum OfflineAlgorithm<O: SimpleOptimizer<B::InnerBackend>, B: AutodiffBackend> {
    DQN {
        q: DQNNet<B>,
        optim: OptimizerAdaptor<O, DQNNet<B>, B>,
        config: DQNConfig,
    },
}

impl<O: SimpleOptimizer<B::InnerBackend>, B: AutodiffBackend> OfflineAlgorithm<O, B> {
    fn train_step(
        &mut self,
        replay_buffer: &ReplayBuffer<B>,
        offline_params: &OfflineAlgParams,
        device: &B::Device,
    ) -> Option<f32> {
        match self {
            OfflineAlgorithm::DQN { q, optim, config: _ } => {
                dqn_train_step::<O, B>(q, optim, replay_buffer, offline_params, device)
            }
        }
    }

    fn act(&self, state: &SpaceSample, step: usize, trainer: &OfflineTrainer<O, B>) -> SpaceSample {
        match self {
            OfflineAlgorithm::DQN { q, optim: _, config } => {
                dqn_act::<O, B>(q, step, config, state, trainer)
            }
        }
    }
}

pub struct OfflineTrainer<O: SimpleOptimizer<B::InnerBackend>, B: AutodiffBackend> {
    pub offline_params: OfflineAlgParams,
    pub env: Box<dyn Env>,
    pub algorithm: OfflineAlgorithm<O, B>,
    pub buffer: ReplayBuffer<B>,
    pub logger: Box<dyn Logger>,
}

impl<O: SimpleOptimizer<B::InnerBackend>, B: AutodiffBackend> OfflineTrainer<O, B> {
    pub fn train(&mut self) {
        let mut state = self.env.reset();

        let device = B::Device::default();

        let mut running_loss = Vec::new();
        let mut running_reward = 0.0;

        for i in 0..self.offline_params.n_steps {
            let action: SpaceSample;

            if i < self.offline_params.warmup_steps {
                action = self.env.action_space().sample();
            } else {
                action = self.algorithm.act(&state, i, &self);
            }

            let step_res = self.env.step(&action);
            let (next_obs, reward, done) = (step_res.obs, step_res.reward, step_res.done);

            running_reward += reward;

            self.buffer
                .add(state, action, next_obs.clone(), reward, done);

            if i >= self.offline_params.warmup_steps {
                if let Some(loss) =
                    self.algorithm
                        .train_step(&self.buffer, &self.offline_params, &device)
                {
                    running_loss.push(loss);
                }
            }

            if done {
                state = self.env.reset();

                self.logger.log(
                    vec![
                        (("global_step").to_string(), LogData::Int(i as i32)),
                        (
                            ("mean_loss").to_string(),
                            LogData::Float(mean(&running_loss)),
                        ),
                        (("ep_reward").to_string(), LogData::Float(running_reward)),
                    ]
                    .into_iter()
                    .collect(),
                )
            } else {
                state = next_obs;
            }
        }
    }
}
