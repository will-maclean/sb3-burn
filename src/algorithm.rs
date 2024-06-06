use burn::config::Config;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Optimizer, SimpleOptimizer};
use burn::tensor::{backend::AutodiffBackend};
use indicatif::{ProgressIterator, ProgressStyle};

use crate::callback::{Callback, EmptyCallback};
use crate::logger::{LogData, Logger};
use crate::spaces::{Action, Obs};
use crate::utils::{mean};
use crate::{buffer::ReplayBuffer, env::Env, policy::Policy};

use crate::dqn::{DQNAgent, DQNConfig, DQNNet};

#[derive(Config)]
pub struct OfflineAlgParams {
    #[config(default = 100)]
    pub n_steps: usize,
    #[config(default = 50)]
    pub memory_size: usize,
    #[config(default = 4)]
    pub batch_size: usize,
    #[config(default = 10)]
    pub warmup_steps: usize,
    #[config(default = 0.99)]
    pub gamma: f32,
    #[config(default = 1e-4)]
    pub lr: f64,
}

// I think this current layout will do for now, will likely need to be refactored at some point
pub enum OfflineAlgorithm<O: SimpleOptimizer<B::InnerBackend>, B: AutodiffBackend> {
    DQN (DQNAgent<O, B>),
}

impl<O: SimpleOptimizer<B::InnerBackend>, B: AutodiffBackend> OfflineAlgorithm<O, B> {
    fn train_step(
        &mut self,
        replay_buffer: &ReplayBuffer<B>,
        offline_params: &OfflineAlgParams,
        device: &B::Device,
    ) -> Option<f32> {
        match self {
            OfflineAlgorithm::DQN (agent) => {
                agent.train_step(replay_buffer, offline_params, device)
            }
        }
    }

    fn act(&self, state: &Obs, step: usize, trainer: &OfflineTrainer<O, B>) -> Action {
        match self {
            OfflineAlgorithm::DQN (agent) => {
                agent.act(step, state, trainer)
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
    pub callback: Box<dyn Callback<O, B>>,
}

impl<O: SimpleOptimizer<B::InnerBackend>, B: AutodiffBackend> OfflineTrainer<O, B> {
    pub fn new(
        offline_params: OfflineAlgParams, 
        env: Box<dyn Env>,
        algorithm: OfflineAlgorithm<O, B>,
        buffer: ReplayBuffer<B>,
        logger: Box<dyn Logger>,
        callback: Option<Box<dyn Callback<O, B>>>,
    ) -> Self {
        let c: Box<dyn Callback<O, B>>;
        match callback {
            Some(callback) => c = callback,
            None => c = Box::new(EmptyCallback{}),
        }
        Self {
            offline_params,
            env,
            algorithm,
            buffer,
            logger,
            callback: c,
        }
    }

    pub fn train(&mut self) {
        let mut state = self.env.reset();

        let device = B::Device::default();

        self.callback.on_training_start(self);

        let mut running_loss = Vec::new();
        let mut running_reward = 0.0;
        let mut episodes = 0;

        let style = ProgressStyle::default_bar().template("{pos:>7}/{len:7} {bar} [{elapsed_precise}], eta: [{eta}]").unwrap();

        for i in (0..self.offline_params.n_steps).progress_with_style(style) {
            let action: Action;

            if i < self.offline_params.warmup_steps {
                action = self.env.action_space().sample();
            } else {
                action = self.algorithm.act(&state, i, self);
            }

            let step_res = self.env.step(&action);
            let (next_obs, reward, done) = (step_res.obs.clone(), step_res.reward, step_res.done);

            running_reward += reward;

            self.buffer
                .add(state, action, next_obs.clone(), reward, done);

            if i >= self.offline_params.warmup_steps {
                let loss = self.algorithm.train_step(&self.buffer, &self.offline_params, &device);
                self.callback.on_step(self, i, step_res.clone(), loss);

                match loss{
                    Some(loss) => running_loss.push(loss),
                    None => {},
                }
            }

            if done {
                state = self.env.reset();
                episodes += 1;
                running_reward = 0.0;
                running_loss = Vec::new();

                self.logger.log(
                    vec![
                        (("global_step").to_string(), LogData::Int(i as i32)),
                        (
                            ("mean_loss").to_string(),
                            LogData::Float(mean(&running_loss)),
                        ),
                        (("ep_reward").to_string(), LogData::Float(running_reward)),
                        (("ep_num").to_string(), LogData::Int(episodes)),
                    ]
                    .into_iter()
                    .collect(),
                )
            } else {
                state = next_obs;
            }
        }

        self.callback.on_training_end(self);
        let _ = self.logger.dump();
    }
}
