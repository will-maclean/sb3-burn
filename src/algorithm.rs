use std::time;

use burn::config::Config;
use burn::optim::SimpleOptimizer;
use burn::tensor::backend::AutodiffBackend;
use indicatif::{ProgressIterator, ProgressStyle};

use crate::buffer::ReplayBuffer;
use crate::callback::{Callback, EmptyCallback};
use crate::env::base::Env;
use crate::eval::{evaluate_policy, EvalConfig};
use crate::logger::{LogData, LogItem, Logger};
use crate::spaces::{Action, Obs};
use crate::utils::mean;

use crate::dqn::DQNAgent;

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
    #[config(default = false)]
    pub render: bool,
    #[config(default = true)]
    evaluate_during_training: bool,
    #[config(default = 1)]
    pub evaluate_every_steps: usize,
    #[config(default = true)]
    pub eval_at_start_of_training: bool,
    #[config(default = true)]
    pub eval_at_end_of_training: bool,
    #[config(default = 1)]
    pub train_every: usize,
    #[config(default = 1)]
    pub grad_steps: usize,
}

// I think this current layout will do for now, will likely need to be refactored at some point
pub enum OfflineAlgorithm<O: SimpleOptimizer<B::InnerBackend>, B: AutodiffBackend> {
    DQN(DQNAgent<O, B>),
}

impl<O: SimpleOptimizer<B::InnerBackend>, B: AutodiffBackend> OfflineAlgorithm<O, B> {
    fn train_step(
        &mut self,
        global_step: usize,
        replay_buffer: &ReplayBuffer<B>,
        offline_params: &OfflineAlgParams,
        train_device: &B::Device,
    ) -> Option<f32> {
        match self {
            OfflineAlgorithm::DQN(agent) => {
                agent.train_step(global_step, replay_buffer, offline_params, train_device)
            }
        }
    }

    fn act(
        &self,
        state: &Obs,
        step: usize,
        trainer: &OfflineTrainer<O, B>,
    ) -> (Action, Option<LogItem>) {
        match self {
            OfflineAlgorithm::DQN(agent) => agent.act(step, state, trainer),
        }
    }

    fn eval(&self, env: &mut dyn Env, cfg: &EvalConfig, logger: &mut dyn Logger) {
        let eval_result = match self {
            OfflineAlgorithm::DQN(agent) => evaluate_policy(&agent.q1, env, cfg),
        };

        logger.log(
            LogItem::default()
                .push(
                    "eval_ep_mean_reward".to_string(),
                    LogData::Float(eval_result.mean_reward),
                )
                .push(
                    "eval_ep_mean_len".to_string(),
                    LogData::Float(eval_result.mean_len),
                ),
        );
    }
}

pub struct OfflineTrainer<'a, O: SimpleOptimizer<B::InnerBackend>, B: AutodiffBackend> {
    pub offline_params: OfflineAlgParams,
    pub env: Box<dyn Env>,
    eval_env: Box<dyn Env>,
    pub algorithm: OfflineAlgorithm<O, B>,
    pub buffer: ReplayBuffer<B>,
    pub logger: Box<dyn Logger>,
    pub callback: Box<dyn Callback<O, B>>,
    pub eval_cfg: EvalConfig,
    pub train_device: &'a B::Device,
    pub buffer_device: &'a B::Device,
}

impl<'a, O: SimpleOptimizer<B::InnerBackend>, B: AutodiffBackend> OfflineTrainer<'a, O, B> {
    pub fn new(
        offline_params: OfflineAlgParams,
        env: Box<dyn Env>,
        eval_env: Box<dyn Env>,
        algorithm: OfflineAlgorithm<O, B>,
        buffer: ReplayBuffer<B>,
        logger: Box<dyn Logger>,
        callback: Option<Box<dyn Callback<O, B>>>,
        eval_cfg: EvalConfig,
        train_device: &'a B::Device,
        buffer_device: &'a B::Device,
    ) -> Self {
        let c = match callback {
            Some(callback) => callback,
            None => Box::new(EmptyCallback {}),
        };

        Self {
            offline_params,
            env,
            eval_env,
            algorithm,
            buffer,
            logger,
            callback: c,
            eval_cfg,
            train_device,
            buffer_device,
        }
    }

    pub fn train(&mut self) {
        
        self.callback.on_training_start(self);

        let mut running_loss = Vec::new();
        let mut running_reward = 0.0;
        let mut episodes = 0;
        let mut ep_len = 0;

        if self.offline_params.eval_at_start_of_training {
            self.algorithm
                .eval(&mut *self.eval_env, &self.eval_cfg, &mut *self.logger);
        }

        let style = ProgressStyle::default_bar()
            .template("{pos:>7}/{len:7} {bar} [{elapsed_precise}], eta: [{eta}]")
            .unwrap();

        let mut state = self.env.reset();
        let mut ep_start_time = time::Instant::now();

        for i in (0..self.offline_params.n_steps).progress_with_style(style) {
            let (action, log) = match i < self.offline_params.warmup_steps {
                true => (self.env.action_space().sample(), None),
                false => self.algorithm.act(&state, i, self),
            };

            if let Some(log) = log {
                self.logger.log(log);
            }

            let step_res = self.env.step(&action);

            if self.offline_params.render & self.env.renderable() {
                self.env.render();
            }

            let (next_obs, reward, done) = (step_res.obs.clone(), step_res.reward, step_res.done);

            running_reward += reward;
            ep_len += 1;

            self.buffer
                .add(state, action, next_obs.clone(), reward, done);

            if (i >= self.offline_params.warmup_steps) & (i % self.offline_params.train_every == 0)
            {
                for _ in 0..self.offline_params.grad_steps {
                    let loss = self.algorithm.train_step(
                        i,
                        &self.buffer,
                        &self.offline_params,
                        self.train_device,
                    );
                    self.callback.on_step(self, i, step_res.clone(), loss);

                    if let Some(loss) = loss {
                        running_loss.push(loss);
                    }
                }
            }

            if self.offline_params.evaluate_during_training
                & (i % self.offline_params.evaluate_every_steps == 0)
            {
                self.algorithm
                    .eval(&mut *self.eval_env, &self.eval_cfg, &mut *self.logger);
            }

            if done {
                let ep_end_time = time::Instant::now();
                let ep_fps = (ep_len as f32) / (ep_end_time - ep_start_time).as_secs_f32();
                self.logger.log(
                    LogItem::default()
                        .push("global_step".to_string(), LogData::Int(i as i32))
                        .push("ep_num".to_string(), LogData::Int(episodes))
                        .push("mean_loss".to_string(), LogData::Float(mean(&running_loss)))
                        .push("ep_reward".to_string(), LogData::Float(running_reward))
                        .push("ep_len".to_string(), LogData::Int(ep_len))
                        .push("ep_fps".to_string(), LogData::Float(ep_fps)),
                );

                ep_start_time = time::Instant::now();
                state = self.env.reset();
                episodes += 1;
                running_reward = 0.0;
                running_loss = Vec::new();
                ep_len = 0;
            } else {
                state = next_obs;
            }
        }

        if self.offline_params.eval_at_end_of_training {
            self.algorithm
                .eval(&mut *self.eval_env, &self.eval_cfg, &mut *self.logger);
        }

        self.callback.on_training_end(self);
        let _ = self.logger.dump();
    }
}
