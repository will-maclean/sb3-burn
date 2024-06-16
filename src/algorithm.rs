use std::fmt::Debug;
use std::time;

use burn::config::Config;
use burn::optim::SimpleOptimizer;
use burn::tensor::backend::AutodiffBackend;
use indicatif::{ProgressIterator, ProgressStyle};

use crate::buffer::ReplayBuffer;
use crate::callback::{Callback, EmptyCallback};
use crate::env::base::Env;
use crate::eval::EvalConfig;
use crate::logger::{LogData, LogItem, Logger};
use crate::policy::Agent;
use crate::utils::mean;

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

pub struct OfflineTrainer<'a, O: SimpleOptimizer<B::InnerBackend>, B: AutodiffBackend, OS: Clone, AS: Clone> {
    pub offline_params: OfflineAlgParams,
    pub env: Box<dyn Env<OS, AS>>,
    pub eval_env: Box<dyn Env<OS, AS>>,
    pub agent: Box<dyn Agent<B, OS, AS>>,
    pub buffer: ReplayBuffer<OS, AS>,
    pub logger: Box<dyn Logger>,
    pub callback: Box<dyn Callback<O, B, OS, AS>>,
    pub eval_cfg: EvalConfig,
    pub train_device: &'a B::Device,
    pub buffer_device: &'a B::Device,
}

impl<'a, O: SimpleOptimizer<B::InnerBackend>, B: AutodiffBackend, OS: Clone + Debug, AS: Clone + Debug> OfflineTrainer<'a, O, B, OS, AS> {
    pub fn new(
        offline_params: OfflineAlgParams,
        env: Box<dyn Env<OS, AS>>,
        eval_env: Box<dyn Env<OS, AS>>,
        agent: Box<dyn Agent<B, OS, AS>>,
        buffer: ReplayBuffer<OS, AS>,
        logger: Box<dyn Logger>,
        callback: Option<Box<dyn Callback<O, B, OS, AS>>>,
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
            agent,
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
            let log = self.agent
                .eval(&mut *self.eval_env, &self.eval_cfg);

            //TODO: log the log
        }

        let style = ProgressStyle::default_bar()
            .template("{pos:>7}/{len:7} {bar} [{elapsed_precise}], eta: [{eta}]")
            .unwrap();

        let mut state = self.env.reset(None, None);
        let mut ep_start_time = time::Instant::now();

        for i in (0..self.offline_params.n_steps).progress_with_style(style) {
            let (action, log) = match i < self.offline_params.warmup_steps {
                true => (self.env.action_space().sample(), Default::default()),
                false => self.agent.act(
                    i,
                    (i as f32) / (self.offline_params.n_steps as f32),
                    &state,
                    self.train_device,
                ),
            };

            self.logger.log(log);

            let step_res = self.env.step(&action);

            if self.offline_params.render & self.env.renderable() {
                self.env.render();
            }

            let done = step_res.terminated | step_res.truncated;

            running_reward += step_res.reward;
            ep_len += 1;

            self.buffer.add(
                state,
                action,
                step_res.obs.clone(),
                step_res.reward,
                step_res.terminated,
                step_res.truncated,
            );

            if (i >= self.offline_params.warmup_steps) & (i % self.offline_params.train_every == 0)
            {
                for _ in 0..self.offline_params.grad_steps {
                    let (loss, log) = self.agent.train_step(
                        i,
                        &self.buffer,
                        &self.offline_params,
                        self.train_device,
                    );
                    self.callback.on_step(self, i, step_res.clone(), loss);

                    if let Some(loss) = loss {
                        running_loss.push(loss);
                    }

                    //TODO: log the log
                }
            }

            if self.offline_params.evaluate_during_training
                & (i % self.offline_params.evaluate_every_steps == 0)
            {
                let log = self.agent
                    .eval(&mut *self.eval_env, &self.eval_cfg);

                //TODO: log the log
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
                state = self.env.reset(None, None);
                episodes += 1;
                running_reward = 0.0;
                running_loss = Vec::new();
                ep_len = 0;
            } else {
                state = step_res.obs;
            }
        }

        if self.offline_params.eval_at_end_of_training {
            let log = self.agent
                .eval(&mut *self.eval_env, &self.eval_cfg);

            // TODO: log the logger
        }

        self.callback.on_training_end(self);
        let _ = self.logger.dump();
    }
}
