use std::fmt::Debug;
use std::time;

use burn::config::Config;
use burn::optim::SimpleOptimizer;
use burn::tensor::backend::AutodiffBackend;
use indicatif::{ProgressIterator, ProgressStyle};

use crate::env::base::Env;

use super::{
    agent::Agent,
    buffer::ReplayBuffer,
    callback::{Callback, EmptyCallback},
    eval::{evaluate_policy, EvalConfig},
    logger::{LogData, LogItem, Logger},
    utils::mean,
};
use crate::common::timer::Profiler;

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
    #[config(default = false)]
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
    #[config(default = false)]
    /// Enable per-phase timing with low-overhead Instants
    pub profile_timers: bool,
    #[config(default = 250)]
    /// Log timing averages every N steps (when profiling)
    pub profile_log_every_steps: usize,
}

pub struct OfflineTrainer<
    'a,
    A: Agent<B, OS, AS>,
    B: AutodiffBackend,
    OS: Clone,
    AS: Clone,
> {
    pub offline_params: OfflineAlgParams,
    pub env: Box<dyn Env<OS, AS>>,
    pub eval_env: Box<dyn Env<OS, AS>>,
    pub agent: A,
    pub buffer: ReplayBuffer<OS, AS>,
    pub logger: Box<dyn Logger>,
    pub callback: Box<dyn Callback<A, B, OS, AS>>,
    pub eval_cfg: EvalConfig,
    pub train_device: &'a B::Device,
}

impl<
        'a,
        A: Agent<B, OS, AS>,
        B: AutodiffBackend,
        OS: Clone + Debug,
        AS: Clone + Debug,
    > OfflineTrainer<'a, A, B, OS, AS>
{
    pub fn new(
        offline_params: OfflineAlgParams,
        env: Box<dyn Env<OS, AS>>,
        eval_env: Box<dyn Env<OS, AS>>,
        agent: A,
        buffer: ReplayBuffer<OS, AS>,
        logger: Box<dyn Logger>,
        callback: Option<Box<dyn Callback<A, B, OS, AS>>>,
        eval_cfg: EvalConfig,
        train_device: &'a B::Device,
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
        }
    }

    pub fn train(&mut self) {
        println!("Starting training...");
        self.callback.on_training_start(self);

        let mut running_loss = Vec::new();
        let mut running_reward = 0.0;
        let mut episodes = 0;
        let mut ep_len = 0;

        if self.offline_params.eval_at_start_of_training {
            let log: LogItem = evaluate_policy(
                &mut self.agent,
                &mut *self.eval_env,
                &self.eval_cfg,
                self.train_device,
            )
            .into();

            self.logger.log(log);
        }

        let style = ProgressStyle::default_bar()
            .template("{pos:>7}/{len:7} {bar} [{elapsed_precise}], eta: [{eta}]")
            .unwrap();

        let mut state = self.env.reset(None, None);
        let mut ep_start_time = time::Instant::now();

        // Reusable interval profiler for the trainer loop
        let mut trainer_prof = Profiler::new(self.offline_params.profile_timers);

        for i in (0..self.offline_params.n_steps).progress_with_style(style) {
            let loop_start = time::Instant::now();
            let (action, log) = match i < self.offline_params.warmup_steps {
                true => (self.env.action_space().sample(), Default::default()),
                false => {
                    let t0 = time::Instant::now();
                    let res = self.agent.act(
                        i,
                        (i as f32) / (self.offline_params.n_steps as f32),
                        &state,
                        false,
                        self.train_device,
                    );
                        trainer_prof.record("act", t0.elapsed().as_secs_f64());
                    res
                }
            };

            self.logger.log(log);

            let t0 = time::Instant::now();
            let step_res = self.env.step(&action);
                trainer_prof.record("env_step", t0.elapsed().as_secs_f64());

            if self.offline_params.render & self.env.renderable() {
                let t0 = time::Instant::now();
                self.env.render();
                    trainer_prof.record("render", t0.elapsed().as_secs_f64());
            }

            let done = step_res.terminated | step_res.truncated;

            running_reward += step_res.reward;
            ep_len += 1;

            let t0 = time::Instant::now();
            self.buffer.add(
                state,
                action,
                step_res.obs.clone(),
                step_res.reward,
                step_res.terminated,
                step_res.truncated,
            );
                trainer_prof.record("buf_add", t0.elapsed().as_secs_f64());

            if (i >= self.offline_params.warmup_steps) & (i % self.offline_params.train_every == 0)
            {
                let t_train0 = time::Instant::now();
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

                    self.logger.log(log)
                }
                    trainer_prof.record("train", t_train0.elapsed().as_secs_f64());
            }

            if self.offline_params.evaluate_during_training
                & (i % self.offline_params.evaluate_every_steps == 0)
            {
                let t0 = time::Instant::now();
                let log: LogItem = evaluate_policy(
                    &mut self.agent,
                    &mut *self.eval_env,
                    &self.eval_cfg,
                    self.train_device,
                )
                .into();

                self.logger.log(log);
                    trainer_prof.record("eval", t0.elapsed().as_secs_f64());
            }

            if self.offline_params.profile_timers {
                trainer_prof.record("loop", loop_start.elapsed().as_secs_f64());

                if (i + 1) % self.offline_params.profile_log_every_steps == 0 {
                    if let Some(item) = trainer_prof
                        .into_logitem(i, self.offline_params.profile_log_every_steps, Some("trainer_"))
                    {
                        self.logger.log(item);
                    }
                    trainer_prof.reset();

                    // Also flush agent-level profiler averages, if any
                    if let Some(agent_log) = self
                        .agent
                        .profile_flush(i, self.offline_params.profile_log_every_steps)
                    {
                        self.logger.log(agent_log);
                    }
                }
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
            let log: LogItem = evaluate_policy(
                &mut self.agent,
                &mut *self.eval_env,
                &self.eval_cfg,
                self.train_device,
            )
            .into();

            self.logger.log(log);
        }

        println!("Training complete. Handling end-of-training procedures...");
        self.callback.on_training_end(self);
        let _ = self.logger.dump();
    }
}
