use burn::{
    config::Config,
    module::Module,
    nn::{
        self,
        loss::{MseLoss, Reduction},
    },
    optim::{adaptor::OptimizerAdaptor, GradientsParams, Optimizer, SimpleOptimizer},
    tensor::{
        activation::relu,
        backend::{AutodiffBackend, Backend},
        ElementConversion, Tensor,
    },
};
use module::DQNNet;

use crate::{
    algorithm::{OfflineAlgParams, OfflineTrainer},
    buffer::ReplayBuffer,
    env::base::Env,
    eval::{evaluate_policy, EvalConfig},
    logger::{LogData, LogItem},
    policy::{Agent, Policy},
    spaces::Space,
    to_tensor::{ToTensorB, ToTensorF, ToTensorI},
    utils::linear_decay,
};

pub mod module;

pub struct DQNAgent<O, B, OS, AS, Q, const D: usize> 
where 
    O: SimpleOptimizer<B::InnerBackend>,
    B: AutodiffBackend,
    OS: Clone,
    AS: Clone,
    Q: DQNNet<B, D> + burn::module::AutodiffModule<B>,
{
    pub q1: Q,
    pub q2: Q,
    pub optim: OptimizerAdaptor<O, Q, B>,
    pub config: DQNConfig,
    pub last_update: usize,
    observation_space: Box<dyn Space<OS>>,
    action_space: Box<dyn Space<AS>>,
}

#[derive(Config)]
pub struct DQNConfig {
    #[config(default = 1.0)]
    eps_start: f32,
    #[config(default = 0.05)]
    eps_end: f32,
    #[config(default = 0.9)]
    eps_end_frac: f32,
    #[config(default = 1000)]
    update_every: usize,
}

impl<O, B, OS, AS, Q, const D: usize>
    DQNAgent<O, B, OS, AS, Q, D>
where
    O: SimpleOptimizer<B::InnerBackend>,
    B: AutodiffBackend,
    OS: Clone,
    AS: Clone,
    Q: DQNNet<B, D> + burn::module::AutodiffModule<B>,
{
    pub fn new(
        q1: Q,
        q2: Q,
        optim: OptimizerAdaptor<O, Q, B>,
        config: DQNConfig,
        observation_space: Box<dyn Space<OS>>,
        action_space: Box<dyn Space<AS>>,
    ) -> Self {
        Self {
            q1,
            q2,
            optim,
            config,
            last_update: 0,
            observation_space,
            action_space,
        }
    }
}

impl<O: SimpleOptimizer<B::InnerBackend>, B: AutodiffBackend, Q> Agent<B, Vec<f32>, usize>
    for DQNAgent<O, B, Vec<f32>, usize, Q, 2>
where 
    Q: DQNNet<B, 2> + burn::module::AutodiffModule<B>,
{
    fn act(
        &self,
        _global_step: usize,
        global_frac: f32,
        obs: &Vec<f32>,
        inference_device: &<B as Backend>::Device,
    ) -> (usize, LogItem) {
        let eps = linear_decay(
            global_frac,
            self.config.eps_start,
            self.config.eps_end,
            self.config.eps_end_frac,
        );

        let a: usize = if rand::random::<f32>() > eps {
            let state = obs.clone().to_tensor(inference_device).unsqueeze_dim(0);
            let q: Tensor<B, 1> = self.q1.forward(state).squeeze(0);
            q.argmax(0).into_scalar().elem::<i32>() as usize
        } else {
            self.action_space().sample()
        };

        let log = LogItem::default().push("eps".to_string(), LogData::Float(eps));

        (a, log)
    }

    fn train_step(
        &mut self,
        global_step: usize,
        replay_buffer: &ReplayBuffer<Vec<f32>, usize>,
        offline_params: &OfflineAlgParams,
        train_device: &<B as Backend>::Device,
    ) -> (Option<f32>, LogItem) {
        // sample from the replay buffer
        let batch_sample = replay_buffer.batch_sample(offline_params.batch_size);

        // can make this mut when we want to log stuff in the loss step
        let log = LogItem::default();

        let loss = match batch_sample {
            Some(sample) => {
                let states = sample.states.to_tensor(train_device);
                let actions = sample.actions.to_tensor(train_device).unsqueeze_dim(1);
                let next_states = sample.next_states.to_tensor(train_device);
                let rewards = sample.rewards.to_tensor(train_device).unsqueeze_dim(1);
                let terminated = sample.terminated.to_tensor(train_device).unsqueeze_dim(1);

                let q_vals_ungathered = self.q1.forward(states);
                let q_vals = q_vals_ungathered.gather(1, actions);
                let next_q_vals_ungathered = self.q2.forward(next_states);
                let next_q_vals = next_q_vals_ungathered.max_dim(1);

                //FIXME: check that we should be using terminated and essentially ignoring truncated
                let targets =
                    rewards + terminated.bool_not().float() * next_q_vals * offline_params.gamma;

                let loss = MseLoss::new().forward(q_vals, targets, Reduction::Mean);

                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &self.q1);

                self.q1 = self.optim.step(offline_params.lr, self.q1.clone(), grads);

                if global_step > (self.last_update + self.config.update_every) {
                    // hard update
                    self.q2.update(&self.q1, None);
                    self.last_update = global_step
                }

                Some(loss.into_scalar().elem())
            }
            None => None,
        };

        (loss, log)
    }

    fn eval(
        &mut self,
        env: &mut dyn Env<Vec<f32>, usize>,
        cfg: &EvalConfig,
        eval_device: &B::Device,
    ) -> LogItem {
        let eval_result = evaluate_policy(self, env, cfg, eval_device);

        LogItem::default()
            .push(
                "eval_ep_mean_reward".to_string(),
                LogData::Float(eval_result.mean_reward),
            )
            .push(
                "eval_ep_mean_len".to_string(),
                LogData::Float(eval_result.mean_len),
            )
    }

    fn observation_space(&self) -> Box<dyn Space<Vec<f32>>> {
        dyn_clone::clone_box(&*self.observation_space)
    }

    fn action_space(&self) -> Box<dyn Space<usize>> {
        dyn_clone::clone_box(&*self.action_space)
    }
}

// #[cfg(test)]
// mod test {
//     use std::path::PathBuf;

//     use burn::{
//         backend::{Autodiff, NdArray},
//         optim::{Adam, AdamConfig},
//         tensor::backend::AutodiffBackend,
//     };

//     use crate::{
//         algorithm::{OfflineAlgParams, OfflineAlgorithm},
//         buffer::ReplayBuffer,
//         dqn::{DQNAgent, DQNConfig, DQNNet},
//         env::{base::Env, gridworld::GridWorldEnv},
//         eval::EvalConfig,
//         logger::CsvLogger,
//     };

//     use super::OfflineTrainer;

//     #[test]
//     fn test_dqn_lightweight() {
//         type TrainingBacked = Autodiff<NdArray>;
//         let device = Default::default();
//         let config_optimizer = AdamConfig::new();
//         let optim = config_optimizer.init();
//         let offline_params = OfflineAlgParams::new()
//             .with_n_steps(10)
//             .with_batch_size(2)
//             .with_memory_size(5)
//             .with_warmup_steps(2);
//         let env = GridWorldEnv::default();
//         let q = DQNNet::<TrainingBacked>::init(
//             &device,
//             env.observation_space().clone(),
//             env.action_space().clone(),
//             2,
//         );
//         let agent = DQNAgent::new(
//             q.clone(),
//             q,
//             optim,
//             DQNConfig::new(),
//             env.observation_space(),
//             env.action_space()
//         );
//         let dqn_alg = OfflineAlgorithm::DQN(agent);
//         let buffer = ReplayBuffer::new(offline_params.memory_size);

//         // create the logs dir
//         let mut log_dir = std::env::current_dir().unwrap();
//         log_dir.push("tmp_logs");
//         let _ = std::fs::create_dir(&log_dir);

//         let logger = CsvLogger::new(
//             PathBuf::from("tmp_logs/log.csv"),
//             true,
//             Some("global_step".to_string()),
//         );

//         let mut trainer = OfflineTrainer::new(
//             offline_params,
//             Box::new(env),
//             Box::<GridWorldEnv>::default(),
//             dqn_alg,
//             buffer,
//             Box::new(logger),
//             None,
//             EvalConfig::new(),
//             &device,
//             &device,
//         );

//         trainer.train();

//         let _ = std::fs::remove_dir_all(log_dir);
//     }
// }
