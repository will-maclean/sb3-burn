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

use crate::{
    algorithm::{OfflineAlgParams, OfflineTrainer},
    buffer::ReplayBuffer,
    logger::{LogData, LogItem},
    policy::Policy,
    spaces::{Action, ActionSpace, Obs, ObsSpace, ObsT},
    utils::{linear_decay, module_update::update_linear},
};

pub struct DQNAgent<O: SimpleOptimizer<B::InnerBackend>, B: AutodiffBackend> {
    pub q1: DQNNet<B>,
    pub q2: DQNNet<B>,
    pub optim: OptimizerAdaptor<O, DQNNet<B>, B>,
    pub config: DQNConfig,
    pub last_update: usize,
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

#[derive(Module, Debug)]
pub struct DQNNet<B: Backend> {
    l1: nn::Linear<B>,
    l2: nn::Linear<B>,
    l3: nn::Linear<B>,
    adv: nn::Linear<B>,
}

impl<B: Backend> DQNNet<B> {
    pub fn init(
        device: &B::Device,
        observation_space: ObsSpace,
        action_space: ActionSpace,
        hidden_size: usize,
    ) -> Self {
        match action_space {
            ActionSpace::Continuous { lows: _, highs: _ } => {
                panic!("Continuous actions are not supported by DQN")
            }
            ActionSpace::Discrete { size: action_size } => {
                let input_size = observation_space.size();

                Self {
                    l1: nn::LinearConfig::new(input_size, hidden_size).init(device),
                    l2: nn::LinearConfig::new(hidden_size, hidden_size).init(device),
                    l3: nn::LinearConfig::new(hidden_size, action_size).init(device),
                    adv: nn::LinearConfig::new(hidden_size, 1).init(device),
                }
            }
        }
    }

    pub fn forward(&self, state: ObsT<B, 2>) -> Tensor<B, 2> {
        let x = relu(self.l1.forward(state));
        let x = relu(self.l2.forward(x));
        self.adv.forward(x.clone()) - self.l3.forward(x)
    }
}

impl<B: Backend> Policy<B> for DQNNet<B> {
    fn act(&self, state: &Obs, action_space: ActionSpace) -> (Action, Option<LogItem>) {
        let binding = self.devices();
        let device = binding.first().unwrap();

        let state_tensor = state
            .clone()
            .to_train_tensor()
            .to_device(device)
            .unsqueeze_dim(0);
        let q_vals = self.predict(state_tensor);
        let a: i32 = q_vals.squeeze::<1>(0).argmax(0).into_scalar().elem();

        (
            Action::Discrete {
                space: action_space,
                idx: a,
            },
            None,
        )
    }

    fn predict(&self, state: ObsT<B, 2>) -> Tensor<B, 2> {
        let binding = self.devices();
        let device = binding.first().unwrap();

        self.forward(state.to_device(device))
    }

    fn update(&mut self, from: &Self, tau: Option<f32>) {
        self.l1 = update_linear(&from.l1, self.l1.clone(), tau);
        self.l2 = update_linear(&from.l2, self.l2.clone(), tau);
        self.l3 = update_linear(&from.l3, self.l3.clone(), tau);
        self.adv = update_linear(&from.adv, self.adv.clone(), tau);
    }
}

impl<O: SimpleOptimizer<B::InnerBackend>, B: AutodiffBackend> DQNAgent<O, B> {
    pub fn new(
        q1: DQNNet<B>,
        q2: DQNNet<B>,
        optim: OptimizerAdaptor<O, DQNNet<B>, B>,
        config: DQNConfig,
    ) -> Self {
        Self {
            q1,
            q2,
            optim,
            config,
            last_update: 0,
        }
    }
    pub fn act(
        &self,
        step: usize,
        state: &Obs,
        trainer: &OfflineTrainer<O, B>,
    ) -> (Action, Option<LogItem>) {
        let eps = linear_decay(
            step as f32 / trainer.offline_params.n_steps as f32,
            self.config.eps_start,
            self.config.eps_end,
            self.config.eps_end_frac,
        );

        if rand::random::<f32>() > eps {
            let (a, log) = self.q1.act(state, trainer.env.action_space().clone());

            match log {
                Some(mut log) => {
                    log = log.push("eps".to_string(), LogData::Float(eps));
                    (a, Some(log))
                }
                None => {
                    let mut log = LogItem::default();
                    log = log.push("eps".to_string(), LogData::Float(eps));
                    (a, Some(log))
                }
            }
        } else {
            let a = trainer.env.action_space().sample();
            let mut log = LogItem::default();
            log = log.push("eps".to_string(), LogData::Float(eps));

            (a, Some(log))
        }
    }

    pub fn train_step(
        &mut self,
        global_step: usize,
        replay_buffer: &ReplayBuffer<B>,
        offline_params: &OfflineAlgParams,
        train_device: &B::Device,
    ) -> Option<f32> {
        // sample from the replay buffer
        let batch_sample = replay_buffer.batch_sample(offline_params.batch_size);

        match batch_sample {
            Some(mut sample) => {
                sample.to_device(train_device);

                let q_vals_ungathered = self.q1.forward(sample.states);
                let q_vals = q_vals_ungathered.gather(1, sample.actions.int());
                let next_q_vals_ungathered = self.q2.forward(sample.next_states);
                let next_q_vals = next_q_vals_ungathered.max_dim(1);
                let targets = sample.rewards
                    + sample.dones.bool().bool_not().float() * next_q_vals * offline_params.gamma;

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
        }
    }
}

#[cfg(test)]
mod test {
    use std::path::PathBuf;

    use burn::{
        backend::{Autodiff, NdArray},
        optim::{Adam, AdamConfig},
        tensor::backend::AutodiffBackend,
    };

    use crate::{
        algorithm::{OfflineAlgParams, OfflineAlgorithm},
        buffer::ReplayBuffer,
        dqn::{DQNAgent, DQNConfig, DQNNet},
        env::{base::Env, gridworld::GridWorldEnv},
        eval::EvalConfig,
        logger::CsvLogger,
    };

    use super::OfflineTrainer;

    #[test]
    fn test_dqn_lightweight() {
        type TrainingBacked = Autodiff<NdArray>;
        let device = Default::default();
        let config_optimizer = AdamConfig::new();
        let optim = config_optimizer.init();
        let offline_params = OfflineAlgParams::new()
            .with_n_steps(10)
            .with_batch_size(2)
            .with_memory_size(5)
            .with_warmup_steps(2);
        let env = GridWorldEnv::default();
        let q = DQNNet::<TrainingBacked>::init(
            &device,
            env.observation_space().clone(),
            env.action_space().clone(),
            2,
        );
        let agent = DQNAgent::<
            Adam<<Autodiff<NdArray> as AutodiffBackend>::InnerBackend>,
            TrainingBacked,
        >::new(q.clone(), q, optim, DQNConfig::new());
        let dqn_alg = OfflineAlgorithm::DQN(agent);
        let buffer = ReplayBuffer::new(
            offline_params.memory_size,
            env.observation_space().size(),
            env.action_space().size(),
            &device,
        );

        // create the logs dir
        let mut log_dir = std::env::current_dir().unwrap();
        log_dir.push("tmp_logs");
        let _ = std::fs::create_dir(&log_dir);

        let logger = CsvLogger::new(
            PathBuf::from("tmp_logs/log.csv"),
            true,
            Some("global_step".to_string()),
        );

        let mut trainer = OfflineTrainer::new(
            offline_params,
            Box::new(env),
            Box::<GridWorldEnv>::default(),
            dqn_alg,
            buffer,
            Box::new(logger),
            None,
            EvalConfig::new(),
            &device,
            &device,
        );

        trainer.train();

        let _ = std::fs::remove_dir_all(log_dir);
    }
}
