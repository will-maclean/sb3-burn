use std::path::PathBuf;

use burn::{backend::{Autodiff, NdArray}, optim::{Adam, AdamConfig}, tensor::backend::AutodiffBackend};
use sb3_burn::{algorithm::{OfflineAlgParams, OfflineAlgorithm, OfflineTrainer}, buffer::ReplayBuffer, dqn::{DQNConfig, DQNNet}, env::{Env, GridWorldEnv}, logger::{CsvLogger, Logger}};

extern crate sb3_burn;

fn main(){
    type TrainingBacked = Autodiff<NdArray>;
        let device = Default::default();
        let config_optimizer = AdamConfig::new();
        let optim = config_optimizer.init();
        let offline_params = OfflineAlgParams::new()
            .with_batch_size(32)
            .with_memory_size(300)
            .with_n_steps(1000)
            .with_warmup_steps(100)
            .with_lr(1e-3);

        let env = GridWorldEnv::default();
        let q = DQNNet::<TrainingBacked>::init(
            &device,
            env.observation_space().clone(),
            env.action_space().clone(),
            16,
        );
        let dqn_alg = OfflineAlgorithm::DQN { q, optim, config: DQNConfig::new() };
        let buffer = ReplayBuffer::new(
            offline_params.memory_size, 
            env.observation_space().size(), 
            env.action_space().size()
        );
        let logger = CsvLogger::new(PathBuf::from("logs/log.csv"), false, Some("global_step".to_string()));

        match logger.check_can_log(false) {
            Ok(_) => {},
            Err(err) => panic!("Error setting up logger: {err}"),
        }

        let mut trainer = OfflineTrainer::<Adam<<Autodiff<NdArray> as AutodiffBackend>::InnerBackend>, TrainingBacked>::new(
            offline_params,
            Box::new(env),
            dqn_alg,
            buffer,
            Box::new(logger),
            None
        );

        trainer.train();
}