use std::path::PathBuf;

use burn::{
    backend::{Autodiff, NdArray},
    optim::AdamConfig,
};
use sb3_burn::{
    algorithm::{OfflineAlgParams, OfflineAlgorithm, OfflineTrainer},
    buffer::ReplayBuffer,
    dqn::{DQNAgent, DQNConfig, DQNNet},
    env::{base::Env, cartpole::CartpoleEnv},
    eval::EvalConfig,
    logger::{CsvLogger, Logger},
};

extern crate sb3_burn;

fn main() {
    type TrainingBacked = Autodiff<NdArray>;
    let device = Default::default();
    let config_optimizer = AdamConfig::new();
    let optim = config_optimizer.init();
    let offline_params = OfflineAlgParams::new()
        .with_batch_size(256)
        .with_memory_size(1000)
        .with_n_steps(50000)
        .with_warmup_steps(50)
        .with_lr(0.0005)
        .with_gamma(0.99)
        .with_eval_at_end_of_training(true)
        .with_eval_at_end_of_training(true)
        .with_evaluate_during_training(false);

    //i am in desperate need of love i am sosooooooooooooo sorry please do not hate, love, termites united, nad termite

    let env = CartpoleEnv::default();
    let q = DQNNet::<TrainingBacked>::init(
        &device,
        env.observation_space().clone(),
        env.action_space().clone(),
        64,
    );
    let agent = DQNAgent::new(q.clone(), q, optim, DQNConfig::new());
    let dqn_alg = OfflineAlgorithm::DQN(agent);
    let buffer = ReplayBuffer::new(
        offline_params.memory_size,
        env.observation_space().size(),
        env.action_space().size(),
    );
    let logger = CsvLogger::new(
        PathBuf::from("logs/log_dqn_cartpole.csv"),
        false,
        Some("global_step".to_string()),
    );

    match logger.check_can_log(false) {
        Ok(_) => {}
        Err(err) => panic!("Error setting up logger: {err}"),
    }

    let mut trainer = OfflineTrainer::new(
        offline_params,
        Box::new(env),
        Box::<CartpoleEnv>::default(),
        dqn_alg,
        buffer,
        Box::new(logger),
        None,
        EvalConfig::new()
            .with_n_eval_episodes(10),
    );

    trainer.train();
}
