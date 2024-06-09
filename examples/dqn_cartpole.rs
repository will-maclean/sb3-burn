use std::path::PathBuf;

use burn::{
    backend::{libtorch::LibTorchDevice, Autodiff, LibTorch},
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
    type TrainingBacked = Autodiff<LibTorch>;

    let train_device = LibTorchDevice::Cuda(0);
    let buffer_device = LibTorchDevice::Cpu;

    let config_optimizer = AdamConfig::new();
    let optim = config_optimizer.init();
    let offline_params = OfflineAlgParams::new()
        .with_batch_size(256)
        .with_memory_size(50000)
        .with_n_steps(100000)
        .with_warmup_steps(50)
        .with_lr(1e-3)
        .with_gamma(0.95)
        .with_eval_at_end_of_training(true)
        .with_eval_at_end_of_training(true)
        .with_evaluate_during_training(false)
        .with_grad_steps(1)
        .with_train_every(1);

    let env = CartpoleEnv::default();
    let q = DQNNet::<TrainingBacked>::init(
        &train_device,
        env.observation_space().clone(),
        env.action_space().clone(),
        256,
    );
    let dqn_config = DQNConfig::new().with_update_every(500).with_eps_end(0.02);
    let agent = DQNAgent::new(q.clone(), q, optim, dqn_config);
    let dqn_alg = OfflineAlgorithm::DQN(agent);
    let buffer = ReplayBuffer::new(
        offline_params.memory_size,
        env.observation_space().size(),
        env.action_space().size(),
        &buffer_device,
    );
    let logger = CsvLogger::new(
        PathBuf::from("logs/dqn_logging/log_dqn_cartpole.csv"),
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
        EvalConfig::new().with_n_eval_episodes(10),
        &train_device,
        &buffer_device,
    );

    trainer.train();
}
