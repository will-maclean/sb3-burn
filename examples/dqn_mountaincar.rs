use std::path::PathBuf;

use burn::{
    backend::{libtorch::LibTorchDevice, Autodiff, LibTorch},
    grad_clipping::GradientClippingConfig,
    optim::{Adam, AdamConfig},
};
use sb3_burn::{
    algorithm::{OfflineAlgParams, OfflineTrainer},
    buffer::ReplayBuffer,
    dqn::{module::LinearAdvDQNNet, DQNAgent, DQNConfig},
    env::{base::Env, classic_control::mountain_car::MountainCarEnv},
    eval::EvalConfig,
    logger::{CsvLogger, Logger},
};

extern crate sb3_burn;

fn main() {
    // Using parameters from:
    // https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml

    type TrainingBacked = Autodiff<LibTorch>;

    let train_device = LibTorchDevice::Cuda(0);
    let buffer_device = LibTorchDevice::Cpu;

    let config_optimizer =
        AdamConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(10.0)));
    let optim = config_optimizer.init();
    let offline_params = OfflineAlgParams::new()
        .with_batch_size(128)
        .with_memory_size(10000)
        .with_n_steps(120000)
        .with_warmup_steps(1000)
        .with_lr(4e-3)
        .with_gamma(0.98)
        .with_eval_at_end_of_training(true)
        .with_eval_at_end_of_training(true)
        .with_evaluate_during_training(false)
        .with_grad_steps(8)
        .with_train_every(16);

    let env = MountainCarEnv::default();
    let q = LinearAdvDQNNet::<TrainingBacked>::init(
        &train_device,
        env.observation_space().shape().len(),
        env.action_space().shape(),
        256,
    );
    let dqn_config = DQNConfig::new()
        .with_update_every(600)
        .with_eps_end(0.07)
        .with_eps_end_frac(0.8);
    let agent = DQNAgent::new(
        q.clone(),
        q,
        optim,
        dqn_config,
        env.observation_space(),
        env.action_space(),
    );

    let buffer = ReplayBuffer::new(offline_params.memory_size);

    let logger = CsvLogger::new(
        PathBuf::from("logs/dqn_logging/log_dqn_cartpole.csv"),
        false,
        Some("global_step".to_string()),
    );

    match logger.check_can_log(false) {
        Ok(_) => {}
        Err(err) => panic!("Error setting up logger: {err}"),
    }

    let mut trainer: OfflineTrainer<Adam<LibTorch>, Autodiff<LibTorch>, Vec<f32>, usize> =
        OfflineTrainer::new(
            offline_params,
            Box::new(env),
            Box::new(MountainCarEnv::default()),
            Box::new(agent),
            buffer,
            Box::new(logger),
            None,
            EvalConfig::new().with_n_eval_episodes(10),
            &train_device,
            &buffer_device,
        );

    trainer.train();
}