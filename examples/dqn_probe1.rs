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
    env::{base::Env, probe::ProbeEnvValueTest},
    eval::EvalConfig,
    logger::{CsvLogger, Logger},
};

extern crate sb3_burn;

fn main() {
    // Using parameters from:
    // https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml

    type TrainingBacked = Autodiff<LibTorch>;

    let train_device = LibTorchDevice::Cuda(0);

    let config_optimizer =
        AdamConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(10.0)));
    let optim = config_optimizer.init();
    let offline_params = OfflineAlgParams::new()
        .with_batch_size(10)
        .with_memory_size(1000)
        .with_n_steps(1000)
        .with_warmup_steps(50)
        .with_lr(5e-3)
        .with_eval_at_end_of_training(true)
        .with_eval_at_end_of_training(true)
        .with_evaluate_during_training(false);

    let env = ProbeEnvValueTest::default();
    let q = LinearAdvDQNNet::<TrainingBacked>::init(
        &train_device,
        env.observation_space().shape().len(),
        env.action_space().shape(),
        1,
    );
    let dqn_config = DQNConfig::new();
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
    );

    match logger.check_can_log(false) {
        Ok(_) => {}
        Err(err) => panic!("Error setting up logger: {err}"),
    }

    let mut trainer: OfflineTrainer<Adam<LibTorch>, Autodiff<LibTorch>, Vec<f32>, usize> =
        OfflineTrainer::new(
            offline_params,
            Box::new(env),
            Box::new(ProbeEnvValueTest::default()),
            Box::new(agent),
            buffer,
            Box::new(logger),
            None,
            EvalConfig::new().with_n_eval_episodes(10),
            &train_device,
        );

    trainer.train();
}
