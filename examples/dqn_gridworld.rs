use std::path::PathBuf;

use burn::{
    backend::{libtorch::LibTorchDevice, Autodiff, LibTorch},
    grad_clipping::GradientClippingConfig,
    optim::{Adam, AdamConfig},
};
use sb3_burn::{
    common::{
        algorithm::{OfflineAlgParams, OfflineTrainer},
        buffer::ReplayBuffer,
        eval::EvalConfig,
        logger::{CsvLogger, Logger},
    },
    dqn::{module::LinearAdvDQNNet, DQNAgent, DQNConfig},
    env::{base::Env, gridworld::GridWorldEnv},
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
        .with_batch_size(50)
        .with_memory_size(1000)
        .with_n_steps(10000)
        .with_warmup_steps(50)
        .with_lr(1e-3);

    let env = GridWorldEnv::default();
    let q = LinearAdvDQNNet::<TrainingBacked>::init(
        &train_device,
        env.observation_space().shape().len(),
        env.action_space().shape(),
        16,
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

    let logger = CsvLogger::new(PathBuf::from("logs/dqn_gridworld/dqn_gridworld.csv"), false);

    match logger.check_can_log(false) {
        Ok(_) => {}
        Err(err) => panic!("Error setting up logger: {err}"),
    }

    let mut trainer: OfflineTrainer<_, Adam<LibTorch>, _, _, _> = OfflineTrainer::new(
        offline_params,
        Box::new(env),
        Box::new(GridWorldEnv::default()),
        agent,
        buffer,
        Box::new(logger),
        None,
        EvalConfig::new().with_n_eval_episodes(10),
        &train_device,
    );

    trainer.train();
}
