use std::path::PathBuf;

use burn::{
    backend::{libtorch::LibTorchDevice, Autodiff, LibTorch},
    grad_clipping::GradientClippingConfig,
    optim::AdamConfig,
};
use sb3_burn::{
    common::{
        algorithm::{OfflineAlgParams, OfflineTrainer},
        buffer::ReplayBuffer,
        eval::EvalConfig,
        logger::{CsvLogger, Logger},
        utils::sb3_seed,
    },
    dqn::{module::LinearDQNNet, DQNAgent, DQNConfig},
    env::{base::Env, probe::ProbeEnvStateActionTest},
};

extern crate sb3_burn;

fn main() {
    // Using parameters from:
    // https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml

    type TrainingBacked = Autodiff<LibTorch>;

    let train_device = LibTorchDevice::default();
    sb3_seed::<TrainingBacked>(1234, &train_device);

    let config_optimizer =
        AdamConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(10.0)));
    let optim = config_optimizer.init();
    let offline_params = OfflineAlgParams::new()
        .with_batch_size(16)
        .with_memory_size(1000)
        .with_n_steps(1000)
        .with_warmup_steps(50)
        .with_lr(0.1)
        .with_eval_at_end_of_training(true)
        .with_eval_at_end_of_training(true)
        .with_evaluate_during_training(false);

    let env = ProbeEnvStateActionTest::default();
    let q = LinearDQNNet::<TrainingBacked>::init(
        &train_device,
        env.observation_space().shape(),
        env.action_space().shape(),
        1, // NOTE -> if using a hidden size of 1, can't use relu as activation, need to use sigmoid
    );
    let dqn_config = DQNConfig::new().with_update_every(10);

    let agent = DQNAgent::new(
        q.clone(),
        q,
        optim,
        dqn_config,
        env.action_space(),
        env.action_space(),
    );

    let buffer = ReplayBuffer::new(offline_params.memory_size);

    let logger = CsvLogger::new(
        PathBuf::from("logs/dqn_probe5/log_dqn_cartpole.csv"),
        false,
        true,
    );

    match logger.check_can_log(false) {
        Ok(_) => {}
        Err(err) => panic!("Error setting up logger: {err}"),
    }

    let mut trainer = OfflineTrainer::new(
        offline_params,
        Box::new(env),
        Box::new(ProbeEnvStateActionTest::default()),
        agent,
        buffer,
        Box::new(logger),
        None,
        EvalConfig::new().with_n_eval_episodes(10),
        &train_device,
    );

    trainer.train();
}
