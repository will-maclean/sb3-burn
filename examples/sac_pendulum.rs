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
        spaces::BoxSpace,
    },
    env::classic_control::pendulum::{make_pendulum, make_pendulum_eval},
    sac::{
        agent::SACAgent,
        models::{PiModel, QModelSet},
    },
};

const N_CRITICS: usize = 2;

fn main() {
    // Using parameters from:
    // https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml

    type TrainingBacked = Autodiff<LibTorch>;
    let train_device = LibTorchDevice::Cuda(0);

    let env = make_pendulum(None);

    let config_optimizer =
        AdamConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(10.0)));

    let pi_optim = config_optimizer.init();

    let qs: QModelSet<TrainingBacked> = QModelSet::new(
        env.observation_space().shape().len(),
        env.action_space().shape().len(),
        &train_device,
        N_CRITICS,
    );

    let q_optim = config_optimizer.init();

    let pi = PiModel::new(
        env.observation_space().shape().len(),
        env.action_space().shape().len(),
        &train_device,
    );

    let offline_params = OfflineAlgParams::new()
        .with_batch_size(256)
        .with_memory_size(100_000)
        .with_gamma(0.99)
        .with_n_steps(100_000)
        .with_warmup_steps(10000)
        .with_lr(3e-4)
        .with_profile_timers(true)
        .with_profile_log_every_steps(250)
        .with_eval_at_start_of_training(true)
        .with_eval_at_end_of_training(true)
        .with_evaluate_during_training(true)
        .with_evaluate_every_steps(2000);

    let agent = SACAgent::new(
        pi,
        qs.clone(),
        qs,
        pi_optim,
        q_optim,
        Some(0.1),
        true,
        None,
        Some(1e-3),
        Some(0.005),
        Box::new(BoxSpace::from((
            vec![-1.0, -1.0, -1.0],
            vec![1.0, 1.0, 1.0],
        ))),
        Box::new(BoxSpace::from(([-1.0].to_vec(), [1.0].to_vec()))),
    );

    let buffer = ReplayBuffer::new(offline_params.memory_size);

    let logger = CsvLogger::new(
        PathBuf::from("logs/sac_pendulum/log_sac_pendulum.csv"),
        false,
        true,
    );

    match logger.check_can_log(false) {
        Ok(_) => {}
        Err(err) => panic!("Error setting up logger: {err}"),
    }

    let mut trainer: OfflineTrainer<_, Adam, _, _, _> = OfflineTrainer::new(
        offline_params,
        env,
        make_pendulum_eval(None),
        agent,
        buffer,
        Box::new(logger),
        None,
        EvalConfig::new().with_n_eval_episodes(2),
        &train_device,
    );

    trainer.train();
}
