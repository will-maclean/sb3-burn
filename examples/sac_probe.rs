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
        spaces::BoxSpace,
    },
    env::{base::Env, probe::ProbeEnvContinuousActions},
    sac::{
        agent::{SACAgent, SACConfig},
        models::{PiModel, QModelSet},
    },
};

const N_CRITICS: usize = 2;

fn main() {
    // Using parameters from:
    // https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml

    type TrainingBacked = Autodiff<LibTorch>;

    let train_device = LibTorchDevice::default();

    let env = ProbeEnvContinuousActions::default();

    let config_optimizer =
        AdamConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(10.0)));

    let pi_optim = config_optimizer.init();

    let qs: QModelSet<TrainingBacked> = QModelSet::new(
        env.observation_space().shape().len(),
        env.action_space().shape().len(),
        4,
        &train_device,
        N_CRITICS,
    );

    let q_optim = config_optimizer.init();

    let pi = PiModel::new(
        env.observation_space().shape().len(),
        env.action_space().shape().len(),
        4,
        &train_device,
    );

    let offline_params = OfflineAlgParams::new()
        .with_batch_size(32)
        .with_memory_size(10000)
        .with_n_steps(10000)
        .with_warmup_steps(256)
        .with_lr(3e-3)
        .with_evaluate_every_steps(2000)
        .with_eval_at_start_of_training(true)
        .with_eval_at_end_of_training(true)
        .with_evaluate_during_training(true);

    let sac_config = SACConfig::new()
        .with_ent_lr(1e-4)
        .with_critic_tau(0.005)
        .with_update_every(1)
        .with_trainable_ent_coef(true)
        .with_target_entropy(None)
        .with_ent_coef(None);

    let agent = SACAgent::new(
        sac_config,
        pi,
        qs.clone(),
        qs,
        pi_optim,
        q_optim,
        Box::new(BoxSpace::from(([0.0].to_vec(), [1.0].to_vec()))),
        Box::new(BoxSpace::from(([0.0].to_vec(), [1.0].to_vec()))),
    );

    let buffer = ReplayBuffer::new(offline_params.memory_size);

    let logger = CsvLogger::new(
        PathBuf::from("logs/sac_probe/log_sac_probe.csv"),
        false,
        true,
    );

    match logger.check_can_log(false) {
        Ok(_) => {}
        Err(err) => panic!("Error setting up logger: {err}"),
    }

    let mut trainer: OfflineTrainer<_,  _, _, _> = OfflineTrainer::new(
        offline_params,
        Box::new(env),
        Box::new(ProbeEnvContinuousActions::default()),
        agent,
        buffer,
        Box::new(logger),
        None,
        EvalConfig::new().with_n_eval_episodes(4).with_print_obs(true).with_print_action(true).with_print_reward(true).with_print_prediction(true),
        &train_device,
    );

    trainer.train();
}
