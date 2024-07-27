use std::path::PathBuf;

use burn::{backend::{libtorch::LibTorchDevice, Autodiff, LibTorch}, grad_clipping::GradientClippingConfig, optim::{Adam, AdamConfig}};
use sb3_burn::{common::{algorithm::{OfflineAlgParams, OfflineTrainer}, buffer::ReplayBuffer, eval::EvalConfig, logger::{CsvLogger, Logger}, spaces::BoxSpace}, env::{base::Env, classic_control::pendulum::{make_pendulum, PendulumEnv}}, simple_sac::agent::{PiModel, QModel, QModelSet, SACAgent}};

const N_CRITICS: usize = 3;

fn main() {
    // Using parameters from:
    // https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml

    type TrainingBacked = Autodiff<LibTorch>;

    let train_device = LibTorchDevice::Cuda(0);
    
    let env = make_pendulum(None);

    let config_optimizer =
        AdamConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(10.0)));

    let pi_optim = config_optimizer.init();
    
    let qs: QModelSet<TrainingBacked> =  QModelSet::new(
        env.observation_space().shape().len(),
        env.action_space().shape().len(),
        &train_device,
        N_CRITICS
    );

    let q_optim = config_optimizer.init();

    let pi = PiModel::new(
        env.observation_space().shape().len(),
        env.action_space().shape().len(),
        &train_device,
    );

    let offline_params = OfflineAlgParams::new()
        .with_batch_size(16)
        .with_memory_size(1000)
        .with_n_steps(1000)
        .with_warmup_steps(50)
        .with_lr(3e-4)
        .with_eval_at_end_of_training(true)
        .with_eval_at_end_of_training(true)
        .with_evaluate_during_training(false);


    let agent = SACAgent::new(
        pi,
        qs.clone(),
        qs,
        pi_optim,
        q_optim,
        1.0,
        Box::new(BoxSpace::from(([0.0].to_vec(), [0.0].to_vec()))),
        Box::new(BoxSpace::from(([0.0].to_vec(), [0.0].to_vec()))),
    );

    let buffer = ReplayBuffer::new(offline_params.memory_size);

    let logger = CsvLogger::new(PathBuf::from("logs/sac_pendulum/log_sac_pendulum.csv"), false);

    match logger.check_can_log(false) {
        Ok(_) => {}
        Err(err) => panic!("Error setting up logger: {err}"),
    }

    let mut trainer: OfflineTrainer<_, Adam<LibTorch>, _, _, _> = OfflineTrainer::new(
        offline_params,
        env,
        make_pendulum(None),
        agent,
        buffer,
        Box::new(logger),
        None,
        EvalConfig::new().with_n_eval_episodes(10),
        &train_device,
    );

    trainer.train();
}