use std::path::PathBuf;

use burn::{backend::Autodiff, grad_clipping::GradientClippingConfig, optim::AdamConfig};
use sb3_burn::{
    common::{
        algorithm::{OfflineAlgParams, OfflineTrainer},
        buffer::ReplayBuffer,
        eval::EvalConfig,
        logger::{CsvLogger, Logger},
        spaces::BoxSpace,
        utils::sb3_seed,
    },
    env::classic_control::pendulum::{make_pendulum, make_pendulum_eval},
    sac::{
        agent::{SACAgent, SACConfig},
        models::{PiModel, QModelSet},
    },
};

const N_CRITICS: usize = 2;

#[cfg(feature = "sb3-tch")]
use burn::backend::{libtorch::LibTorchDevice, LibTorch};
#[cfg(not(feature = "sb3-tch"))]
use burn::backend::{ndarray::NdArrayDevice, NdArray};

#[cfg(not(feature = "sb3-tch"))]
type B = Autodiff<NdArray>;
#[cfg(feature = "sb3-tch")]
type B = Autodiff<LibTorch>;

extern crate sb3_burn;

fn main() {
    // Using parameters from:
    // https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml

    #[cfg(feature = "sb3-tch")]
    let train_device = if tch::utils::has_cuda() {
        LibTorchDevice::Cuda(0)
    } else {
        LibTorchDevice::Cpu
    };

    #[cfg(not(feature = "sb3-tch"))]
    let train_device = NdArrayDevice::default();

    sb3_seed::<B>(1234, &train_device);

    let env = make_pendulum(None);

    let config_optimizer =
        AdamConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(10.0)));

    let pi_optim = config_optimizer.init();

    let qs: QModelSet<B> = QModelSet::new(
        env.observation_space().shape().len(),
        env.action_space().shape().len(),
        32,
        &train_device,
        N_CRITICS,
    );

    let q_optim = config_optimizer.init();

    let pi = PiModel::new(
        env.observation_space().shape().len(),
        env.action_space().shape().len(),
        32,
        &train_device,
    );

    let offline_params = OfflineAlgParams::new()
        .with_batch_size(32)
        .with_memory_size(200_000)
        .with_gamma(0.99)
        .with_n_steps(50_000)
        .with_warmup_steps(10_000)
        .with_profile_timers(true)
        .with_profile_log_every_steps(250)
        .with_eval_at_start_of_training(true)
        .with_eval_at_end_of_training(true)
        .with_evaluate_during_training(true)
        .with_evaluate_every_steps(10_000);

    let sac_config = SACConfig::new()
        .with_ent_lr(3e-4)
        .with_pi_lr(3e-4)
        .with_q_lr(1e-3)
        .with_critic_tau(0.005)
        .with_update_every(1)
        .with_trainable_ent_coef(false)
        .with_target_entropy(None)
        .with_ent_coef(Some(0.2));

    let agent = SACAgent::new(
        sac_config,
        pi,
        qs.clone(),
        qs,
        pi_optim,
        q_optim,
        Box::new(BoxSpace::from((
            [-1.0, -1.0, -1.0].to_vec(),
            [1.0, 1.0, 1.0].to_vec(),
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

    let mut trainer = OfflineTrainer::new(
        offline_params,
        env,
        make_pendulum_eval(None),
        agent,
        buffer,
        Box::new(logger),
        None,
        EvalConfig::new().with_n_eval_episodes(10),
        &train_device,
    );

    trainer.train();
}
