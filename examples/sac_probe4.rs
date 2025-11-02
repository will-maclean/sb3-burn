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
    env::{base::Env, continuous_probe::ProbeEnvContinuousActions4},
    sac::{
        agent::{SACAgent, SACConfig},
        models::{PiModel, QModelSet},
    },
};

const N_CRITICS: usize = 2;

#[cfg(feature = "sb3-tch")]
use burn::backend::{libtorch::LibTorchDevice, LibTorch};
#[cfg(not(feature = "sb3-tch"))]
use burn::backend::{wgpu::WgpuDevice, Wgpu};

#[cfg(not(feature = "sb3-tch"))]
type B = Autodiff<Wgpu>;
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
    let train_device = WgpuDevice::default();

    sb3_seed::<B>(1234, &train_device);

    let env = ProbeEnvContinuousActions4::default();

    let config_optimizer =
        AdamConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(10.0)));

    let pi_optim = config_optimizer.init();

    let qs: QModelSet<B> = QModelSet::new(
        env.observation_space().shape().len(),
        env.action_space().shape().len(),
        64,
        &train_device,
        N_CRITICS,
    );

    let q_optim = config_optimizer.init();

    let pi = PiModel::new(
        env.observation_space().shape().len(),
        env.action_space().shape().len(),
        64,
        &train_device,
    );

    let offline_params = OfflineAlgParams::new()
        .with_batch_size(32)
        .with_memory_size(10000)
        .with_n_steps(1000)
        .with_warmup_steps(200)
        .with_lr(1e-3)
        .with_evaluate_every_steps(2000)
        .with_eval_at_start_of_training(true)
        .with_eval_at_end_of_training(true)
        .with_evaluate_during_training(false);

    let sac_config = SACConfig::new()
        .with_ent_lr(1e-4)
        .with_critic_tau(0.005)
        .with_update_every(1)
        .with_trainable_ent_coef(false)
        .with_target_entropy(None)
        .with_ent_coef(Some(0.5));

    let agent = SACAgent::new(
        sac_config,
        pi,
        qs.clone(),
        qs.clone(),
        pi_optim,
        q_optim,
        Box::new(BoxSpace::from(([-1.0].to_vec(), [1.0].to_vec()))),
        Box::new(BoxSpace::from(([-1.0].to_vec(), [1.0].to_vec()))),
    );

    let buffer = ReplayBuffer::new(offline_params.memory_size);

    let logger = CsvLogger::new(
        PathBuf::from("logs/sac_probe2/log_sac_probe2.csv"),
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
        Box::new(ProbeEnvContinuousActions4::default()),
        agent,
        buffer,
        Box::new(logger),
        None,
        EvalConfig::new()
            .with_n_eval_episodes(4)
            .with_print_obs(true)
            .with_print_action(true)
            .with_print_reward(true)
            .with_print_prediction(true),
        &train_device,
    );

    trainer.train();
}
