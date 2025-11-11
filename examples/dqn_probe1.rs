use std::path::PathBuf;

use burn::{backend::Autodiff, grad_clipping::GradientClippingConfig, optim::AdamConfig};
use sb3_burn::{
    common::{
        algorithm::{OfflineAlgParams, OfflineTrainer},
        buffer::ReplayBuffer,
        eval::EvalConfig,
        logger::{CsvLogger, Logger},
        utils::sb3_seed,
    },
    dqn::{module::LinearDQNNet, DQNAgent, DQNConfig},
    env::{base::Env, probe::ProbeEnvValueTest},
};

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

    let config_optimizer =
        AdamConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(10.0)));
    let optim = config_optimizer.init();
    let offline_params = OfflineAlgParams::new()
        .with_batch_size(10)
        .with_memory_size(1000)
        .with_n_steps(1000)
        .with_warmup_steps(50)
        .with_eval_at_end_of_training(true)
        .with_eval_at_end_of_training(true)
        .with_evaluate_during_training(false);

    let env = ProbeEnvValueTest::default();
    let q: LinearDQNNet<B> = LinearDQNNet::init(
        &train_device,
        env.observation_space().shape().len(),
        env.action_space().shape(),
        1,
    );
    let dqn_config = DQNConfig::new().with_update_every(10);
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
        PathBuf::from("logs/dqn_probe1/log_dqn_cartpole.csv"),
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
        Box::new(ProbeEnvValueTest::default()),
        agent,
        buffer,
        Box::new(logger),
        None,
        EvalConfig::new().with_n_eval_episodes(10),
        &train_device,
    );

    trainer.train();
}
