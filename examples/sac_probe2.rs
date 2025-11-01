use std::path::PathBuf;

use burn::{
    backend::{libtorch::LibTorchDevice, Autodiff, LibTorch},
    grad_clipping::GradientClippingConfig,
    module::Module,
    optim::AdamConfig,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{ElementConversion, Tensor},
};
use sb3_burn::{
    common::{
        algorithm::{OfflineAlgParams, OfflineTrainer},
        buffer::ReplayBuffer,
        eval::EvalConfig,
        logger::{CsvLogger, Logger},
        spaces::BoxSpace,
    },
    env::{base::Env, continuous_probe::ProbeEnvContinuousActions2},
    sac::{
        agent::{SACAgent, SACConfig},
        models::{PiModel, QModelSet},
    },
};

const N_CRITICS: usize = 1;

fn main() {
    // Using parameters from:
    // https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml

    type TrainingBacked = Autodiff<LibTorch>;

    let train_device = LibTorchDevice::Cuda(0);

    let env = ProbeEnvContinuousActions2::default();

    let config_optimizer =
        AdamConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(10.0)));

    let pi_optim = config_optimizer.init();

    let qs: QModelSet<TrainingBacked> = QModelSet::new(
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
        .with_n_steps(5000)
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
        .with_ent_coef(Some(0.05));

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

    let mut trainer: OfflineTrainer<_, _, _, _> = OfflineTrainer::new(
        offline_params,
        Box::new(env),
        Box::new(ProbeEnvContinuousActions2::default()),
        agent,
        buffer,
        Box::new(logger),
        None,
        EvalConfig::new()
            .with_n_eval_episodes(20)
            .with_print_obs(true)
            .with_print_action(true)
            .with_print_reward(true)
            .with_print_prediction(true),
        &train_device,
    );

    let save_dir = PathBuf::from("weights/sac_probe2/");

    trainer.train();
    trainer.save(&save_dir);

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let trained_qs = qs
        .load_file(save_dir.join("qs_model"), &recorder, &train_device)
        .unwrap();

    let fresh_obs: Tensor<TrainingBacked, 2> =
        Tensor::<TrainingBacked, 1>::from_floats([0.5], &train_device)
            .unsqueeze()
            .require_grad();
    let fresh_action: Tensor<TrainingBacked, 2> =
        Tensor::<TrainingBacked, 1>::from_floats([0.0], &train_device)
            .unsqueeze()
            .require_grad();

    let q_vals = trained_qs.q_from_actions(fresh_obs, fresh_action.clone());
    let q_min = Tensor::cat(q_vals, 1).min_dim(1);

    // calculate the grads of q_min w.r.t. fresh_action
    let grads = q_min.clone().mean().backward();
    if let Some(grad) = fresh_action.grad(&grads) {
        let abs = grad.abs();
        println!(
            "∂Q/∂a mean/max: {:?}",
            (
                abs.clone().mean().into_scalar().elem::<f32>(),
                abs.max().into_scalar().elem::<f32>()
            )
        );
    } else {
        println!("fresh_action gradient not retained");
    }
}
