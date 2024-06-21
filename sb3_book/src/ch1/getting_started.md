# Getting started
The `examples` directory shows a few different ways to get started. The simplest
way is to use a pre-defined algorithm (like `DQN`) with a pre-defined network
(like `LinearDQNNet`) on a pre-defined environment (like `MountainCarEnv`). We
can use the below example:

```rust 
// Define all our imports
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
    env::{base::Env, classic_control::cartpole::CartpoleEnv},
};
//!
extern crate sb3_burn;
//!
fn main() {
    // Using parameters from:
    // https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml
//!
    // define the training backend
    type TrainingBacked = Autodiff<LibTorch>;
    
    // we can specify different devices for different parts of the training process
    // e.g. different devices for training the models and storing the replay buffer
    let train_device = LibTorchDevice::Cuda(0);
    let buffer_device = LibTorchDevice::Cpu;
//!
    // define the optimizer with any parameters
    // Here, we define some gradient clipping
    let config_optimizer =
        AdamConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(10.0)));
    let optim = config_optimizer.init();

    // OfflineAlgParams stores general (algorithm agnostic) training parameters.
    let offline_params = OfflineAlgParams::new()
        .with_batch_size(128)
        .with_memory_size(10000)
        .with_n_steps(120000)
        .with_warmup_steps(1000)
        .with_lr(4e-3)
        .with_gamma(0.98)
        .with_eval_at_end_of_training(true)
        .with_eval_at_end_of_training(true)
        .with_evaluate_during_training(false)
        .with_grad_steps(8)
        .with_train_every(16);
//!
    // define the environment
    let env = MountainCarEnv::default();

    // define the q network
    let q = LinearAdvDQNNet::<TrainingBacked>::init(
//1         &train_device,
        env.observation_space().shape().len(),
        env.action_space().shape(),
        256,    // size of the hidden layers in the model
    );

    // alg-specific parameters
    let dqn_config = DQNConfig::new()
        .with_update_every(600)
        .with_eps_end(0.07)
        .with_eps_end_frac(0.8);

    // we can then create the agent from the networks
    let agent = DQNAgent::new(
        q.clone(),  // Q learning uses 2 networks, so we just clone our model for q1...
        q,          // ... and this is q2
        optim,
        dqn_config,
        env.observation_space(),
        env.action_space(),
    );
//!
    // Define the replay buffer. Note that we don't need to specify the 
    // type of the observation / action here, as it can be inferred later.
    let buffer = ReplayBuffer::new(offline_params.memory_size);
//!
    // create the logger. Make sure the logging directory is 
    // created and empty
    let logger = CsvLogger::new(
        PathBuf::from("logs/dqn_logging/log_dqn_cartpole.csv"),
        false,
        Some("global_step".to_string()),
    );
    
    // It's a good idea to check that the logger is able to function
    // and to error out if not - it can be frustrating to wait for a
    // training run to happen and then realise none of the data is
    // being logged!
    match logger.check_can_log(false) {
        Ok(_) => {}
        Err(err) => panic!("Error setting up logger: {err}"),
    }
//!
    // We can then create the trainer itself...
    let mut trainer: OfflineTrainer<_, Adam<LibTorch>, _, _, _> =
        OfflineTrainer::new(
            offline_params,
            Box::new(env),
            Box::new(MountainCarEnv::default()),
            agent,
            buffer,
            Box::new(logger),
            None,
            EvalConfig::new().with_n_eval_episodes(10),
            &train_device,
            &buffer_device,
        );
//!
    // ... and run the training
    trainer.train();
}
```
