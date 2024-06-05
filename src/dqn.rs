use burn::{
    config::Config,
    module::Module,
    nn::{
        self,
        loss::{MseLoss, Reduction},
    },
    optim::{adaptor::OptimizerAdaptor, GradientsParams, Optimizer, SimpleOptimizer},
    tensor::{
        backend::{AutodiffBackend, Backend},
        ElementConversion, Int, Tensor,
    },
};

use crate::{
    algorithm::{OfflineAlgParams, OfflineTrainer},
    buffer::ReplayBuffer,
    policy::Policy,
    spaces::{Action, ActionSpace, Obs, ObsSpace, ObsT},
    utils::linear_decay,
};

#[derive(Config)]
pub struct DQNConfig {
    #[config(default = 1.0)]
    eps_start: f32,
    #[config(default = 0.05)]
    eps_end: f32,
    #[config(default = 0.1)]
    eps_end_frac: f32,
}

#[derive(Module, Debug)]
pub struct DQNNet<B: Backend> {
    l1: nn::Linear<B>,
    l2: nn::Linear<B>,
}

impl<B: Backend> DQNNet<B> {
    pub fn init(
        device: &B::Device,
        observation_space: ObsSpace,
        action_space: ActionSpace,
        hidden_size: usize,
    ) -> Self {
        match action_space {
            ActionSpace::Continuous { lows: _, highs: _ } => {
                panic!("Continuous actions are not supported by DQN")
            }
            ActionSpace::Discrete { size: action_size } => {
                let input_size = observation_space.size();

                Self {
                    l1: nn::LinearConfig::new(input_size, hidden_size).init(device),
                    l2: nn::LinearConfig::new(hidden_size, action_size).init(device),
                }
            }
        }
    }

    pub fn forward(&self, state: ObsT<B, 2>) -> Tensor<B, 2> {
        let x = self.l1.forward(state);
        self.l2.forward(x)
    }
}

impl<B: Backend> Policy<B> for DQNNet<B> {
    fn act(&self, state: &Obs, action_space: ActionSpace) -> Action {
        let state_tensor = state.clone().to_tensor().unsqueeze_dim(0);
        let q_vals = self.predict(state_tensor);
        let a: i32 = q_vals.squeeze::<1>(0).argmax(0).into_scalar().elem();

        Action::Discrete {
            space: action_space,
            idx: a,
        }
    }

    fn predict(&self, state: ObsT<B, 2>) -> Tensor<B, 2> {
        self.forward(state)
    }
}

pub fn dqn_act<O: SimpleOptimizer<B::InnerBackend>, B: AutodiffBackend>(
    q: &DQNNet<B>,
    step: usize,
    config: &DQNConfig,
    state: &Obs,
    trainer: &OfflineTrainer<O, B>,
) -> Action {
    {
        if rand::random::<f32>()
            > linear_decay(
                step as f32 / trainer.offline_params.n_steps as f32,
                config.eps_start,
                config.eps_end,
                config.eps_end_frac,
            )
        {
            q.act(state, trainer.env.action_space().clone())
        } else {
            trainer.env.action_space().sample()
        }
    }
}

pub fn dqn_train_step<O: SimpleOptimizer<B::InnerBackend>, B: AutodiffBackend>(
    q: & mut DQNNet<B>,
    optim: & mut OptimizerAdaptor<O, DQNNet<B>, B>,
    replay_buffer: &ReplayBuffer<B>,
    offline_params: &OfflineAlgParams,
    device: &B::Device,
) -> Option<f32> {
    // sample from the replay buffer
    let batch_sample = replay_buffer.batch_sample(offline_params.batch_size);

    match batch_sample {
        Some((s, a, s_, r, d)) => {
            let q_vals_ungathered = q.forward(s);
            let q_vals = q_vals_ungathered.gather(1, a.int());
            let next_q_vals_ungathered = q.forward(s_);
            let next_q_vals = next_q_vals_ungathered.max_dim(1);
            let targets = r
                + (Tensor::<B, 2, Int>::ones([offline_params.batch_size, 1], device) - d).float()
                    * next_q_vals
                    * offline_params.gamma;

            let loss = MseLoss::new().forward(q_vals, targets, Reduction::Mean);

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, q);

            *q = optim.step(offline_params.lr, q.clone(), grads);

            Some(loss.into_scalar().elem())
        }
        None => None,
    }
}
