use burn::{
    module::Module,
    nn::{
        loss::{MseLoss, Reduction},
        Initializer, Linear, LinearConfig,
    },
    optim::{adaptor::OptimizerAdaptor, Adam, AdamConfig, GradientsParams, Optimizer},
    tensor::{
        backend::{AutodiffBackend, Backend},
        ElementConversion, Shape, Tensor,
    },
};

use crate::common::{
    agent::{Agent, Policy},
    buffer::ReplayBuffer,
    logger::{LogData, LogItem},
    spaces::{BoxSpace, Space},
    to_tensor::{ToTensorB, ToTensorF},
};

use super::models::{PiModel, QModelSet};

#[derive(Debug, Module)]
pub struct EntCoefModule<B: Backend> {
    pub ent: Linear<B>,
}

impl<B: Backend> EntCoefModule<B> {
    pub fn new(starting_val: f32) -> Self {
        Self {
            ent: LinearConfig::new(1, 1)
                .with_bias(false)
                .with_initializer(Initializer::Constant {
                    value: starting_val as f64,
                })
                .init(&Default::default()),
        }
    }

    pub fn mul(&self, other: Tensor<B, 1>) -> Tensor<B, 1> {
        self.ent.forward(other)
    }

    pub fn val(&self) -> f32 {
        self.ent.weight.val().into_scalar().elem()
    }
}

enum EntCoef<B: AutodiffBackend> {
    Constant(f32),
    Trainable(
        EntCoefModule<B>,
        OptimizerAdaptor<Adam<B::InnerBackend>, EntCoefModule<B>, B>,
        f32,
    ),
}

impl<B: AutodiffBackend> EntCoef<B> {
    pub fn new(starting_val: f32, trainable: bool, target_entropy: f32) -> Self {
        if trainable {
            let module = EntCoefModule::new(starting_val);
            let optim = AdamConfig::new().init();
            EntCoef::Trainable(module, optim, target_entropy)
        } else {
            EntCoef::Constant(starting_val)
        }
    }

    pub fn train_step(
        &mut self,
        log_probs: Tensor<B, 1>,
        lr: f64,
        device: &B::Device,
    ) -> (f32, Option<f32>) {
        match self {
            EntCoef::Constant(val) => (*val, None),
            EntCoef::Trainable(module, optim, target_entropy) => {
                // (log) alpha = 0.0
                // => alpha = 1.0
                //
                // loss = alpha * (log prob + target entropy)
                // grad(loss) = d(loss)/d(alpha) = log prob + target entropy
                // new alpha = -learning rate * mean(grad(loss))
                //
                // Takeaways:
                // - it's fine if loss is 0, there should be gradients as long as (log prob + target entropy) != 0

                let temp_module = module.clone().to_device(device);
                println!("log_prbs: {log_probs}");

                let loss = -temp_module.mul(log_probs.add_scalar(*target_entropy).detach().mean());

                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &temp_module);

                println!("n registered grad params: {:?}", grads.len()); // return 0

                println!("ent coef before step: {:?}", temp_module.val());
                let temp_module = optim.step(lr, temp_module, grads);
                println!("ent coef after step: {:?}", temp_module.val()); // no change
                *module = temp_module;

                (module.val().exp(), Some(loss.into_scalar().elem()))
            }
        }
    }
}

pub struct SACAgent<B: AutodiffBackend> {
    // models
    pi: PiModel<B>,
    qs: QModelSet<B>,
    target_qs: QModelSet<B>,

    // optimisers
    pi_optim: OptimizerAdaptor<Adam<B::InnerBackend>, PiModel<B>, B>,
    q_optim: OptimizerAdaptor<Adam<B::InnerBackend>, QModelSet<B>, B>,

    // parameters
    ent_coef: EntCoef<B>,

    // housekeeping
    observation_space: Box<BoxSpace<Vec<f32>>>,
    action_space: Box<BoxSpace<Vec<f32>>>,
    last_update: usize,
    update_every: usize,
}

impl<B: AutodiffBackend> SACAgent<B> {
    pub fn new(
        // models
        pi: PiModel<B>,
        qs: QModelSet<B>,
        target_qs: QModelSet<B>,

        // optimisers
        pi_optim: OptimizerAdaptor<Adam<B::InnerBackend>, PiModel<B>, B>,
        q_optim: OptimizerAdaptor<Adam<B::InnerBackend>, QModelSet<B>, B>,

        // parameters
        ent_coef: Option<f32>,
        trainable_ent_coef: bool,
        target_entropy: Option<f32>,

        // housekeeping
        observation_space: Box<BoxSpace<Vec<f32>>>,
        action_space: Box<BoxSpace<Vec<f32>>>,
    ) -> Self {
        let target_entropy = match target_entropy {
            Some(val) => val,
            None => -(action_space.shape().len() as f32),
        };

        let ent_coef = match (ent_coef, trainable_ent_coef) {
            (None, false) => panic!("If not training ent_coef, an ent_coef must be supplied"),
            (None, true) => 0.01,
            (Some(val), false) => val,
            (Some(val), true) => {
                // Note: we optimize the log of the entropy coeff which is slightly different from the paper
                // as discussed in https://github.com/rail-berkeley/softlearning/issues/37
                val.ln()
            }
        };

        Self {
            pi,
            qs,
            target_qs,
            pi_optim,
            q_optim,
            ent_coef: EntCoef::new(ent_coef, trainable_ent_coef, target_entropy),
            observation_space,
            action_space,
            last_update: 0,
            update_every: 100,
        }
    }
}

impl<B: AutodiffBackend> Agent<B, Vec<f32>, Vec<f32>> for SACAgent<B> {
    fn act(
        &self,
        _global_step: usize,
        _global_frac: f32,
        obs: &Vec<f32>,
        _greedy: bool,
        inference_device: &<B>::Device,
    ) -> (Vec<f32>, LogItem) {
        // don't judge me
        let a: Vec<f32> = self
            .pi
            .act(&obs.clone().to_tensor(inference_device))
            .detach()
            .into_data()
            .value
            .into_iter()
            .map(|x| x.elem())
            .collect();

        (a, LogItem::default())
    }

    fn train_step(
        &mut self,
        global_step: usize,
        replay_buffer: &ReplayBuffer<Vec<f32>, Vec<f32>>,
        offline_params: &crate::common::algorithm::OfflineAlgParams,
        train_device: &<B as Backend>::Device,
    ) -> (Option<f32>, LogItem) {
        let log_dict = LogItem::default();

        let sample_data = replay_buffer.batch_sample(32);

        let states = sample_data.states.to_tensor(train_device);
        let actions = sample_data.actions.to_tensor(train_device);
        let next_states = sample_data.next_states.to_tensor(train_device);
        let rewards = sample_data.rewards.to_tensor(train_device).unsqueeze_dim(0);
        let terminated = sample_data
            .terminated
            .to_tensor(train_device)
            .unsqueeze_dim(0);
        let truncated = sample_data
            .truncated
            .to_tensor(train_device)
            .unsqueeze_dim(0);
        let dones = (terminated.float() + truncated.float()).bool();

        let (actions_pi, log_prob) = self.pi.act_log_prob(states.clone());

        // train entropy coeficient if required to do so
        let (ent_coef, ent_coef_loss) = self.ent_coef.train_step(
            log_prob.clone().flatten(0, 1),
            offline_params.lr,
            train_device,
        );

        let log_dict = log_dict.push("ent_coef".to_string(), LogData::Float(ent_coef));

        let log_dict = if let Some(l) = ent_coef_loss {
            log_dict.push("ent_coef_loss".to_string(), LogData::Float(l))
        } else {
            log_dict
        };

        // select action according to policy
        let (next_action_sampled, next_action_log_prob) = self.pi.act_log_prob(next_states.clone());
        // next_action_sampled
        let next_q_vals = self
            .target_qs
            .q_from_actions(next_states, next_action_sampled);
        let next_q_vals = Tensor::cat(next_q_vals, 1);
        let next_q_vals = next_q_vals.min_dim(1);
        // add the entropy term
        let next_q_vals = next_q_vals - next_action_log_prob.mul_scalar(ent_coef);
        // td error + entropy term
        let target_q_vals = rewards
            + dones
                .bool_not()
                .float()
                .mul(next_q_vals)
                .mul_scalar(offline_params.gamma);

        // calculate the critic loss
        let q_vals = self.qs.q_from_actions(states.clone(), actions.clone());

        let mut critic_loss: Tensor<B, 1> = Tensor::zeros(Shape::new([1]), train_device);
        for q in q_vals {
            critic_loss =
                critic_loss + MseLoss::new().forward(q, target_q_vals.clone(), Reduction::Mean);
        }
        critic_loss = critic_loss.mul_scalar(1.0 / (self.qs.len() as f32));

        let log_dict = log_dict.push(
            "critic_loss_combined".to_string(),
            LogData::Float(critic_loss.clone().into_scalar().elem()),
        );

        // optimise the critics
        let critic_loss_grads = critic_loss.clone().backward();
        let critic_grads = GradientsParams::from_grads(critic_loss_grads, &self.qs);
        self.qs = self
            .q_optim
            .step(offline_params.lr, self.qs.clone(), critic_grads);

        // Policy loss
        // recalculate q values with new critics
        let q_vals = self.qs.q_from_actions(states.clone(), actions_pi);
        let q_vals = Tensor::cat(q_vals, 1);
        let min_q = q_vals.min_dim(1);
        let actor_loss = log_prob.mul_scalar(ent_coef) - min_q;
        let actor_loss = actor_loss.mean();

        let log_dict = log_dict.push(
            "actor_loss".to_string(),
            LogData::Float(actor_loss.clone().into_scalar().elem()),
        );

        let actor_loss_back = actor_loss.backward();
        let actor_grads = GradientsParams::from_grads(actor_loss_back, &self.pi);
        self.pi = self
            .pi_optim
            .step(offline_params.lr, self.pi.clone(), actor_grads);

        // target critic updates
        if global_step > (self.last_update + self.update_every) {
            // hard update
            self.target_qs.update(&self.qs, None);

            self.last_update = global_step;
        }

        (None, log_dict)
    }

    fn observation_space(&self) -> Box<dyn Space<Vec<f32>>> {
        self.observation_space.clone()
    }

    fn action_space(&self) -> Box<dyn Space<Vec<f32>>> {
        self.action_space.clone()
    }
}
