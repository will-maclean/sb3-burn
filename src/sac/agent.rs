use std::time;

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
    utils::{disp_tensorb, disp_tensorf},
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

    pub fn mul(&self, other: Tensor<B, 2>) -> Tensor<B, 2> {
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
            EntCoef::Trainable(m, optim, target_entropy) => {
                // loss = alpha * (log prob + target entropy)
                // grad(loss) = d(loss)/d(alpha) = log prob + target entropy
                // new alpha = -learning rate * mean(grad(loss))
                //
                // Takeaways:
                // - it's fine if loss is 0, there should be gradients as long as (log prob + target entropy) != 0

                // println!("log_probs: {log_probs}");

                let temp_m = m.clone().fork(device);

                let log_probs = log_probs.detach().unsqueeze_dim(1);
                let loss = -temp_m.mul(log_probs.add_scalar(*target_entropy)).mean();

                // println!("loss shape: {:?}", loss.shape().dims);
                // println!("loss: {loss}");

                let g: <B as AutodiffBackend>::Gradients = loss.backward();
                let grads = GradientsParams::from_grads(g, &temp_m);

                *m = optim.step(lr, temp_m, grads);

                (m.val().exp(), Some(loss.into_scalar().elem()))
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
    critic_tau: Option<f32>,
    update_every: usize,

    // housekeeping
    observation_space: Box<BoxSpace<Vec<f32>>>,
    action_space: Box<BoxSpace<Vec<f32>>>,
    last_update: usize,
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
        critic_tau: Option<f32>,

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
            target_qs: target_qs.no_grad(),
            pi_optim,
            q_optim,
            ent_coef: EntCoef::new(ent_coef, trainable_ent_coef, target_entropy),
            critic_tau,
            observation_space,
            action_space,
            last_update: 0,
            update_every: 1,
        }
    }
}

impl<B: AutodiffBackend> Agent<B, Vec<f32>, Vec<f32>> for SACAgent<B> {
    fn act(
        &mut self,
        _global_step: usize,
        _global_frac: f32,
        obs: &Vec<f32>,
        greedy: bool,
        inference_device: &<B>::Device,
    ) -> (Vec<f32>, LogItem) {
        // don't judge me
        let a: Vec<f32> = self
            .pi
            .act(&obs.clone().to_tensor(inference_device), greedy)
            .detach()
            .into_data()
            .to_vec()
            .unwrap();

        (a, LogItem::default())
    }

    fn train_step(
        &mut self,
        global_step: usize,
        replay_buffer: &ReplayBuffer<Vec<f32>, Vec<f32>>,
        offline_params: &crate::common::algorithm::OfflineAlgParams,
        train_device: &<B as Backend>::Device,
    ) -> (Option<f32>, LogItem) {
        let print_instant = false; //(global_step > 500) & ( global_step % 1000 == 0);

        let log_dict = LogItem::default();

        let instant_timer = time::Instant::now();
        let sample_data = replay_buffer.batch_sample(offline_params.batch_size);
        if print_instant {
            let duration = (time::Instant::now() - instant_timer).as_micros();

            println!("i={global_step}. buffer sample duration: {}", duration);
        }

        let states = sample_data.states.to_tensor(train_device);
        let actions = sample_data.actions.to_tensor(train_device);
        let next_states = sample_data.next_states.to_tensor(train_device);
        let rewards = sample_data.rewards.to_tensor(train_device).unsqueeze_dim(1);
        let terminated = sample_data
            .terminated
            .to_tensor(train_device)
            .unsqueeze_dim(1);
        let truncated = sample_data
            .truncated
            .to_tensor(train_device)
            .unsqueeze_dim(1);
        let dones = (terminated.float() + truncated.float()).bool();

        disp_tensorf("states", &states);
        disp_tensorf("actions", &actions);
        disp_tensorf("next_states", &next_states);
        disp_tensorf("rewards", &rewards);
        disp_tensorb("dones", &dones);

        let instant_timer = time::Instant::now();
        let (actions_pi, log_prob) = self.pi.act_log_prob(states.clone());
        if print_instant {
            let duration = (time::Instant::now() - instant_timer).as_micros();

            println!("i={global_step}. actions_pi duration: {}", duration);
        }

        disp_tensorf("actions_pi", &actions_pi);
        disp_tensorf("log_prob", &log_prob);

        // train entropy coeficient if required to do so
        let instant_timer = time::Instant::now();
        let (ent_coef, ent_coef_loss) = self.ent_coef.train_step(
            log_prob.clone().flatten(0, 1),
            offline_params.lr,
            train_device,
        );
        if print_instant {
            let duration = (time::Instant::now() - instant_timer).as_micros();

            println!("i={global_step}. ent_coef duration: {}", duration);
        }

        let log_dict = log_dict.push("ent_coef".to_string(), LogData::Float(ent_coef));

        let log_dict = if let Some(l) = ent_coef_loss {
            log_dict.push("ent_coef_loss".to_string(), LogData::Float(l))
        } else {
            log_dict
        };

        // select action according to policy
        let instant_timer = time::Instant::now();

        let (next_action_sampled, next_action_log_prob) = self.pi.act_log_prob(next_states.clone());
        if print_instant {
            let duration = (time::Instant::now() - instant_timer).as_micros();

            println!(
                "i={global_step}. next_action_sampled duration: {}",
                duration
            );
        }
        disp_tensorf("next_action_sampled", &next_action_sampled);
        disp_tensorf("next_action_log_prob", &next_action_log_prob);

        // next_action_sampled
        let instant_timer = time::Instant::now();
        let next_q_vals = self
            .target_qs
            .q_from_actions(next_states, next_action_sampled);
        if print_instant {
            let duration = (time::Instant::now() - instant_timer).as_micros();

            println!("i={global_step}. next_q_vals duration: {}", duration);
        }

        let instant_timer = time::Instant::now();
        let next_q_vals = Tensor::cat(next_q_vals, 1);
        disp_tensorf("1next_q_vals", &next_q_vals);

        let next_q_vals = next_q_vals.min_dim(1);
        disp_tensorf("2next_q_vals", &next_q_vals);

        // add the entropy term
        let next_q_vals = next_q_vals - next_action_log_prob.mul_scalar(ent_coef);
        disp_tensorf("3next_q_vals", &next_q_vals);

        // td error + entropy term
        let target_q_vals = rewards
            + dones
                .bool_not()
                .float()
                .mul(next_q_vals)
                .mul_scalar(offline_params.gamma);

        disp_tensorf("target_q_vals", &target_q_vals);

        let target_q_vals = target_q_vals.detach();

        if print_instant {
            let duration = (time::Instant::now() - instant_timer).as_micros();

            println!("i={global_step}. target_q_vals duration: {}", duration);
        }

        // calculate the critic loss
        let instant_timer = time::Instant::now();
        let q_vals = self.qs.q_from_actions(states.clone(), actions.clone());
        if print_instant {
            let duration = (time::Instant::now() - instant_timer).as_micros();

            println!("i={global_step}. q_vals duration: {}", duration);
        }

        let instant_timer = time::Instant::now();

        let mut critic_loss: Tensor<B, 1> = Tensor::zeros(Shape::new([1]), train_device);
        for q in q_vals {
            critic_loss =
                critic_loss + MseLoss::new().forward(q, target_q_vals.clone(), Reduction::Mean);
        }
        disp_tensorf("critic_loss", &critic_loss);

        // Confirmed with sb3 community that the
        // 0.5 scaling has nothing to do with the number
        // of critics - rather, it is just to remove
        // the factor of 2 that would otherwise appear
        // in MSE gradient calculations. (Convention)
        critic_loss = critic_loss.mul_scalar(0.5);

        if print_instant {
            let duration = (time::Instant::now() - instant_timer).as_micros();

            println!("i={global_step}. critic_loss duration: {}", duration);
        }

        let log_dict = log_dict.push(
            "critic_loss_combined".to_string(),
            LogData::Float(critic_loss.clone().into_scalar().elem()),
        );

        // optimise the critics
        let instant_timer = time::Instant::now();
        let critic_loss_grads = critic_loss.clone().backward();
        let critic_grads = GradientsParams::from_grads(critic_loss_grads, &self.qs);
        self.qs = self
            .q_optim
            .step(offline_params.lr, self.qs.clone(), critic_grads);

        if print_instant {
            let duration = (time::Instant::now() - instant_timer).as_micros();

            println!(
                "i={global_step}. critic_loss backprop duration: {}",
                duration
            );
        }

        // Policy loss
        // recalculate q values with new critics
        let q_vals = self.qs.q_from_actions(states.clone(), actions_pi);
        let q_vals = Tensor::cat(q_vals, 1).detach();
        disp_tensorf("q_vals", &q_vals);
        let min_q = q_vals.min_dim(1);
        disp_tensorf("min_q", &min_q);
        let actor_loss = log_prob.mul_scalar(ent_coef) - min_q;
        disp_tensorf("1actor_loss", &actor_loss);
        let actor_loss = actor_loss.mean();
        disp_tensorf("2actor_loss", &actor_loss);

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
            self.target_qs.update(&self.qs, self.critic_tau);

            // Important! The way polyak updates are currently implemented
            // will override the setting for whether gradients are required
            self.target_qs = self.target_qs.clone().no_grad();

            self.last_update = global_step;
        }

        // panic!("");

        (None, log_dict)
    }

    fn observation_space(&self) -> Box<dyn Space<Vec<f32>>> {
        self.observation_space.clone()
    }

    fn action_space(&self) -> Box<dyn Space<Vec<f32>>> {
        self.action_space.clone()
    }
}

#[cfg(test)]
mod test {

    use burn::{
        backend::{Autodiff, NdArray},
        optim::{AdamConfig, GradientsParams, Optimizer},
        tensor::Tensor,
    };

    use crate::sac::agent::EntCoef;

    use super::EntCoefModule;

    #[test]
    fn test_ent_coef_module() {
        type Backend = Autodiff<NdArray>;

        let target_entropy = -1;
        let lr = 0.001;
        let mut optim = AdamConfig::new().init();

        let model: EntCoefModule<Backend> = EntCoefModule::new(0.0);

        let log_probs: Tensor<Backend, 1> =
            Tensor::from_floats([1.0, 2.0, 3.0, -1.0], &Default::default());

        let log_probs = log_probs.detach().unsqueeze_dim(1);
        let loss = -model.mul(log_probs.add_scalar(target_entropy)).mean();

        let g = loss.backward();
        let grads = GradientsParams::from_grads(g, &model);

        assert_eq!(grads.len(), 1);

        optim.step(lr, model, grads);
    }

    #[test]
    fn test_ent_coef() {
        type Backend = Autodiff<NdArray>;

        let target_entropy = -1.0;
        let lr = 0.001;
        let starting_ent = 0.0;
        let log_probs: Tensor<Backend, 1> =
            Tensor::from_floats([1.0, 2.0, 3.0, -1.0], &Default::default());

        let mut ent: EntCoef<Backend> = EntCoef::new(starting_ent, true, target_entropy);

        let ent_before = match &ent {
            EntCoef::Constant(_) => panic!("shouldn't be here"),
            EntCoef::Trainable(m, _, _) => m.val(),
        };

        ent.train_step(log_probs, lr, &Default::default());

        let ent_after = match &ent {
            EntCoef::Constant(_) => panic!("shouldn't be here"),
            EntCoef::Trainable(m, _, _) => m.val(),
        };

        assert!(ent_before != ent_after);
    }
}
