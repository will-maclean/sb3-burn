use burn::{
    config::Config,
    module::Module,
    nn::{
        loss::{MseLoss, Reduction},
        Initializer, Linear, LinearConfig,
    },
    optim::{adaptor::OptimizerAdaptor, Adam, AdamConfig, GradientsParams, Optimizer},
    tensor::{
        backend::{AutodiffBackend, Backend},
        Bool, ElementConversion, Shape, Tensor,
    },
};

use crate::common::{
    agent::{Agent, Policy},
    buffer::ReplayBuffer,
    logger::{LogData, LogItem},
    spaces::{BoxSpace, Space},
    to_tensor::{ToTensorB, ToTensorF},
};

use crate::common::timer::Profiler;

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
        OptimizerAdaptor<Adam, EntCoefModule<B>, B>,
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

                // Standard SAC alpha training: minimize
                // J(alpha) = E[ alpha * (-log_pi - target_entropy) ]
                // Optimize w.r.t. log_alpha for positivity and scale-invariance.
                let log_probs = log_probs.detach().unsqueeze_dim(1);
                let log_alpha = temp_m.ent.weight.val(); // shape [1,1]

                // Use log_alpha directly in the loss, without exp().
                let loss = -(log_alpha * (log_probs.add_scalar(*target_entropy))).mean();

                let g: <B as AutodiffBackend>::Gradients = loss.backward();
                let grads = GradientsParams::from_grads(g, &temp_m);

                *m = optim.step(lr, temp_m, grads);

                (m.val().exp(), Some(loss.into_scalar().elem()))
            }
        }
    }
}

#[derive(Config)]
pub struct SACConfig {
    ent_coef: Option<f32>,
    #[config(default = 1)]
    update_every: usize,
    #[config(default = 1e-4)]
    ent_lr: f64,
    #[config(default = 0.005)]
    critic_tau: f32,
    target_entropy: Option<f32>,
    #[config(default = true)]
    trainable_ent_coef: bool,
}

pub struct SACAgent<B: AutodiffBackend> {
    config: SACConfig,

    // models
    pi: PiModel<B>,
    qs: QModelSet<B>,
    target_qs: QModelSet<B>,

    // optimisers
    pi_optim: OptimizerAdaptor<Adam, PiModel<B>, B>,
    q_optim: OptimizerAdaptor<Adam, QModelSet<B>, B>,

    // parameters
    ent_coef: EntCoef<B>,

    // housekeeping
    observation_space: Box<BoxSpace<Vec<f32>>>,
    action_space: Box<BoxSpace<Vec<f32>>>,
    last_update: usize,
    profiler: Profiler,
}

impl<B: AutodiffBackend> SACAgent<B> {
    pub fn new(
        mut config: SACConfig,

        // models
        pi: PiModel<B>,
        qs: QModelSet<B>,
        target_qs: QModelSet<B>,

        // optimisers
        pi_optim: OptimizerAdaptor<Adam, PiModel<B>, B>,
        q_optim: OptimizerAdaptor<Adam, QModelSet<B>, B>,

        // housekeeping
        observation_space: Box<BoxSpace<Vec<f32>>>,
        action_space: Box<BoxSpace<Vec<f32>>>,
    ) -> Self {
        // First, check the action space bounds are all [-1, 1]
        let a_high = action_space.high();
        let a_low = action_space.low();
        for i in 0..action_space.shape().len() {
            if a_high[i] != 1.0 || a_low[i] != -1.0 {
                panic!("ERROR: SAC only supports action bounds of [-1, 1]. You'll need to handle the scaling yourself");
            }
        }

        config.target_entropy = match config.target_entropy {
            Some(val) => Some(val),
            None => Some(-(action_space.shape().len() as f32)),
        };

        let ent_coef = match (config.ent_coef, config.trainable_ent_coef) {
            (None, false) => panic!("If not training ent_coef, an ent_coef must be supplied"),
            (None, true) => 0.0,
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
            ent_coef: EntCoef::new(
                ent_coef,
                config.trainable_ent_coef,
                config.target_entropy.unwrap(),
            ),
            observation_space,
            action_space,
            last_update: 0,
            profiler: Profiler::new(true),
            config,
        }
    }

    fn train_critics(
        &mut self,
        states: Tensor<B, 2>,
        actions: Tensor<B, 2>,
        next_states: Tensor<B, 2>,
        rewards: Tensor<B, 2>,
        dones: Tensor<B, 2, Bool>,
        gamma: f32,
        train_device: &B::Device,
        ent_coef: f32,
        lr: f64,
        log_dict: LogItem,
    ) -> LogItem {
        // select action according to policy
        let (next_action_sampled, next_action_log_prob) = self.pi.act_log_prob(next_states.clone());

        let next_action_log_prob = next_action_log_prob.sum_dim(1);
        // disp_tensorf("next_action_sampled", &next_action_sampled);
        // disp_tensorf("next_action_log_prob", &next_action_log_prob);

        // next_action_sampled
        let next_q_vals = self
            .target_qs
            .q_from_actions(next_states, next_action_sampled);

        let next_q_vals = Tensor::cat(next_q_vals, 1);
        // disp_tensorf("1next_q_vals", &next_q_vals);

        let next_q_vals = next_q_vals.min_dim(1);
        // disp_tensorf("2next_q_vals", &next_q_vals);

        // add the entropy term
        let next_q_vals = next_q_vals - next_action_log_prob.mul_scalar(ent_coef);
        // disp_tensorf("3next_q_vals", &next_q_vals);

        // td error + entropy term
        let target_q_vals = rewards + dones.bool_not().float().mul(next_q_vals).mul_scalar(gamma);

        // disp_tensorf("target_q_vals", &target_q_vals);

        let target_q_vals = target_q_vals.detach();

        // calculate the critic loss
        let q_vals: Vec<Tensor<B, 2>> = self.qs.q_from_actions(states, actions);

        let loss_fn = MseLoss::new();
        let mut critic_loss: Tensor<B, 1> = Tensor::zeros(Shape::new([1]), train_device);
        for q in q_vals {
            // disp_tensorf("q", &q);
            critic_loss = critic_loss + loss_fn.forward(q, target_q_vals.clone(), Reduction::Mean);
        }

        // disp_tensorf("critic_loss", &critic_loss);

        // Confirmed with sb3 community that the 0.5 scaling has nothing to do with the number
        // of critics - rather, it is just to remove the factor of 2 that would otherwise appear
        // in MSE gradient calculations. (Convention)
        critic_loss = critic_loss.mul_scalar(0.5);

        let log_dict = log_dict.push(
            "critic_loss_combined".to_string(),
            LogData::Float(critic_loss.clone().into_scalar().elem()),
        );

        // optimise the critics
        let critic_loss_grads = critic_loss.clone().backward();
        let critic_grads = GradientsParams::from_grads(critic_loss_grads, &self.qs);
        self.qs = self.q_optim.step(lr, self.qs.clone(), critic_grads);

        log_dict
    }

    fn train_policy(
        &mut self,
        states: Tensor<B, 2>,
        ent_coef: f32,
        lr: f64,
        actions_pi: Tensor<B, 2>,
        log_prob: Tensor<B, 2>,
        log_dict: LogItem,
    ) -> LogItem {
        // Policy loss
        // recalculate q values with new critics
        let q_vals = self.qs.q_from_actions(states, actions_pi);
        let q_vals = Tensor::cat(q_vals, 1);
        // disp_tensorf("q_vals", &q_vals);
        let min_q = q_vals.min_dim(1);
        // disp_tensorf("min_q", &min_q);
        let actor_loss = log_prob.mul_scalar(ent_coef) - min_q;
        // disp_tensorf("1actor_loss", &actor_loss);
        let actor_loss = actor_loss.mean();
        // disp_tensorf("2actor_loss", &actor_loss);

        let log_dict = log_dict.push(
            "actor_loss".to_string(),
            LogData::Float(actor_loss.clone().into_scalar().elem()),
        );

        let actor_loss_back = actor_loss.backward();
        let actor_grads = GradientsParams::from_grads(actor_loss_back, &self.pi);
        self.pi = self.pi_optim.step(lr, self.pi.clone(), actor_grads);

        log_dict
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
        outputs_in_log: bool,
    ) -> (Vec<f32>, LogItem) {
        // don't judge me
        let a_t = self
            .pi
            .act(&obs.clone().to_tensor(inference_device), greedy)
            .detach();
        let a = a_t.clone().into_data().to_vec().unwrap();

        let mut log_item = LogItem::default();

        if outputs_in_log {
            // get the q val for the q, a pair
            let q: f32 = Tensor::cat(
                self.qs.q_from_actions(
                    obs.clone().to_tensor(inference_device).unsqueeze(),
                    a_t.unsqueeze(),
                ),
                1,
            )
            .min_dim(1)
            .into_data()
            .to_vec()
            .unwrap()[0];

            log_item = log_item.push("q".to_string(), LogData::Float(q));
        }

        (a, log_item)
    }

    fn train_step(
        &mut self,
        global_step: usize,
        replay_buffer: &ReplayBuffer<Vec<f32>, Vec<f32>>,
        offline_params: &crate::common::algorithm::OfflineAlgParams,
        train_device: &<B as Backend>::Device,
    ) -> (Option<f32>, LogItem) {
        let log_dict = LogItem::default();
        let t_buffer_sample = std::time::Instant::now();
        let sample_data = self.profiler.time("sample", || {
            replay_buffer.batch_sample(offline_params.batch_size)
        });

        self.profiler
            .record("buffer_sample", t_buffer_sample.elapsed().as_secs_f64());

        let t_to_tensor0 = std::time::Instant::now();
        let states = sample_data.states.to_tensor(train_device);
        let actions = sample_data.actions.to_tensor(train_device);
        let next_states = sample_data.next_states.to_tensor(train_device);
        let rewards = sample_data.rewards.to_tensor(train_device).unsqueeze_dim(1);
        let terminated = sample_data
            .terminated
            .to_tensor(train_device)
            .unsqueeze_dim(1);
        let dones = terminated.clone();
        self.profiler
            .record("to_tensor", t_to_tensor0.elapsed().as_secs_f64());

        // disp_tensorf("states", &states);
        // disp_tensorf("actions", &actions);
        // disp_tensorf("next_states", &next_states);
        // disp_tensorf("rewards", &rewards);
        // disp_tensorb("dones", &dones);

        let t_policy0 = std::time::Instant::now();
        let (actions_pi, log_prob) = self.pi.act_log_prob(states.clone());

        let log_dict = log_dict.push(
            "action_saturation_frac".to_string(),
            LogData::Float(
                actions_pi
                    .clone()
                    .abs()
                    .greater_elem(0.99)
                    .float()
                    .mean()
                    .into_scalar()
                    .elem(),
            ),
        );

        let log_prob = log_prob.sum_dim(1);

        self.profiler
            .record("policy", t_policy0.elapsed().as_secs_f64());

        let log_dict = log_dict.push(
            "entropy_proxy".to_string(),
            LogData::Float(-log_prob.clone().mean().into_scalar().elem::<f32>()),
        );
        // disp_tensorf("actions_pi", &actions_pi);
        // disp_tensorf("log_prob", &log_prob);

        // train entropy coeficient if required to do so
        let t_ent0 = std::time::Instant::now();
        let (ent_coef, ent_coef_loss) = self.ent_coef.train_step(
            log_prob.clone().flatten(0, 1),
            self.config.ent_lr,
            train_device,
        );
        self.profiler
            .record("ent_coef", t_ent0.elapsed().as_secs_f64());

        // println!("ent_cof: {}\n", ent_coef);

        let log_dict = log_dict.push("ent_coef".to_string(), LogData::Float(ent_coef));

        let log_dict = if let Some(l) = ent_coef_loss {
            log_dict.push("ent_coef_loss".to_string(), LogData::Float(l))
        } else {
            log_dict
        };

        let t_critic0 = std::time::Instant::now();
        let log_dict = self.train_critics(
            states.clone(),
            actions,
            next_states,
            rewards,
            dones,
            offline_params.gamma,
            train_device,
            ent_coef,
            offline_params.lr,
            log_dict,
        );
        self.profiler
            .record("critic", t_critic0.elapsed().as_secs_f64());

        let t_actor0 = std::time::Instant::now();
        let log_dict = self.train_policy(
            states,
            ent_coef,
            offline_params.lr,
            actions_pi,
            log_prob,
            log_dict,
        );
        self.profiler
            .record("actor", t_actor0.elapsed().as_secs_f64());

        // target critic updates
        if global_step >= (self.last_update + self.config.update_every) {
            // hard update
            self.target_qs
                .update(&self.qs, Some(self.config.critic_tau));

            // Important! The way polyak updates are currently implemented
            // will override the setting for whether gradients are required
            self.target_qs = self.target_qs.clone().no_grad();

            self.last_update = global_step;
        }

        // panic!();

        (None, log_dict)
    }

    fn observation_space(&self) -> Box<dyn Space<Vec<f32>>> {
        self.observation_space.clone()
    }

    fn action_space(&self) -> Box<dyn Space<Vec<f32>>> {
        self.action_space.clone()
    }

    fn profile_flush(&mut self, step: usize, interval_steps: usize) -> Option<LogItem> {
        let item = self
            .profiler
            .into_logitem(step, interval_steps, Some("agent_"));
        self.profiler.reset();
        item
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
