use std::path::Path;

use burn::{
    config::Config,
    module::{Module, Param},
    nn::loss::{MseLoss, Reduction},
    optim::{adaptor::OptimizerAdaptor, Adam, AdamConfig, GradientsParams, Optimizer},
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
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
    utils::ModuleParamSummary,
};

use crate::common::timer::Profiler;

use super::models::{PiModel, QModelSet};

#[derive(Debug, Module)]
pub struct LogAlphaModule<B: Backend> {
    log_alpha: Param<Tensor<B, 1>>,
}

impl<B: Backend> LogAlphaModule<B> {
    pub fn new(starting_log_alpha: f32, device: &B::Device) -> Self {
        Self {
            log_alpha: Param::from_tensor(Tensor::from_floats([starting_log_alpha], device)),
        }
    }

    pub fn alpha(&self) -> f32 {
        self.log_alpha.val().into_scalar().elem::<f32>().exp()
    }

    pub fn calc_loss(&self, log_probs: Tensor<B, 2>, target_entropy: f32) -> Tensor<B, 1> {
        let target = log_probs
            .sum_dim(1)
            .flatten::<1>(0, 1)
            .add_scalar(target_entropy)
            .detach();

        (self.log_alpha.val() * (-target)).mean()
    }
}

enum EntCoef<B: AutodiffBackend> {
    Constant(f32),
    Trainable(
        LogAlphaModule<B>,
        OptimizerAdaptor<Adam, LogAlphaModule<B>, B>,
        f32,
    ),
}

impl<B: AutodiffBackend> EntCoef<B> {
    pub fn new(
        starting_val: f32,
        trainable: bool,
        target_entropy: f32,
        train_device: Option<&B::Device>,
    ) -> Self {
        if trainable {
            let module = LogAlphaModule::new(starting_val.ln(), train_device.unwrap());
            let optim = AdamConfig::new().init();
            EntCoef::Trainable(module, optim, target_entropy)
        } else {
            EntCoef::Constant(starting_val)
        }
    }

    pub fn train_step(
        &mut self,
        log_probs: Tensor<B, 2>,
        lr: f64,
        device: &B::Device,
    ) -> (f32, Option<f32>) {
        match self {
            EntCoef::Constant(val) => (*val, None),
            EntCoef::Trainable(m, optim, target_entropy) => {
                let temp_m = m.clone().fork(device);

                let alpha = temp_m.alpha();

                let loss = temp_m.calc_loss(log_probs, *target_entropy);
                let g: <B as AutodiffBackend>::Gradients = loss.backward();
                let grads = GradientsParams::from_grads(g, &temp_m);

                *m = optim.step(lr, temp_m, grads);

                (alpha, Some(loss.into_scalar().elem()))
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
            (None, true) => 1.0,
            (Some(val), false) => val,
            (Some(val), true) => val,
        };

        Self {
            pi: pi.clone(),
            qs,
            target_qs: target_qs.no_grad(),
            pi_optim,
            q_optim,
            ent_coef: EntCoef::new(
                ent_coef,
                config.trainable_ent_coef,
                config.target_entropy.unwrap(),
                Some(&pi.devices()[0]),
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
        // let actor_loss = -min_q;
        // disp_tensorf("1actor_loss", &actor_loss);
        let actor_loss = actor_loss.mean();
        // disp_tensorf("2actor_loss", &actor_loss);

        let log_dict = log_dict.push(
            "actor_loss".to_string(),
            LogData::Float(actor_loss.clone().into_scalar().elem()),
        );

        let actor_loss_back = actor_loss.backward();
        let actor_grads = GradientsParams::from_grads(actor_loss_back, &self.pi);

        // do some checks to see that pi is actually updating
        // let mut pre_pi_summary = ModuleParamSummary::default();
        // self.pi.visit(&mut pre_pi_summary);

        self.pi = self.pi_optim.step(lr, self.pi.clone(), actor_grads);
        // let mut post_pi_summary = ModuleParamSummary::default();
        // self.pi.visit(&mut post_pi_summary);

        // println!("Pi Summary pre-step");
        // pre_pi_summary.print();
        // println!("Pi Summary post-step");
        // post_pi_summary.print();
        //
        // panic!();

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
        let dones = terminated;
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
        let (ent_coef, ent_coef_loss) =
            self.ent_coef
                .train_step(log_prob.clone(), self.config.ent_lr, train_device);
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

    fn save(&self, path: &Path) {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let _ = self.pi.clone().save_file(path.join("pi_model"), &recorder);
        let _ = self.qs.clone().save_file(path.join("qs_model"), &recorder);
    }
}

#[cfg(test)]
mod test {

    use burn::{
        backend::{Autodiff, NdArray},
        optim::{AdamConfig, GradientsParams, Optimizer, SgdConfig},
        tensor::Tensor,
    };

    use crate::sac::agent::EntCoef;

    use super::LogAlphaModule;

    #[test]
    fn test_ent_coef_module() {
        type Ba = Autodiff<NdArray>;
        let device: <Autodiff<NdArray> as burn::prelude::Backend>::Device = Default::default();

        let target_entropy = -1.0;
        let lr = 0.001;
        let mut optim = AdamConfig::new().init();

        let model: LogAlphaModule<Ba> = LogAlphaModule::new(0.0, &device);

        let log_probs: Tensor<Ba, 1> =
            Tensor::from_floats([1.0, 2.0, 3.0, -1.0], &Default::default());

        let log_probs = log_probs.detach().unsqueeze_dim(1);
        let loss = model.calc_loss(log_probs, target_entropy);

        let g = loss.backward();
        let grads = GradientsParams::from_grads(g, &model);

        assert_eq!(grads.len(), 1);

        optim.step(lr, model, grads);
    }

    #[test]
    fn test_log_alpha_module() {
        type Backend = Autodiff<NdArray>;
        let device: <Autodiff<NdArray> as burn::prelude::Backend>::Device = Default::default();

        let target_entropy = -1.0;
        let lr = 1.0;
        let starting_alpha: f32 = 1.0;
        let log_probs: Tensor<Backend, 2> =
            Tensor::<Backend, 1>::from_floats([-2.0], &Default::default()).unsqueeze();

        let log_alpha_module: LogAlphaModule<Backend> =
            LogAlphaModule::new(starting_alpha.ln(), &device);

        let starting_alpha = log_alpha_module.alpha();
        let loss = log_alpha_module.calc_loss(log_probs, target_entropy);

        let mut sgd = SgdConfig::new().init();

        let g = loss.backward();
        let grads = GradientsParams::from_grads(g, &log_alpha_module);

        let log_alpha_after = sgd.step(lr, log_alpha_module, grads);
        let alpha_after = log_alpha_after.alpha();

        // note - we train log alpha = beta = log(alpha)
        // alpha0 = 0.0 => beta0 = log(alpha0) = 0

        // Loss func is Loss(beta) = beta * (-log_probs.sum_dim(1) - target_entropy).mean()
        // => Grad(Loss(beta)) = (-log_probs.sum_dim(1) - target_entropy).mean()
        //                     = (-[-2, -2] - (-1)).mean()
        //                     = ([2, 2] + 1).mean()
        //                     = ([3, 3]).mean()
        //                     = 3
        // so:
        // beta1 = beta0 - lr * Grad(Loss(beta)) = 0 - 1.0 * 3 = -3
        //
        // Also, we expect the loss to be:
        //  Loss(beta) = beta * (-log_probs.sum_dim(1) - target_entropy).mean() = (0) * (-3) = 0
        //
        //  Note: alpha updates are done with adam, so with its momentum and other params we might
        //  expect the actual update size to be smaller. but, we can expect it to move in the
        //  correct direction.
        assert_approx_eq::assert_approx_eq!(starting_alpha, 1.0);
        assert_approx_eq::assert_approx_eq!(alpha_after, (-3.0 as f32).exp());
        assert_approx_eq::assert_approx_eq!(loss.into_scalar(), 0.0);
    }

    #[test]
    fn test_ent_coef() {
        type Backend = Autodiff<NdArray>;
        let device: <Autodiff<NdArray> as burn::prelude::Backend>::Device = Default::default();

        let target_entropy = -1.0;
        let lr = 1.0;
        let starting_ent = 1.0;
        let log_probs: Tensor<Backend, 2> =
            Tensor::<Backend, 1>::from_floats([-2.0], &Default::default()).unsqueeze();

        let mut ent: EntCoef<Backend> =
            EntCoef::new(starting_ent, true, target_entropy, Some(&device));

        let ent_before = match &ent {
            EntCoef::Constant(_) => panic!("shouldn't be here"),
            EntCoef::Trainable(m, _, _) => m.alpha(),
        };

        ent.train_step(log_probs, lr, &device);

        let ent_after = match &ent {
            EntCoef::Constant(_) => panic!("shouldn't be here"),
            EntCoef::Trainable(m, _, _) => m.alpha(),
        };

        assert!(ent_after < ent_before);
    }
}
