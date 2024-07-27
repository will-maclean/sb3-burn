use std::iter::zip;

use burn::{module::Module, nn::{loss::{MseLoss, Reduction}, Linear}, optim::{adaptor::OptimizerAdaptor, Adam, GradientsParams, Optimizer}, tensor::{backend::{AutodiffBackend, Backend}, ElementConversion, Shape, Tensor}};

use crate::common::{agent::{Agent, Policy}, buffer::ReplayBuffer, distributions::{distribution::BaseDistribution, normal::Normal}, logger::{LogData, LogItem}, spaces::{BoxSpace, Space}, to_tensor::{ToTensorB, ToTensorF}, utils::modules::MLP};

#[derive(Debug, Module)]
pub struct PiModel<B: Backend>{
    mlp: MLP<B>
}

impl <B: Backend> PiModel<B>{
    pub fn new(obs_size: usize, n_actions: usize, device: &B::Device) -> Self{
        Self {
            mlp: MLP::new(&[obs_size, 64, 2 * n_actions].to_vec(), device)
        }
    }
}


impl <B: Backend> PiModel<B>{
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2>{
        self.mlp.forward(x)
    }

    pub fn get_dist_params(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>){
        let x = self.forward(obs);
        
        let b = x.shape().dims[0];
        let n = x.shape().dims[1] / 2;
        
        let means = x.clone().slice([0..b, 0..n]);
        let stds = x.slice([0..b, n..n*2]).exp();

        (means, stds)
    }
    
    pub fn act(&self, obs: &Tensor<B, 1>) -> Tensor<B, 1>{
        let (means, stds) = self.get_dist_params(obs.clone().unsqueeze_dim(0));
        
        let mut dist = Normal::new(means, stds);
        
        dist.rsample().squeeze(0)
    }
    
    pub fn act_log_prob(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>){
        let (means, stds) = self.get_dist_params(obs);
        
        let mut dist = Normal::new(means, stds);
        
        let actions = dist.rsample();
        let log_prob = dist.log_prob(actions.clone());
        
        (actions, log_prob)
    }
}

#[derive(Debug, Module)]
pub struct QModel<B: Backend>{
    mlp: MLP<B>
}

impl <B: Backend> QModel<B>{
    pub fn new(obs_size: usize, n_actions: usize, device: &B::Device) -> Self{
        Self {
            mlp: MLP::new(&[obs_size + n_actions, 64, n_actions].to_vec(), device)
        }
    }
}

impl<B: Backend> Policy<B> for QModel<B>{
    fn update(&mut self, from: &Self, tau: Option<f32>) {
        self.mlp.update(&from.mlp, tau)
    }
}

impl <B: Backend> QModel<B>{
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.mlp.forward(x)
    }

    pub fn q_from_actions(&self, obs: Tensor<B, 2>, actions: Tensor<B, 2>) -> Tensor<B, 2>{
        let x = Tensor::cat(Vec::from([obs, actions]), 1);

        self.forward(x)
    }
}

#[derive(Debug, Module)]
pub struct QModelSet<B: Backend>{
    qs: Vec<QModel<B>>
}

impl<B: AutodiffBackend> QModelSet<B>{
    pub fn new(obs_size: usize, n_actions: usize, device: &B::Device, n_critics: usize) -> Self {
        let mut qs = Vec::new();

        for _ in 0..n_critics {
            qs.push(QModel::new(
                obs_size,
                n_actions,
                device,
            ));
        }

        Self {
            qs,
        }
    }
    pub fn q_from_actions(&self, obs: Tensor<B, 2>, actions: Tensor<B, 2>) -> Vec<Tensor<B, 2>>{
        self.qs.iter().map(|q| q.q_from_actions(obs.clone(), actions.clone())).collect()
    }
}

impl<B: AutodiffBackend> Policy<B> for QModelSet<B>{
    fn update(&mut self, from: &Self, tau: Option<f32>) {
        for i in 0..self.qs.len() {
            self.qs[i].update(&from.qs[i], tau);
        }
    }
}

pub struct SACAgent<B: AutodiffBackend>{
    // models
    pi: PiModel<B>,
    qs: QModelSet<B>,
    target_qs: QModelSet<B>,

    // optimisers
    pi_optim: OptimizerAdaptor<Adam<B::InnerBackend>, PiModel<B>, B>,
    q_optim: OptimizerAdaptor<Adam<B::InnerBackend>, QModelSet<B>, B>,

    // parameters
    ent_coef: f32,

    // housekeeping
    observation_space: Box<BoxSpace<Vec<f32>>>,
    action_space: Box<BoxSpace<Vec<f32>>>,
    last_update: usize,
    update_every: usize,
}

impl<B: AutodiffBackend> SACAgent<B>{
    pub fn new(
        // models
        pi: PiModel<B>,
        qs: QModelSet<B>,
        target_qs: QModelSet<B>,

        // optimisers
        pi_optim: OptimizerAdaptor<Adam<B::InnerBackend>, PiModel<B>, B>,
        q_optim: OptimizerAdaptor<Adam<B::InnerBackend>, QModelSet<B>, B>,

        // parameters
        ent_coef: f32,

        // housekeeping
        observation_space: Box<BoxSpace<Vec<f32>>>,
        action_space: Box<BoxSpace<Vec<f32>>>,
    ) -> Self{
        Self{
            pi,
            qs,
            target_qs,
            pi_optim,
            q_optim,
            ent_coef,
            observation_space,
            action_space,
            last_update: 0,
            update_every: 100,
        }
    }
}

impl<B: AutodiffBackend> Agent<B, Vec<f32>, Vec<f32>> for SACAgent<B>{
    fn act(
        &self,
        _global_step: usize,
        _global_frac: f32,
        obs: &Vec<f32>,
        _greedy: bool,
        inference_device: &<B>::Device,
    ) -> (Vec<f32>, LogItem) {

        // don't judge me
        let a: Vec<f32> = self.pi.act(
            &obs
            .clone()
            .to_tensor(inference_device))
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
        let terminated = sample_data.terminated.to_tensor(train_device).unsqueeze_dim(0);
        let truncated = sample_data.truncated.to_tensor(train_device).unsqueeze_dim(0);
        let dones = (terminated.float() + truncated.float()).bool();

        // select action according to policy
        let (next_action_sampled, next_action_log_prob) = self.pi.act_log_prob(next_states.clone());
        // next_action_sampled
        let next_q_vals = self.target_qs.q_from_actions(next_states, next_action_sampled);
        let next_q_vals = Tensor::cat(next_q_vals, 1);
        let next_q_vals = next_q_vals.min_dim(1);
        // add the entropy term
        let next_q_vals = next_q_vals - next_action_log_prob.mul_scalar(self.ent_coef);
        // td error + entropy term
        let target_q_vals = rewards + dones.bool_not().float().mul(next_q_vals).mul_scalar(offline_params.gamma);
        
        // calculate the critic loss
        let q_vals = self.qs.q_from_actions(states.clone(), actions.clone());
        
        let mut critic_loss: Tensor<B, 1> = Tensor::zeros(Shape::new([1]), train_device);
        for q in q_vals{
            critic_loss = critic_loss +  MseLoss::new().forward(q, target_q_vals.clone(), Reduction::Mean);
        }
        critic_loss = critic_loss.mul_scalar(0.5);

        let log_dict = log_dict.push("critic_loss_combined".to_string(), LogData::Float(critic_loss.clone().into_scalar().elem()));
        
        // optimise the critics
        let critic_loss_grads = critic_loss.clone().backward();
        let critic_grads = GradientsParams::from_grads(critic_loss_grads, &self.qs);
        self.qs = self.q_optim.step(offline_params.lr, self.qs.clone(), critic_grads);
        
        // Policy loss
        // recalculate q values with new critics
        let (actions_pi, log_prob) = self.pi.act_log_prob(states.clone());
        let q_vals = self.qs.q_from_actions(states.clone(), actions_pi);
        let q_vals = Tensor::cat(q_vals, 1);
        let min_q = q_vals.min_dim(1);
        let actor_loss = log_prob.mul_scalar(self.ent_coef) - min_q;
        let actor_loss = actor_loss.mean();

        let log_dict = log_dict.push("actor_loss".to_string(), LogData::Float(actor_loss.clone().into_scalar().elem()));

        let actor_loss_back = actor_loss.backward();
        let actor_grads = GradientsParams::from_grads(actor_loss_back, &self.pi);
        self.pi = self.pi_optim.step(offline_params.lr, self.pi.clone(), actor_grads);

        // target critic updates
        if global_step > (self.last_update + self.update_every) {
            // hard update
            self.target_qs.update(&self.qs, None);

            self.last_update = global_step;
        }

        // TODO: create log dict
        (None, log_dict)
    }

    fn observation_space(&self) -> Box<dyn Space<Vec<f32>>> {
        self.observation_space.clone()
    }

    fn action_space(&self) -> Box<dyn Space<Vec<f32>>> {
        self.action_space.clone()
    }
}
