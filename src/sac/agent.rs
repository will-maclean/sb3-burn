use std::iter::zip;

use burn::{module::AutodiffModule, nn::loss::{MseLoss, Reduction}, optim::{adaptor::OptimizerAdaptor, GradientsParams, Optimizer, SimpleOptimizer}, tensor::{backend::AutodiffBackend, ElementConversion, Shape, Tensor}};

use crate::common::{agent::Agent, algorithm::OfflineAlgParams, buffer::ReplayBuffer, distributions::action_distribution::DiagGaussianDistribution, logger::LogItem, spaces::{BoxSpace, Space}, to_tensor::{ToTensorB, ToTensorF}};

use super::{module::{QVals, SACNet}, utils::{ActionDist, ActionLogProb, EntCoef, SACConfig}};

pub struct SACAgent<O, B, S, OS, AS> 
where
    B: AutodiffBackend,
    O: SimpleOptimizer<B::InnerBackend>,
    S: SACNet<B, OS, AS, DiagGaussianDistribution<B>> + AutodiffModule<B>,
    OS: Clone,
    AS: Clone + 'static,
    BoxSpace<AS> : Space<AS>
{
    pub net: S,
    pub target_net: S,
    pub pi_optim: OptimizerAdaptor<O, S, B>,
    pub q_optim: OptimizerAdaptor<O, S, B>,
    pub config: SACConfig,
    pub target_entropy: f32,
    pub critic_loss: MseLoss<B>,
    pub log_ent_coef: EntCoef<B>,
    pub observation_space: Box<dyn Space<OS>>,
    pub action_space: BoxSpace<AS>,
    pub action_dist_type: ActionDist,
}

impl<O, B, S, OS, AS> SACAgent<O, B, S, OS, AS> 
where
    B: AutodiffBackend,
    O: SimpleOptimizer<B::InnerBackend>,
    S: SACNet<B, OS, AS, DiagGaussianDistribution<B>> + AutodiffModule<B>,
    OS: Clone,
    AS: Clone + 'static,
    BoxSpace<AS> : Space<AS>
{
    fn action_log_prob(&self, obs: Vec<OS>) -> ActionLogProb<B, AS>{
        todo!();
    }

    fn q_val_from_action(&self, obs: Vec<OS>, acts: Vec<AS>) -> QVals<B>{
        // take min over the q networks 
        todo!();
    }

    fn reset_noise(&self){
        todo!();
    }
}

impl <O, B, S, OS, AS> Agent<B, OS, AS> for SACAgent<O, B, S, OS, AS>
where
    B: AutodiffBackend,
    O: SimpleOptimizer<B::InnerBackend>,
    S: SACNet<B, OS, AS, DiagGaussianDistribution<B>> + AutodiffModule<B>,
    OS: Clone,
    AS: Clone,
    BoxSpace<AS> : Space<AS>
{
    fn act(
        &self,
        global_step: usize,
        global_frac: f32,
        obs: &OS,
        greedy: bool,
        inference_device: &B::Device,
    ) -> (AS, LogItem) {
        todo!()
    }

    fn train_step(
        &mut self,
        global_step: usize,
        replay_buffer: &ReplayBuffer<OS, AS>,
        offline_params: &OfflineAlgParams,
        train_device: &B::Device,
    ) -> (Option<f32>, LogItem) {
        
        let sample = replay_buffer.batch_sample(offline_params.batch_size);
        let rewards = sample.rewards
            .to_tensor(train_device)
            .unsqueeze_dim(1);

        let dones = sample.terminated
            .to_tensor(train_device)
            .float()
            .add(sample.truncated.to_tensor(train_device).float())
            .bool()
            .unsqueeze_dim(1);

        let act_log_prob = self.action_log_prob(sample.states);

        // optimise the entropy coefficient if it is trainable
        let ent_coef = match self.log_ent_coef{
            EntCoef::Static(ec) => ec,
            EntCoef::Trainable(t, o, lr) => {
                let ent_coef = t.clone().into_scalar().elem::<f32>().exp();

                let ent_coef_loss = -t.mul(act_log_prob.log_pi.squeeze(1).add_scalar(self.target_entropy)).detach().mean();

                let g = ent_coef_loss.backward();
                let g = GradientsParams::from_grads(g, &t);
                t = o.step(lr, t, g);

                ent_coef
            },
        };

        let next_act_log_prob = self.action_log_prob(sample.next_states);
        let next_q_vals = self.q_val_from_action(sample.next_states, next_act_log_prob.pi);

        // add the entropy term
        let mut targets = Vec::new();
        for q in next_q_vals.q{
            let next_q = - next_act_log_prob.log_pi.mul_scalar(ent_coef) + 1;
            let target = rewards + dones.bool_not().float().mul(next_q).mul_scalar(offline_params.gamma);
            targets.push(target.detach());
        }

        // calculate the critic loss
        let q_vals = self.q_val_from_action(sample.states, sample.actions);

        let mut critic_loss: Tensor<_, 1> = Tensor::zeros(Shape::new([1]), train_device);
        for (q, target) in zip(q_vals.q, targets){
            critic_loss = critic_loss + self.critic_loss.forward(q, target, Reduction::Mean);
        }
        critic_loss = critic_loss.mul_scalar(0.5);
        
        // TODO: optimise the critics only
        let critic_loss = critic_loss.backward();
        let critic_grads = GradientsParams::from_grads(critic_loss, &self.net);
        self.net = self.q_optim.step(offline_params.lr, self.net, critic_grads);

        // TODO: Train the policy network

        // TODO: create log dict
    }

    fn observation_space(&self) -> Box<dyn Space<OS>> {
        dyn_clone::clone_box(&*self.observation_space)
    }

    fn action_space(&self) -> Box<dyn Space<AS>> {
        Box::new(self.action_space.clone())
    }
}