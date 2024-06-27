use burn::{module::AutodiffModule, nn::loss::{MseLoss, Reduction}, optim::{adaptor::OptimizerAdaptor, SimpleOptimizer}, tensor::backend::AutodiffBackend};

use crate::common::{agent::Agent, algorithm::OfflineAlgParams, buffer::ReplayBuffer, logger::LogItem, spaces::{BoxSpace, Space}, to_tensor::ToTensorB};

use super::{module::{QVals, SACNet}, utils::{ActionLogProb, EntCoef, SACConfig}};

pub struct SACAgent<O, B, S, OS, AS> 
where
    B: AutodiffBackend,
    O: SimpleOptimizer<B::InnerBackend>,
    S: SACNet<B, OS, AS> + AutodiffModule<B>,
    OS: Clone,
    AS: Clone + 'static,
    BoxSpace<AS> : Space<AS>
{
    pub net: S,
    pub target_net: S,
    pub optim: OptimizerAdaptor<O, S, B>,
    pub config: SACConfig,
    pub target_entropy: f32,
    pub critic_loss: MseLoss<B>,
    pub ent_coef: EntCoef<B>,
    pub observation_space: Box<dyn Space<OS>>,
    pub action_space: BoxSpace<AS>,
}

impl<O, B, S, OS, AS> SACAgent<O, B, S, OS, AS> 
where
    B: AutodiffBackend,
    O: SimpleOptimizer<B::InnerBackend>,
    S: SACNet<B, OS, AS> + AutodiffModule<B>,
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
    S: SACNet<B, OS, AS> + AutodiffModule<B>,
    OS: Clone,
    AS: Clone + 'static,
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

        if self.config.use_sde{
            self.reset_noise();
        }

        let act_log_prob = self.action_log_prob(sample.states);

        // TODO: optimise the entropy coefficient if it is trainable
        // TODO: how do I do with th.no_grad() ?
        let next_act_log_prob = self.action_log_prob(sample.next_states);
        let next_q_vals = self.q_val_from_action(sample.next_states, next_act_log_prob.pi);

        //TODO: SB3 has support for n q nets, rather than just 2. Add support for this

        // add the entropy term
        let next_q1_vals = next_q_vals.q1 - next_act_log_prob.log_pi.mul_scalar(self.ent_coef.to_float());
        let next_q2_vals = next_q_vals.q2 - next_act_log_prob.log_pi.mul_scalar(self.ent_coef.to_float());
        let target_q1 = rewards + dones.bool_not().float().mul(next_q1_vals).mul_scalar(offline_params.gamma);
        let target_q2 = rewards + dones.bool_not().float().mul(next_q2_vals).mul_scalar(offline_params.gamma);

        let q_vals = self.q_val_from_action(sample.states, sample.actions);

        let critic_loss = (
                self.critic_loss.forward(q_vals.q1, target_q1, Reduction::Mean) 
                + self.critic_loss.forward(q_vals.q2, target_q2, Reduction::Mean)
            ).mul_scalar(0.5);
        
        // TODO: optimise the critics

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