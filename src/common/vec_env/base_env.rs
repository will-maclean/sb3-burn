use crate::{
    common::spaces::Space,
    env::base::{EnvObservation, ResetOptions, RewardRange},
};

pub trait VecEnv<O, A> {
    async fn step(&mut self, action: Vec<A>) -> Vec<EnvObservation<O>> {
        self.step_async(action).await;
        self.step_wait().await
    }
    async fn step_async(&mut self, action: Vec<A>);
    async fn step_wait(&mut self) -> Vec<EnvObservation<O>>;
    async fn reset(&mut self, seed: Option<u64>, options: Option<ResetOptions>) -> Vec<O>;
    fn action_space(&self) -> Box<dyn Space<A>>;
    fn observation_space(&self) -> Box<dyn Space<O>>;
    fn reward_range(&self) -> RewardRange;
    fn render(&self);
    fn renderable(&self) -> bool;
    fn close(&mut self);
}
