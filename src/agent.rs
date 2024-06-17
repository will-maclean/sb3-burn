use burn::module::Module;
use burn::tensor::backend::Backend;

use crate::{
    algorithm::OfflineAlgParams, buffer::ReplayBuffer, env::base::Env, eval::EvalConfig,
    logger::LogItem, spaces::Space,
};

/// `Agent` is the base trait for all Reinforcement Learning
/// Agents.
/// 
/// `Agent` is the base trait for all Reinforcement Learning 
/// Agents, and is roughly synonymous with a Policy in 
/// Stable-Baselines3. The Observation and Action types
/// are generic so as to be as flexible as possible with 
/// different agent and environment types. The `Agent` class
/// itself is not a burn `Module`, although it should be possible
/// for users to implement `Agent` and `Module` from the same
/// struct.
pub trait Agent<B: Backend, O: Clone, A: Clone> {

    /// Ask the agent to generate an action
    /// 
    /// Params:
    /// global_step: the current global step of training, can be useful for exploration schemes
    /// global_frac: the current percentage of training complete (as a decimal), can be useful for exploration schemes
    /// obs: the observation from the environment
    /// greedy: hard control over whether the agent should be exploiting. If true, the agent is expected to 
    /// act greedily. If false, the agent may use an exploration scheme.
    /// inference device: The Backend device on which the act operation should run. This may require moving modules
    /// to the correct device.
    fn act(
        &self,
        global_step: usize,
        global_frac: f32,
        obs: &O,
        greedy: bool,
        inference_device: &<B as Backend>::Device,
    ) -> (A, LogItem);

    /// Performs a single step of training
    /// 
    /// Params:
    /// global_step: current global step of training
    /// replay_buffer: the replay buffer instance storing the training data
    /// offline_params: the general training parameters. Agents will generally have their own config/parameters
    /// as well, but some generic ones are stored in offline_params.
    /// train_device: The Backend device on which the train_step operation should run. This may require moving modules
    /// to the correct device.
    fn train_step(
        &mut self,
        global_step: usize,
        replay_buffer: &ReplayBuffer<O, A>,
        offline_params: &OfflineAlgParams,
        train_device: &B::Device,
    ) -> (Option<f32>, LogItem);

    /// Evaluate the current policy on the given environment.
    /// 
    /// Params:
    /// env: the environment on which to evaluate
    /// cfg: the config defining the evaluation process, such as the number of evaluation episodes to run
    /// eval_device: The Backend device on which the eval operation should run. This may require moving modules
    /// to the correct device.
    fn eval(
        &mut self,
        env: &mut dyn Env<O, A>,
        cfg: &EvalConfig,
        eval_device: &B::Device,
    ) -> LogItem;

    /// The observation space for the Agent. Should be passed in from the environment
    /// 
    /// `Space<T>` is cloned using dyn_clone.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// // This is an example agent
    /// struct MyAgent{
    ///     // Our agent will only work with Discrete<usize> observation spaces
    ///     obs_space: Discrete<usize>
    /// 
    ///     // Our agent will only work with BoxSpace<Vec<f32>> action spaces
    ///     action_space: BoxSpace<Vec<f32>>
    /// }
    /// 
    /// impl<B: Backend> Agent<B, usize, Vec<f32>> for MyAgent{
    ///     // ... Other function implementations here
    /// 
    ///     fn observation_space(&self) -> Box<dyn Space<usize>> {
    ///         dyn_clone::clone_box(&*self.observation_space)
    ///     }
    /// 
    ///     fn action_space(&self) -> Boc<dyn Space<Vec<f32>>> {
    ///         dyn_clone::clone_box(&*self.action_space)
    ///     }
    /// }
    /// ```
    fn observation_space(&self) -> Box<dyn Space<O>>;

    /// The action space for the Agent. Should be passed in from the environment
    /// 
    /// Space<T> is cloned using dyn_clone.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// // This is an example agent
    /// struct MyAgent{
    ///     // Our agent will only work with Discrete<usize> observation spaces
    ///     obs_space: Discrete<usize>
    /// 
    ///     // Our agent will only work with BoxSpace<Vec<f32>> action spaces
    ///     action_space: BoxSpace<Vec<f32>>
    /// }
    /// 
    /// impl<B: Backend> Agent<B, usize, Vec<f32>> for MyAgent{
    ///     // ... Other function implementations here
    /// 
    ///     fn observation_space(&self) -> Box<dyn Space<usize>> {
    ///         dyn_clone::clone_box(&*self.observation_space)
    ///     }
    /// 
    ///     fn action_space(&self) -> Boc<dyn Space<Vec<f32>>> {
    ///         dyn_clone::clone_box(&*self.action_space)
    ///     }
    /// }
    /// ```
    fn action_space(&self) -> Box<dyn Space<A>>;
}


/// The Policy trait defines the specific behaviours we need from
/// a burn Module used inside an Reinforcement Learning Agent.
pub trait Policy<B: Backend>: Module<B> + Clone + Sized {

    /// Perform a weight update from another Policy.
    /// This can either be a "hard" update, meaning weights
    /// are copied entirely from `from` to 'self', or 
    /// can be a "soft" copy, where the copy follows 
    /// the form:
    /// 
    /// self = tau * self + (1 - tau) * from
    fn update(&mut self, from: &Self, tau: Option<f32>);
}
