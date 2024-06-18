# Agents
The `Agent` trait defines the behaviour of a Reinforcement Learning (RL) agent in `sb3-burn`. 
All RL agents need to implement this trait. 

## Features
_Generic States and Actions_

Different environments will take different state and action types. `Agent` allows for the
generics `O` and `A`, for the type of state and action respectively. A particular `Agent`
may have different implementations for different state and action types e.g. do different
pre-processing based on the observation type.


## Definition
The trait is defined as:

```rust
pub trait Agent<B: Backend, O: Clone, A: Clone> {
    fn act(
        &self,
        global_step: usize,
        global_frac: f32,
        obs: &O,
        greedy: bool,
        inference_device: &<B as Backend>::Device,
    ) -> (A, LogItem);

    fn train_step(
        &mut self,
        global_step: usize,
        replay_buffer: &ReplayBuffer<O, A>,
        offline_params: &OfflineAlgParams,
        train_device: &B::Device,
    ) -> (Option<f32>, LogItem);

    fn eval(
        &mut self,
        env: &mut dyn Env<O, A>,
        cfg: &EvalConfig,
        eval_device: &B::Device,
    ) -> LogItem;

    fn observation_space(&self) -> Box<dyn Space<O>>;

    fn action_space(&self) -> Box<dyn Space<A>>;
}
```

## Example implementation
As an example, we can define a `RandomAgent`, which takes
random actions in any space:

```rust
struct RandomAgent<O, A>{
    observation_space: Box<Space<O>>;
    action_space: Box<Space<A>>;
}

impl <O, A> RandomAgent<O, A>{
    fn new(observation_space: Box<Space<O>>, action_space: Box<Space<A>>) -> Self {
        Self {
            observation_space,
            action_space,
        }
    }
}

impl<O, A> Agent<O, A> for RandomAgent<O, A>{
    fn act(
        &self,
        _global_step: usize,
        _global_frac: f32,
        _obs: &O,
        _greedy: bool,
        _inference_device: &<B as Backend>::Device,
    ) -> (A, LogItem){
        self.action_space.sample()
    }

    fn train_step(
        &mut self,
        global_step: usize,
        replay_buffer: &ReplayBuffer<O, A>,
        offline_params: &OfflineAlgParams,
        train_device: &B::Device,
    ) -> (Option<f32>, LogItem){
        (None, Default::default())
    }

    fn eval(
        &mut self,
        env: &mut dyn Env<O, A>,
        cfg: &EvalConfig,
        eval_device: &B::Device,
    ) -> LogItem{
        evaluate_policy(self, env, cfg, eval_device)
    }

    fn observation_space(&self) -> Box<dyn Space<O>>{
        dyn_clone::clone_box(&*self.observation_space)
    }

    fn action_space(&self) -> Box<dyn Space<A>>{
        dyn_clone::clone_box(&*self.action_space)
    }
}