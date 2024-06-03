use burn::tensor::backend::Backend;

use crate::{buffer::ReplayBuffer, env::Env, policy::{DQNNet, Policy}, spaces::{Space, SpaceSample}};

struct OfflineAlgParams{
    n_steps: usize,
    batch_size: usize,
    warmup_steps: usize,
}

// I think this current layout will do for now, will likely need to be refactored at some point
enum OfflineAlgorithm<B: Backend>{
    DQN {q: DQNNet<B>},
    DDQN {},
    SAC {},
    RainbowDQN{},
}

impl<B: Backend> OfflineAlgorithm<B> {
    fn train_step(&self, replay_buffer: &ReplayBuffer<B>, offline_params: &OfflineAlgParams) {
        match self {
            OfflineAlgorithm::DQN { q } => {
                
            },
            _ => todo!(),
        }
    }

    fn act(&self, state: &SpaceSample) -> SpaceSample {
        match self {
            OfflineAlgorithm::DQN { q } => q.act(state),
            _ => todo!(),
        }
    }
}

struct OfflineTrainer<B: Backend>{
    offline_params: OfflineAlgParams,
    env: Box<dyn Env>,
    algorithm: OfflineAlgorithm<B>,
    buffer: ReplayBuffer<B>,
}

impl<B: Backend> OfflineTrainer<B> {
    fn train(&mut self) {
        let mut state = self.env.reset();

        for i in 0..self.offline_params.n_steps {
            let action: SpaceSample;

            if i < self.offline_params.warmup_steps {
                action = self.env.action_space().sample();
            } else {
                action = self.algorithm.act(&state);
            }

            let step_res = self.env.step(&action);
            let (next_obs, reward, done) = (step_res.obs, step_res.reward, step_res.done);

            self.buffer.add(
                state, 
                action, 
                next_obs.clone(), 
                reward, 
                done
            );

            if i >= self.offline_params.warmup_steps {
                self.algorithm.train_step(&self.buffer, &self.offline_params);
            }

            if done {
                state = self.env.reset();
            } else {
                state = next_obs;
            }
        }
    }
}