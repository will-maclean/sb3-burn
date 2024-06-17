use rand::seq::SliceRandom;
use rand::thread_rng;

pub struct BatchedReplayBufferSlice<O, A> {
    pub states: Vec<O>,
    pub actions: Vec<A>,
    pub next_states: Vec<O>,
    pub rewards: Vec<f32>,
    pub terminated: Vec<bool>,
    pub truncated: Vec<bool>,
}

pub struct ReplayBufferSlice<O, A> {
    state: O,
    action: A,
    next_state: O,
    reward: f32,
    terminated: bool,
    truncated: bool,
}

pub struct ReplayBuffer<O: Clone, A: Clone> {
    states: Vec<O>,
    actions: Vec<A>,
    next_states: Vec<O>,
    rewards: Vec<f32>,
    terminated: Vec<bool>,
    truncated: Vec<bool>,
    size: usize,
    full: bool,
    ptr: usize,
}

impl<O: Clone, A: Clone> ReplayBuffer<O, A> {
    pub fn new(size: usize) -> Self {
        Self {
            states: Vec::with_capacity(size),
            actions: Vec::with_capacity(size),
            next_states: Vec::with_capacity(size),
            rewards: Vec::with_capacity(size),
            terminated: Vec::with_capacity(size),
            truncated: Vec::with_capacity(size),
            size,
            full: false,
            ptr: 0,
        }
    }

    pub fn full(&self) -> bool {
        self.full
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn get(&self, idx: usize) -> Result<ReplayBufferSlice<O, A>, &'static str> {
        Ok(ReplayBufferSlice {
            state: self.states[idx].clone(),
            action: self.actions[idx].clone(),
            next_state: self.next_states[idx].clone(),
            reward: self.rewards[idx].clone(),
            terminated: self.terminated[idx].clone(),
            truncated: self.truncated[idx].clone(),
        })
    }

    pub fn add_slice(&mut self, item: ReplayBufferSlice<O, A>) {
        if self.full() {
            self.states[self.ptr] = item.state;
            self.actions[self.ptr] = item.action;
            self.next_states[self.ptr] = item.next_state;
            self.rewards[self.ptr] = item.reward;
            self.terminated[self.ptr] = item.terminated;
            self.truncated[self.ptr] = item.truncated;
        } else {
            self.states.push(item.state);
            self.actions.push(item.action);
            self.next_states.push(item.next_state);
            self.rewards.push(item.reward);
            self.terminated.push(item.terminated);
            self.truncated.push(item.truncated);
        }

        if self.ptr == self.len() - 1 {
            self.full = true;
        }

        self.ptr = (self.ptr + 1) % self.len();
    }

    pub fn add(
        &mut self,
        state: O,
        action: A,
        next_state: O,
        reward: f32,
        terminated: bool,
        truncated: bool,
    ) {
        self.add_slice(ReplayBufferSlice {
            state,
            action,
            next_state,
            reward,
            terminated,
            truncated,
        })
    }

    pub fn batch_sample(&self, batch_size: usize) -> Option<BatchedReplayBufferSlice<O, A>> {
        if (self.full & (batch_size > self.size)) | (!self.full & (batch_size > self.ptr)) {
            return None;
        }

        let mut rng = thread_rng();

        Some(BatchedReplayBufferSlice {
            states: self
                .states
                .choose_multiple(&mut rng, batch_size)
                .cloned()
                .collect(),
            actions: self
                .actions
                .choose_multiple(&mut rng, batch_size)
                .cloned()
                .collect(),
            next_states: self
                .next_states
                .choose_multiple(&mut rng, batch_size)
                .cloned()
                .collect(),
            rewards: self
                .rewards
                .choose_multiple(&mut rng, batch_size)
                .cloned()
                .collect(),
            terminated: self
                .terminated
                .choose_multiple(&mut rng, batch_size)
                .cloned()
                .collect(),
            truncated: self
                .truncated
                .choose_multiple(&mut rng, batch_size)
                .cloned()
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::spaces::{BoxSpace, Discrete, Space};

    use super::ReplayBuffer;

    #[test]
    fn test_create_replay_buffer() {
        // let device = burn::backend::wgpu::WgpuDevice::default();
        let mut observation_space = Discrete::from(6);
        let mut action_space = BoxSpace::from((vec![0.0, 0.0, 0.1], vec![1.0, 0.2, 0.1]));

        let mut buffer = ReplayBuffer::new(1000);

        buffer.add(
            observation_space.sample(),
            action_space.sample(),
            observation_space.sample(),
            0.5,
            false,
            false,
        );
    }

    #[test]
    fn test_batch_sample_before_ready() {
        let buffer = ReplayBuffer::<usize, Vec<f32>>::new(1000);

        let sample = buffer.batch_sample(64);

        assert!(sample.is_none());
    }

    #[test]
    fn test_batch_sample() {
        let mut observation_space = Discrete::from(6);
        let mut action_space = BoxSpace::from((vec![0.0, 0.0, 0.1], vec![1.0, 0.2, 0.1]));

        let mut buffer = ReplayBuffer::new(1000);

        for _ in 0..32 {
            buffer.add(
                observation_space.sample(),
                action_space.sample(),
                observation_space.sample(),
                0.5,
                false,
                false,
            );
        }

        let _ = buffer.batch_sample(4);
    }
}
