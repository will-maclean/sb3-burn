use rand::seq::SliceRandom;

/// Stores a batch of `ReplayBuffer<O, A>` samples
pub struct BatchedReplayBufferSlice<O, A> {
    pub states: Vec<O>,
    pub actions: Vec<A>,
    pub next_states: Vec<O>,
    pub rewards: Vec<f32>,
    pub terminated: Vec<bool>,
    pub truncated: Vec<bool>,
}

/// Stores a single `ReplayBuffer<O, A>` sample
pub struct ReplayBufferSlice<O, A> {
    state: O,
    action: A,
    next_state: O,
    reward: f32,
    terminated: bool,
    truncated: bool,
}

/// ReplayBuffer stores samples of training data
///
/// ReplayBuffer stores samples of training data, specifically
/// state, action, next state, reward, terminated, and truncated.
/// The types of state and action are generic for compatability
/// with any observation/action type.
pub struct ReplayBuffer<O: Clone, A: Clone> {
    states: Vec<O>,
    actions: Vec<A>,
    next_states: Vec<O>,
    rewards: Vec<f32>,
    terminated: Vec<bool>,
    truncated: Vec<bool>,

    /// stores the maximum size of the buffer
    size: usize,

    /// stores the current replace position in the circular array
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
            ptr: 0,
        }
    }

    /// Whether the replay buffer is currently full
    pub fn full(&self) -> bool {
        self.curr_len() == self.size
    }

    /// The number of samples currently stored in the buffer
    pub fn curr_len(&self) -> usize {
        self.states.len()
    }

    /// The maximum number of samples this buffer can hold
    pub fn size(&self) -> usize {
        self.size
    }

    /// Indexes the replay buffer. No verification on idx
    /// so will panic if an illegal idx is supplied
    ///
    /// # Panics
    ///
    /// Panics if an illegal idx is supplied
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

    /// Adds a ReplayBufferSlice<O, A>, which is a single
    /// sample of data, the to buffer
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

        self.ptr = (self.ptr + 1) % self.size();
    }

    /// Adds a single piece of data to the replay buffer
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

    /// Randomly samples `batch_size` samples from the buffer and returns
    /// them as a BatchedReplayBuffer<O, A>.
    ///
    /// Note, the data is cloned out of the data (not removed).
    ///
    /// # Panics
    /// Will panic if there is not enough data in the buffer for the batch sample.
    pub fn batch_sample(&self, batch_size: usize) -> BatchedReplayBufferSlice<O, A> {
        if (self.full() & (batch_size > self.size()))
            | (!self.full() & (batch_size > self.curr_len()))
        {
            panic!(
                "Not enough samples in here! self.len: {:?}, batch_size: {:?}",
                self.curr_len(),
                batch_size
            );
        }

        let mut rng = rand::rng();

        // Generate a list of indices
        let mut indices: Vec<usize> = (0..self.states.len()).collect();
        indices.shuffle(&mut rng);

        // Take the first n indices
        let indices: Vec<usize> = indices.into_iter().take(batch_size).collect();

        BatchedReplayBufferSlice {
            states: indices.iter().map(|&i| self.states[i].clone()).collect(),
            actions: indices.iter().map(|&i| self.actions[i].clone()).collect(),
            next_states: indices
                .iter()
                .map(|&i| self.next_states[i].clone())
                .collect(),
            rewards: indices.iter().map(|&i| self.rewards[i].clone()).collect(),
            terminated: indices
                .iter()
                .map(|&i| self.terminated[i].clone())
                .collect(),
            truncated: indices.iter().map(|&i| self.truncated[i].clone()).collect(),
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::common::spaces::{BoxSpace, Discrete, Space};

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

    #[should_panic]
    #[test]
    fn test_batch_sample_before_ready() {
        let buffer = ReplayBuffer::<usize, Vec<f32>>::new(1000);

        buffer.batch_sample(64);
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

    #[test]
    fn test_full() {
        let mut buffer: ReplayBuffer<usize, usize> = ReplayBuffer::new(5);

        for _ in 0..5 {
            assert!(!buffer.full());
            buffer.add(0, 0, 0, 0.0, false, false)
        }

        assert!(buffer.full());
    }

    #[test]
    fn allocated_vec_sanity_check() {
        let v1: Vec<usize> = Vec::with_capacity(10);

        assert_eq!(v1.len(), 0);
    }
}
