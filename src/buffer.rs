use burn::prelude::*;

use crate::{spaces::SpaceSample, utils::generate_rand_bool_vector};

pub struct ReplayBufferSlice<B: Backend> {
    state: Tensor<B, 1>,
    action: Tensor<B, 1>,
    next_state: Tensor<B, 1>,
    reward: f32,
    done: bool,
}

pub struct ReplayBuffer<B: Backend> {
    states: Tensor<B, 2>,
    actions: Tensor<B, 2>,
    next_states: Tensor<B, 2>,
    rewards: Tensor<B, 2>,
    dones: Tensor<B, 2, Int>,
    size: usize,
    full: bool,
    ptr: usize,
}

impl<B: Backend> ReplayBuffer<B> {
    //TODO: probably want to take in spaces instead of usize's
    pub fn new(size: usize, state_dim: usize, action_dim: usize) -> Self {
        let d = Default::default();
        Self {
            states: Tensor::<B, 2>::zeros([size, state_dim], &d),
            actions: Tensor::<B, 2>::zeros([size, action_dim], &d),
            next_states: Tensor::<B, 2>::zeros([size, state_dim], &d),
            rewards: Tensor::<B, 2>::zeros([size, 1], &d),
            dones: Tensor::<B, 2, Int>::empty([size, 1], &d),
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

    pub fn get(&self, idx: usize) -> Result<ReplayBufferSlice<B>, &'static str> {
        if (!self.full & (idx >= self.ptr)) | (self.full & (idx >= self.len())) {
            return Err("Buffer idx out of range");
        }

        Ok(ReplayBufferSlice {
            state: self.states.clone().slice([idx..idx + 1]).squeeze(0),
            action: self.actions.clone().slice([idx..idx + 1]).squeeze(0),
            next_state: self.next_states.clone().slice([idx..idx + 1]).squeeze(0),
            reward: 0.0, // FIXME: self.rewards.to_data().value[0],
            done: false, // FIXME: self.dones.clone().slice([idx..idx+1]).squeeze::<1>(0).into_scalar(),
        })
    }

    pub fn add_slice(&mut self, item: ReplayBufferSlice<B>) {
        // let d = Default::default();

        self.states = self
            .states
            .clone()
            .slice_assign([self.ptr..self.ptr + 1], item.state.unsqueeze());
        self.actions = self
            .actions
            .clone()
            .slice_assign([self.ptr..self.ptr + 1], item.action.unsqueeze());
        self.next_states = self
            .next_states
            .clone()
            .slice_assign([self.ptr..self.ptr + 1], item.next_state.unsqueeze());

        self.rewards = self.rewards.clone().slice_assign(
            [self.ptr..self.ptr + 1],
            Tensor::<B, 1>::from_floats([item.reward], &self.rewards.device()).unsqueeze_dim(0),
        );
        self.dones = self.dones.clone().slice_assign(
            [self.ptr..self.ptr + 1],
            Tensor::<B, 1, Int>::from_ints([item.done as i32], &self.dones.device())
                .unsqueeze_dim(0),
        );

        if self.ptr == self.len() - 1 {
            self.full = true;
        }

        self.ptr = (self.ptr + 1) % self.len();
    }

    pub fn add_processed(
        &mut self,
        state: Tensor<B, 1>,
        action: Tensor<B, 1>,
        next_state: Tensor<B, 1>,
        reward: f32,
        done: bool,
    ) {
        self.add_slice(ReplayBufferSlice {
            state,
            action,
            next_state,
            reward,
            done,
        })
    }

    pub fn add(
        &mut self,
        state: SpaceSample,
        action: SpaceSample,
        next_state: SpaceSample,
        reward: f32,
        done: bool,
    ) {
        self.add_processed(
            state.to_tensor(),
            action.to_tensor(),
            next_state.to_tensor(),
            reward,
            done,
        )
    }

    pub fn batch_sample(
        &self,
        batch_size: usize,
    ) -> Option<(
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 2, Int>,
    )> {
        if (self.full & (batch_size > self.size)) | (!self.full & (batch_size > self.ptr)) {
            return None;
        }

        let sample_max = if self.full { self.size } else { self.ptr };

        // create the index slice
        let mut slice_indices = generate_rand_bool_vector(sample_max, batch_size);
        slice_indices.extend(vec![false; self.size - sample_max]);

        let indices = slice_indices
            .into_iter()
            .enumerate()
            .filter_map(|(i, keep)| if keep { Some(i as i32) } else { None })
            .collect::<Vec<i32>>();
        let len = indices.len();
        let indices: Tensor<B, 1, Int> =
            Tensor::from_ints(Data::new(indices, Shape::new([len])), &self.states.device());

        Some((
            self.states.clone().select(0, indices.clone()),
            self.actions.clone().select(0, indices.clone()),
            self.next_states.clone().select(0, indices.clone()),
            self.rewards.clone().select(0, indices.clone()),
            self.dones.clone().select(0, indices.clone()),
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::spaces::Space;
    use burn::backend::{wgpu::AutoGraphicsApi, Wgpu};

    use super::ReplayBuffer;
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;

    #[test]
    fn test_create_replay_buffer1() {
        // let device = burn::backend::wgpu::WgpuDevice::default();
        let observation_space = Space::Continuous {
            lows: vec![0.0; 5],
            highs: vec![1.0; 5],
        };
        let action_space = Space::Continuous {
            lows: vec![0.0; 5],
            highs: vec![1.0; 5],
        };

        let mut buffer =
            ReplayBuffer::<MyBackend>::new(100, observation_space.size(), action_space.size());

        buffer.add(
            observation_space.sample(),
            action_space.sample(),
            observation_space.sample(),
            0.5,
            false,
        );
    }

    #[test]
    fn test_create_replay_buffer2() {
        // let device = burn::backend::wgpu::WgpuDevice::default();
        let observation_space = Space::Discrete { size: 3 };
        let action_space = Space::Continuous {
            lows: vec![0.0; 5],
            highs: vec![1.0; 5],
        };

        let mut buffer =
            ReplayBuffer::<MyBackend>::new(100, observation_space.size(), action_space.size());

        buffer.add(
            observation_space.sample(),
            action_space.sample(),
            observation_space.sample(),
            0.5,
            false,
        );
    }

    #[test]
    fn test_create_replay_buffer3() {
        // let device = burn::backend::wgpu::WgpuDevice::default();
        let observation_space = Space::Continuous {
            lows: vec![0.0; 5],
            highs: vec![1.0; 5],
        };
        let action_space = Space::Discrete { size: 3 };

        let mut buffer =
            ReplayBuffer::<MyBackend>::new(100, observation_space.size(), action_space.size());

        buffer.add(
            observation_space.sample(),
            action_space.sample(),
            observation_space.sample(),
            0.5,
            false,
        );
    }

    #[test]
    fn test_create_replay_buffer4() {
        // let device = burn::backend::wgpu::WgpuDevice::default();
        let observation_space = Space::Discrete { size: 1 };
        let action_space = Space::Discrete { size: 3 };

        let mut buffer =
            ReplayBuffer::<MyBackend>::new(100, observation_space.size(), action_space.size());

        buffer.add(
            observation_space.sample(),
            action_space.sample(),
            observation_space.sample(),
            0.5,
            false,
        );
    }

    #[should_panic]
    #[test]
    fn test_batch_sample_before_ready() {
        let observation_space = Space::Discrete { size: 1 };
        let action_space = Space::Discrete { size: 3 };

        let buffer =
            ReplayBuffer::<MyBackend>::new(100, observation_space.size(), action_space.size());

        buffer.batch_sample(64);
    }

    #[test]
    fn test_batch_sample() {
        let observation_space = Space::Discrete { size: 1 };
        let action_space = Space::Discrete { size: 3 };

        let mut buffer =
            ReplayBuffer::<MyBackend>::new(100, observation_space.size(), action_space.size());

        for _ in 0..32 {
            buffer.add(
                observation_space.sample(),
                action_space.sample(),
                observation_space.sample(),
                0.5,
                false,
            );
        }

        let _ = buffer.batch_sample(4);
    }
}
