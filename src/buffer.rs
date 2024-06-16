use burn::prelude::*;

use crate::{spaces::SpaceSample, utils::generate_rand_bool_vector};

pub struct ReplayBufferSlice<B: Backend> {
    state: Tensor<B, 1>,
    action: Tensor<B, 1>,
    next_state: Tensor<B, 1>,
    reward: f32,
    terminated: bool,
    truncated: bool,
}

pub struct BatchedReplayBufferSliceT<B: Backend> {
    pub states: Tensor<B, 2>,
    pub actions: Tensor<B, 2>,
    pub next_states: Tensor<B, 2>,
    pub rewards: Tensor<B, 2>,
    pub terminated: Tensor<B, 2, Int>,
    pub truncated: Tensor<B, 2, Int>,
}

impl<B: Backend> BatchedReplayBufferSliceT<B> {
    pub fn to_device(&mut self, device: &B::Device) {
        self.states = self.states.clone().to_device(device);
        self.actions = self.actions.clone().to_device(device);
        self.next_states = self.next_states.clone().to_device(device);
        self.rewards = self.rewards.clone().to_device(device);
        self.terminated = self.terminated.clone().to_device(device);
        self.truncated = self.truncated.clone().to_device(device);
    }
}

pub struct ReplayBuffer<B: Backend> {
    states: Tensor<B, 2>,
    actions: Tensor<B, 2>,
    next_states: Tensor<B, 2>,
    rewards: Tensor<B, 2>,
    terminated: Tensor<B, 2, Int>,
    truncated: Tensor<B, 2, Int>,
    size: usize,
    full: bool,
    ptr: usize,
}

impl<B: Backend> ReplayBuffer<B> {
    //TODO: probably want to take in spaces instead of usize's
    pub fn new(size: usize, state_dim: usize, action_dim: usize, device: &B::Device) -> Self {
        Self {
            states: Tensor::<B, 2>::zeros([size, state_dim], &device),
            actions: Tensor::<B, 2>::zeros([size, action_dim], &device),
            next_states: Tensor::<B, 2>::zeros([size, state_dim], &device),
            rewards: Tensor::<B, 2>::zeros([size, 1], &device),
            terminated: Tensor::<B, 2, Int>::empty([size, 1], &device),
            truncated: Tensor::<B, 2, Int>::empty([size, 1], &device),
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
            reward: self
                .rewards
                .clone()
                .slice([idx..idx + 1])
                .squeeze::<1>(0)
                .into_scalar()
                .elem(),
            terminated: self
                .terminated
                .clone()
                .slice([idx..idx + 1])
                .squeeze::<1>(0)
                .into_scalar()
                .elem::<i32>()
                != 0,
            truncated: self
                .truncated
                .clone()
                .slice([idx..idx + 1])
                .squeeze::<1>(0)
                .into_scalar()
                .elem::<i32>()
                != 0,
        })
    }

    pub fn add_slice(&mut self, item: ReplayBufferSlice<B>) {
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
        self.terminated = self.terminated.clone().slice_assign(
            [self.ptr..self.ptr + 1],
            Tensor::<B, 1, Int>::from_ints([item.terminated as i32], &self.terminated.device())
                .unsqueeze_dim(0),
        );
        self.truncated = self.truncated.clone().slice_assign(
            [self.ptr..self.ptr + 1],
            Tensor::<B, 1, Int>::from_ints([item.truncated as i32], &self.truncated.device())
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

    pub fn add(
        &mut self,
        state: SpaceSample,
        action: SpaceSample,
        next_state: SpaceSample,
        reward: f32,
        terminated: bool,
        truncated: bool,
    ) {
        self.add_processed(
            state.to_train_tensor(),
            action.to_tensor(),
            next_state.to_train_tensor(),
            reward,
            terminated,
            truncated,
        )
    }

    pub fn batch_sample(&self, batch_size: usize) -> Option<BatchedReplayBufferSliceT<B>> {
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

        Some(BatchedReplayBufferSliceT {
            states: self.states.clone().select(0, indices.clone()),
            actions: self.actions.clone().select(0, indices.clone()),
            next_states: self.next_states.clone().select(0, indices.clone()),
            rewards: self.rewards.clone().select(0, indices.clone()),
            terminated: self.terminated.clone().select(0, indices.clone()),
            truncated: self.truncated.clone().select(0, indices.clone()),
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::spaces::Space;
    use burn::backend::NdArray;

    use super::ReplayBuffer;
    type MyBackend = NdArray;

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

        let device = Default::default();

        let mut buffer = ReplayBuffer::<MyBackend>::new(
            100,
            observation_space.size(),
            action_space.size(),
            &device,
        );

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
    fn test_create_replay_buffer2() {
        // let device = burn::backend::wgpu::WgpuDevice::default();
        let observation_space = Space::Discrete { size: 3 };
        let action_space = Space::Continuous {
            lows: vec![0.0; 5],
            highs: vec![1.0; 5],
        };

        let device = Default::default();

        let mut buffer = ReplayBuffer::<MyBackend>::new(
            100,
            observation_space.size(),
            action_space.size(),
            &device,
        );

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
    fn test_create_replay_buffer3() {
        // let device = burn::backend::wgpu::WgpuDevice::default();
        let observation_space = Space::Continuous {
            lows: vec![0.0; 5],
            highs: vec![1.0; 5],
        };
        let action_space = Space::Discrete { size: 3 };

        let device = Default::default();

        let mut buffer = ReplayBuffer::<MyBackend>::new(
            100,
            observation_space.size(),
            action_space.size(),
            &device,
        );

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
    fn test_create_replay_buffer4() {
        // let device = burn::backend::wgpu::WgpuDevice::default();
        let observation_space = Space::Discrete { size: 1 };
        let action_space = Space::Discrete { size: 3 };

        let device = Default::default();

        let mut buffer = ReplayBuffer::<MyBackend>::new(
            100,
            observation_space.size(),
            action_space.size(),
            &device,
        );

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
        let observation_space = Space::Discrete { size: 1 };
        let action_space = Space::Discrete { size: 3 };

        let device = Default::default();

        let buffer = ReplayBuffer::<MyBackend>::new(
            100,
            observation_space.size(),
            action_space.size(),
            &device,
        );

        let sample = buffer.batch_sample(64);

        assert!(sample.is_none());
    }

    #[test]
    fn test_batch_sample() {
        let observation_space = Space::Discrete { size: 1 };
        let action_space = Space::Discrete { size: 3 };

        let device = Default::default();

        let mut buffer = ReplayBuffer::<MyBackend>::new(
            100,
            observation_space.size(),
            action_space.size(),
            &device,
        );

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
