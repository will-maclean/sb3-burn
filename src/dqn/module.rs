use std::fmt::Debug;

use burn::{
    module::Module,
    nn,
    tensor::{
        activation::{relu, sigmoid},
        backend::Backend,
        Tensor,
    },
};

use crate::common::{
    agent::Policy,
    spaces::Space,
    to_tensor::ToTensorF,
    utils::{
        module_update::{update_conv2d, update_linear},
        vec_usize_to_one_hot,
    },
};

pub trait DQNNet<B: Backend, OS: Clone>: Policy<B> {
    fn forward(
        &self,
        obs: Vec<OS>,
        obs_space: Box<dyn Space<OS>>,
        device: &B::Device,
    ) -> Tensor<B, 2>;
}

#[derive(Module, Debug)]
pub struct LinearAdvDQNNet<B: Backend> {
    l1: nn::Linear<B>,
    l2: nn::Linear<B>,
    l3: nn::Linear<B>,
    adv: nn::Linear<B>,
}

impl<B: Backend> LinearAdvDQNNet<B> {
    pub fn init(device: &B::Device, obs_size: usize, act_size: usize, hidden_size: usize) -> Self {
        Self {
            l1: nn::LinearConfig::new(obs_size, hidden_size).init(device),
            l2: nn::LinearConfig::new(hidden_size, hidden_size).init(device),
            l3: nn::LinearConfig::new(hidden_size, act_size).init(device),
            adv: nn::LinearConfig::new(hidden_size, 1).init(device),
        }
    }

    fn _forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = sigmoid(self.l1.forward(x));
        let x = sigmoid(self.l2.forward(x));
        self.adv.forward(x.clone()) - self.l3.forward(x)
    }
}

impl<B: Backend> DQNNet<B, usize> for LinearAdvDQNNet<B> {
    fn forward(
        &self,
        state: Vec<usize>,
        obs_space: Box<dyn Space<usize>>,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let state = vec_usize_to_one_hot(state, obs_space.shape(), device);
        self._forward(state)
    }
}

impl<B: Backend> DQNNet<B, Vec<f32>> for LinearAdvDQNNet<B> {
    fn forward(
        &self,
        state: Vec<Vec<f32>>,
        _obs_space: Box<dyn Space<Vec<f32>>>,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let state = state.to_tensor(device);
        self._forward(state)
    }
}

impl<B: Backend> Policy<B> for LinearAdvDQNNet<B> {
    fn update(&mut self, from: &Self, tau: Option<f32>) {
        self.l1 = update_linear(&from.l1, self.l1.clone(), tau);
        self.l2 = update_linear(&from.l2, self.l2.clone(), tau);
        self.l3 = update_linear(&from.l3, self.l3.clone(), tau);
        self.adv = update_linear(&from.adv, self.adv.clone(), tau);
    }
}

#[derive(Module, Debug)]
pub struct LinearDQNNet<B: Backend> {
    l1: nn::Linear<B>,
    l2: nn::Linear<B>,
    l3: nn::Linear<B>,
}

impl<B: Backend> LinearDQNNet<B> {
    pub fn init(device: &B::Device, obs_size: usize, act_size: usize, hidden_size: usize) -> Self {
        Self {
            l1: nn::LinearConfig::new(obs_size, hidden_size).init(device),
            l2: nn::LinearConfig::new(hidden_size, hidden_size).init(device),
            l3: nn::LinearConfig::new(hidden_size, act_size).init(device),
        }
    }

    pub fn _forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = relu(self.l1.forward(x));
        let x = relu(self.l2.forward(x));
        self.l3.forward(x)
    }
}

impl<B: Backend> DQNNet<B, Tensor<B, 1>> for LinearDQNNet<B> {
    fn forward(
        &self,
        state: Vec<Tensor<B, 1>>,
        _obs_space: Box<dyn Space<Tensor<B, 1>>>,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let state = Tensor::stack(state, 0).to_device(device);
        self._forward(state)
    }
}

impl<B: Backend> DQNNet<B, Vec<f32>> for LinearDQNNet<B> {
    fn forward(
        &self,
        state: Vec<Vec<f32>>,
        _obs_space: Box<dyn Space<Vec<f32>>>,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let state = state.to_tensor(device);
        self._forward(state)
    }
}

impl<B: Backend> DQNNet<B, usize> for LinearDQNNet<B> {
    fn forward(
        &self,
        state: Vec<usize>,
        obs_space: Box<dyn Space<usize>>,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let state = vec_usize_to_one_hot(state, obs_space.shape(), device);
        self._forward(state)
    }
}

impl<B: Backend> Policy<B> for LinearDQNNet<B> {
    fn update(&mut self, from: &Self, tau: Option<f32>) {
        self.l1 = update_linear(&from.l1, self.l1.clone(), tau);
        self.l2 = update_linear(&from.l2, self.l2.clone(), tau);
        self.l3 = update_linear(&from.l3, self.l3.clone(), tau);
    }
}

#[derive(Module, Debug)]
pub struct ConvDQNNet<B: Backend> {
    c1: nn::conv::Conv2d<B>,
    c2: nn::conv::Conv2d<B>,
    l1: nn::Linear<B>,
    l2: nn::Linear<B>,
}

impl<B: Backend> ConvDQNNet<B> {
    pub fn init(
        device: &B::Device,
        obs_shape: Tensor<B, 3>,
        act_size: usize,
        hidden_size: usize,
    ) -> Self {
        //TODO: take in channels, calculate linear input size

        Self {
            c1: nn::conv::Conv2dConfig::new([obs_shape.shape().dims[0], 2], [3, 3]).init(device),
            c2: nn::conv::Conv2dConfig::new([2, 3], [3, 3]).init(device),
            l1: nn::LinearConfig::new(432, hidden_size).init(device),
            l2: nn::LinearConfig::new(hidden_size, act_size).init(device),
        }
    }
}

impl<B: Backend> DQNNet<B, Tensor<B, 3>> for ConvDQNNet<B> {
    fn forward(
        &self,
        state: Vec<Tensor<B, 3>>,
        _obs_space: Box<dyn Space<Tensor<B, 3>>>,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let state = Tensor::stack(state, 0).to_device(device);
        let x = relu(self.c1.forward(state));
        let x = relu(self.c2.forward(x));
        let x = x.flatten::<2>(1, 3);
        let x = relu(self.l1.forward(x));
        self.l2.forward(x)
    }
}

impl<B: Backend> Policy<B> for ConvDQNNet<B> {
    fn update(&mut self, from: &Self, tau: Option<f32>) {
        self.l1 = update_linear(&from.l1, self.l1.clone(), tau);
        self.l2 = update_linear(&from.l2, self.l2.clone(), tau);
        self.c1 = update_conv2d(&from.c1, self.c1.clone(), tau);
        self.c2 = update_conv2d(&from.c2, self.c2.clone(), tau);
    }
}

#[cfg(test)]
mod test {
    use burn::{
        backend::NdArray,
        tensor::{Shape, Tensor},
    };

    use crate::common::spaces::{BoxSpace, Discrete, Space};

    use super::{ConvDQNNet, DQNNet, LinearAdvDQNNet, LinearDQNNet};

    #[test]
    fn test_broadcast_sanity() {
        let a = [0.0, 1.0, 2.0];
        let b = [1.0];

        let a_t = Tensor::<NdArray, 1>::from_floats(a, &Default::default());
        let b_t = Tensor::<NdArray, 1>::from_floats(b, &Default::default());

        let c_t = a_t.clone() + b_t.clone();
        let c: Vec<f32> = c_t.into_data().to_vec().unwrap();

        assert_eq!(c, vec![1.0, 2.0, 3.0]);

        let c_t = a_t - b_t;
        let c: Vec<f32> = c_t.into_data().to_vec().unwrap();

        assert_eq!(c, vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_linear_net_usize_usize() {
        let mut obs_space = Discrete::from(3);

        let dqn: LinearDQNNet<NdArray> =
            LinearDQNNet::init(&Default::default(), obs_space.shape(), 1, 2);

        dqn.forward(
            vec![obs_space.sample()],
            Box::new(obs_space),
            &Default::default(),
        );
    }

    #[test]
    fn test_linear_net_vecf32_usize() {
        let mut obs_space = BoxSpace::from((vec![0.0, 1.0], vec![0.5, 8.1]));

        let dqn: LinearDQNNet<NdArray> =
            LinearDQNNet::init(&Default::default(), obs_space.shape().len(), 1, 2);

        dqn.forward(
            vec![obs_space.sample()],
            Box::new(obs_space),
            &Default::default(),
        );
    }

    #[test]
    fn test_linear_adv_net_usize_usize() {
        let mut obs_space = Discrete::from(3);

        let dqn: LinearAdvDQNNet<NdArray> =
            LinearAdvDQNNet::init(&Default::default(), obs_space.shape(), 1, 2);

        dqn.forward(
            vec![obs_space.sample()],
            Box::new(obs_space),
            &Default::default(),
        );
    }

    #[test]
    fn test_linear_adv_net_vecf32_usize() {
        let mut obs_space = BoxSpace::from((vec![0.0, 1.0], vec![0.5, 8.1]));

        let dqn: LinearAdvDQNNet<NdArray> =
            LinearAdvDQNNet::init(&Default::default(), obs_space.shape().len(), 1, 2);

        dqn.forward(
            vec![obs_space.sample()],
            Box::new(obs_space),
            &Default::default(),
        );
    }

    #[test]
    fn test_conv_usize() {
        let shape = Shape::new([3, 16, 16]);
        let low: Tensor<NdArray, 3> = Tensor::zeros(shape.clone(), &Default::default());
        let high: Tensor<NdArray, 3> = Tensor::zeros(shape, &Default::default());
        let mut obs_space = BoxSpace::from((low, high));

        let dqn: ConvDQNNet<NdArray> =
            ConvDQNNet::init(&Default::default(), obs_space.shape(), 1, 2);

        dqn.forward(
            vec![obs_space.sample()],
            Box::new(obs_space),
            &Default::default(),
        );
    }
}
