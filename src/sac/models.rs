use burn::{module::Module, prelude::Backend, tensor::Tensor};

use crate::common::{
    agent::Policy,
    distributions::action_distribution::{ActionDistribution, SquashedDiagGaussianDistribution},
    utils::modules::MLP,
};

#[derive(Debug, Module)]
pub struct PiModel<B: Backend> {
    mlp: MLP<B>,
    dist: SquashedDiagGaussianDistribution<B>,
    n_actions: usize,
}

impl<B: Backend> PiModel<B> {
    pub fn new(obs_size: usize, n_actions: usize, hidden_size: usize, device: &B::Device) -> Self {
        Self {
            mlp: MLP::new(&[obs_size, hidden_size, hidden_size].to_vec(), device, None),
            dist: SquashedDiagGaussianDistribution::new(hidden_size, n_actions, device, 1e-6),
            n_actions,
        }
    }
}

impl<B: Backend> PiModel<B> {
    pub fn act(&mut self, obs: &Tensor<B, 1>, deterministic: bool) -> Tensor<B, 1> {
        let latent = self.mlp.forward(obs.clone().unsqueeze());

        // println!("Obs: {obs}");

        self.dist
            .actions_from_obs(latent, deterministic)
            .squeeze_dim(0)
    }

    pub fn act_log_prob(&mut self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let latent = self.mlp.forward(obs);
        self.dist.actions_from_obs_with_log_probs(latent, false)
    }
}

#[derive(Debug, Module)]
pub struct QModel<B: Backend> {
    mlp: MLP<B>,
}

impl<B: Backend> QModel<B> {
    pub fn new(obs_size: usize, n_actions: usize, hidden_size: usize, device: &B::Device) -> Self {
        let mlp = MLP::new(
            &[obs_size + n_actions, hidden_size, hidden_size, 1].to_vec(),
            device,
            None,
        );

        Self { mlp: mlp }
    }
}

impl<B: Backend> Policy<B> for QModel<B> {
    fn update(&mut self, from: &Self, tau: Option<f32>) {
        self.mlp.update(&from.mlp, tau);
    }
}

impl<B: Backend> QModel<B> {
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.mlp.forward(x)
    }

    pub fn q_from_actions(&self, obs: Tensor<B, 2>, actions: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = Tensor::cat(Vec::from([obs, actions]), 1);

        self.forward(x)
    }
}

#[derive(Debug, Module)]
pub struct QModelSet<B: Backend> {
    qs: Vec<QModel<B>>,
}

impl<B: Backend> QModelSet<B> {
    pub fn new(
        obs_size: usize,
        n_actions: usize,
        hidden_size: usize,
        device: &B::Device,
        n_critics: usize,
    ) -> Self {
        let mut qs = Vec::new();

        for _ in 0..n_critics {
            qs.push(QModel::new(obs_size, n_actions, hidden_size, device));
        }

        Self { qs }
    }
    pub fn q_from_actions(&self, obs: Tensor<B, 2>, actions: Tensor<B, 2>) -> Vec<Tensor<B, 2>> {
        self.qs
            .iter()
            .map(|q| q.q_from_actions(obs.clone(), actions.clone()))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.qs.len()
    }
}

impl<B: Backend> Policy<B> for QModelSet<B> {
    fn update(&mut self, from: &Self, tau: Option<f32>) {
        for i in 0..self.qs.len() {
            self.qs[i].update(&from.qs[i], tau);
        }
    }
}

#[cfg(test)]
mod test {
    use burn::{
        backend::{ndarray::NdArrayDevice, Autodiff, NdArray},
        module::{Module, Param},
        optim::{GradientsParams, Optimizer, SgdConfig},
        prelude::Backend,
        Tensor,
    };

    #[derive(Module, Debug)]
    struct TestModel<B: Backend> {
        pub x1: Param<Tensor<B, 2>>,
        pub x2: Param<Tensor<B, 2>>,
    }

    impl<B: Backend> TestModel<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                x1: Param::from_tensor(Tensor::<B, 2>::from_floats(
                    [
                        [0.3227, 0.0161],
                        [0.0943, 0.5760],
                        [0.5722, 0.1274],
                        [0.0710, 0.5388],
                    ],
                    device,
                )),
                x2: Param::from_tensor(Tensor::<B, 2>::from_floats(
                    [
                        [0.1726, 0.1411],
                        [0.8992, 0.3996],
                        [0.2777, 0.8487],
                        [0.2426, 0.8821],
                    ],
                    device,
                )),
            }
        }

        fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
            let y1 = x.clone().matmul(self.x1.val());
            let y2 = x.clone().matmul(self.x2.val());

            Tensor::cat(vec![y1, y2], 1).min_dim(1)
        }
    }

    #[test]
    fn test_min_dim_grad() {
        // Sanity check that backprop through a min_dim(1)
        // does the same thing as in PyTorch
        type B = Autodiff<NdArray>;
        let dev = NdArrayDevice::default();

        let mut model = TestModel::new(&dev);

        let data = Tensor::<B, 2>::from_floats(
            [
                [0.8419, 0.9508, 0.8376, 0.2873],
                [0.0637, 0.1694, 0.7541, 0.3553],
                [0.3960, 0.2973, 0.7402, 0.3870],
            ],
            &dev,
        );

        let y_batched = model.forward(data);
        println!("y_batched={:?}", y_batched);

        let y_batched_expected = Tensor::<B, 2>::from_floats([[0.8227], [0.3861], [0.4804]], &dev);

        let y = y_batched.clone().mean();

        let mut sgd = SgdConfig::new().init();

        let grads = y.backward();

        let grads = GradientsParams::from_grads(grads, &model);
        model = sgd.step(1.0, model, grads);

        assert!(y_batched.all_close(y_batched_expected, None, Some(1e-3)));

        let x1_after_expected = Tensor::<B, 2>::from_floats(
            [
                [0.3227, -0.4178],
                [0.0943, 0.1035],
                [0.5722, -0.6499],
                [0.0710, 0.1957],
            ],
            &dev,
        );

        let x2_after_expected = Tensor::<B, 2>::from_floats(
            [
                [0.1726, 0.1411],
                [0.8992, 0.3996],
                [0.2777, 0.8487],
                [0.2426, 0.8821],
            ],
            &dev,
        );

        assert!(model
            .x1
            .val()
            .all_close(x1_after_expected, None, Some(1e-3)));
        assert!(model
            .x2
            .val()
            .all_close(x2_after_expected, None, Some(1e-3)));
    }
}
