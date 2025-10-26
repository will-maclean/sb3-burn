use burn::{
    module::Param,
    nn::{conv::Conv2d, Linear},
    tensor::{backend::Backend, Tensor},
};

// From https://github.com/benbaarber/rl/blob/main/examples/dqn_snake/model.rs#L89, thanks benbaarber :)
fn soft_update_tensor<B: Backend, const D: usize>(
    from: &Param<Tensor<B, D>>,
    to: Param<Tensor<B, D>>,
    tau: f32,
) -> Param<Tensor<B, D>> {
    to.map(|tensor| tensor * (1.0 - tau) + from.val() * tau)
}

pub fn update_linear<B: Backend>(from: &Linear<B>, to: Linear<B>, tau: Option<f32>) -> Linear<B> {
    assert_eq!(from.weight.shape(), to.weight.shape());

    let mut result = from.clone(); // already does a hard copy

    match tau {
        Some(tau) => {
            result.weight = soft_update_tensor(&from.weight, to.weight, tau);

            match from.bias.clone() {
                Some(from_bias) => {
                    result.bias = Some(soft_update_tensor(&from_bias, to.bias.unwrap(), tau));
                }
                None => todo!(),
            }
        }
        None => {}
    }

    result
}

pub fn update_conv2d<B: Backend>(from: &Conv2d<B>, to: Conv2d<B>, tau: Option<f32>) -> Conv2d<B> {
    assert_eq!(from.weight.shape(), to.weight.shape());

    let mut result = from.clone(); // already does a hard copy

    match tau {
        Some(tau) => {
            result.weight = soft_update_tensor(&from.weight, to.weight, tau);

            match from.bias.clone() {
                Some(from_bias) => {
                    result.bias = Some(soft_update_tensor(&from_bias, to.bias.unwrap(), tau));
                }
                None => todo!(),
            }
        }
        None => {}
    }

    result
}

#[cfg(test)]
mod test {
    use burn::{
        backend::NdArray,
        module::{Module, Param},
        nn::{Linear, LinearConfig},
        tensor::{backend::Backend, Float, Tensor},
    };

    use crate::common::{agent::Policy, utils::module_update::soft_update_tensor};

    use super::update_linear;

    #[derive(Module, Debug)]
    struct LinearPolicy<B: Backend> {
        layer: Linear<B>,
    }

    impl<B: Backend> LinearPolicy<B> {
        fn new(in_size: usize, out_size: usize, device: &B::Device) -> Self {
            Self {
                layer: LinearConfig::new(in_size, out_size).init(device),
            }
        }
    }

    impl<B: Backend> Policy<B> for LinearPolicy<B> {
        fn update(&mut self, from: &Self, tau: Option<f32>) {
            self.layer = update_linear(&from.layer, self.layer.clone(), tau);
        }
    }

    #[test]
    fn test_hard_update() {
        type B = NdArray;
        let mut a = LinearPolicy::<B>::new(3, 4, &Default::default());
        let b = LinearPolicy::<B>::new(3, 4, &Default::default());

        a.update(&b, None);
    }

    #[test]
    fn test_soft_update() {
        type B = NdArray;
        let mut a = LinearPolicy::<B>::new(3, 4, &Default::default());
        let b = LinearPolicy::<B>::new(3, 4, &Default::default());

        a.update(&b, Some(0.05));
    }

    #[should_panic]
    #[test]
    fn test_soft_update_bad() {
        type B = NdArray;
        let mut a = LinearPolicy::<B>::new(3, 4, &Default::default());
        let b = LinearPolicy::<B>::new(4, 4, &Default::default());

        a.update(&b, Some(0.05));
    }

    #[should_panic]
    #[test]
    fn test_hard_update_bad() {
        type B = NdArray;
        let mut a = LinearPolicy::<B>::new(3, 4, &Default::default());
        let b = LinearPolicy::<B>::new(4, 4, &Default::default());

        a.update(&b, None);
    }

    #[test]
    fn test_soft_update_value() {
        type B = NdArray;
        let from = Param::from_tensor(Tensor::from_floats([1.0], &Default::default()));
        let to = Param::from_tensor(Tensor::from_floats([0.0], &Default::default()));
        let tau = 0.05;

        let new_to: Param<Tensor<B, 1, Float>> = soft_update_tensor(&from, to, tau);

        let new_to_f: f32 = new_to.val().into_scalar();

        let diff = tau - new_to_f;

        assert!(
            diff < 1e-6,
            "tau update value gave unexpected result. Gave {new_to_f}, expected {tau}"
        );
    }
}
