use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
    grad_clipping::GradientClippingConfig,
    optim::{adaptor::OptimizerAdaptor, Adam, AdamConfig},
    tensor::{ElementConversion, Tensor},
    train,
};
use sb3_burn::{
    common::{
        spaces::{BoxSpace, Space},
        to_tensor::ToTensorF,
    },
    sac::models::{PiModel, QModelSet},
};

fn main() {
    type B = Autodiff<Wgpu>;

    let n_critics = 2;

    let train_device = WgpuDevice::default();

    let mut action_space = BoxSpace::from(([0.0].to_vec(), [1.0].to_vec()));
    let mut obs_space = BoxSpace::from(([0.0].to_vec(), [1.0].to_vec()));

    // let config_optimizer =
    //     AdamConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(10.0)));

    // let pi_optim: OptimizerAdaptor<Adam, PiModel<B>, B> = config_optimizer.init();

    let qs: QModelSet<B> = QModelSet::new(
        obs_space.shape().len(),
        action_space.shape().len(),
        64,
        &train_device,
        n_critics,
    );

    // let q_optim: OptimizerAdaptor<Adam, QModelSet<B>, B> = config_optimizer.init();

    // let pi = PiModel::<B>::new(
    //     obs_space.shape().len(),
    //     action_space.shape().len(),
    //     4,
    //     &train_device,
    // );

    // check gradients on an unoptimised critic
    let fresh_obs: Tensor<B, 2> = obs_space.sample().to_tensor(&train_device).unsqueeze().require_grad();
    let fresh_action: Tensor<B, 2> = action_space.sample().to_tensor(&train_device).unsqueeze().require_grad();

    let q_vals = qs.q_from_actions(fresh_obs, fresh_action.clone());
    let q_min = Tensor::cat(q_vals, 1).min_dim(1);

    // calculate the grads of q_min w.r.t. fresh_action
    let grads = q_min.clone().mean().backward();
    if let Some(grad) = fresh_action.grad(&grads) {
        let abs = grad.abs();
        println!(
            "∂Q/∂a mean/max: {:?}",
            (
                abs.clone().mean().into_scalar().elem::<f32>(),
                abs.max().into_scalar().elem::<f32>()
            )
        );
    } else {
        println!("fresh_action gradient not retained");
    }
}
