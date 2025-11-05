#![recursion_limit = "256"]
use std::path::PathBuf;

use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{ElementConversion, Tensor},
};
use sb3_burn::{
    common::spaces::{BoxSpace, Space},
    sac::models::QModelSet,
};

fn main() {
    type B = Autodiff<Wgpu>;

    let n_critics = 2;

    let train_device = WgpuDevice::default();

    let action_space = BoxSpace::from(([0.0].to_vec(), [1.0].to_vec()));
    let obs_space = BoxSpace::from(([0.0].to_vec(), [1.0].to_vec()));

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
    //     64,
    //     &train_device,
    // );

    let save_dir = PathBuf::from("weights/sac_probe3/");

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let trained_qs = qs
        .load_file(save_dir.join("qs_model"), &recorder, &train_device)
        .unwrap();

    for obs in [-1.0, 0.0, 1.0] {
        for act in [-1.0, 0.0, 1.0] {
            let fresh_obs: Tensor<B, 2> = Tensor::<B, 1>::from_floats([obs], &train_device)
                .unsqueeze()
                .require_grad();
            let fresh_action: Tensor<B, 2> = Tensor::<B, 1>::from_floats([act], &train_device)
                .unsqueeze()
                .require_grad();

            let q_vals = trained_qs.q_from_actions(fresh_obs, fresh_action.clone());
            let q_min = Tensor::cat(q_vals, 1).min_dim(1);

            // calculate the grads of q_min w.r.t. fresh_action
            let grads = q_min.clone().mean().backward();
            if let Some(grad) = fresh_action.grad(&grads) {
                let abs = grad.abs();
                println!(
                    "obs={obs}, act={act}. Q(s,a)={q_min}. ∂Q/∂a mean/max: {:?}",
                    (
                        abs.clone().mean().into_scalar().elem::<f32>(),
                        abs.max().into_scalar().elem::<f32>()
                    )
                );
            } else {
                println!("fresh_action gradient not retained");
            }
        }
    }
}
