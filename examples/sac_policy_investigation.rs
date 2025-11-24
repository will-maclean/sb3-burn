use std::path::PathBuf;

use burn::{
    backend::{ndarray::NdArrayDevice, Autodiff, NdArray},
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{ElementConversion, Tensor},
};
use sb3_burn::{
    common::spaces::{BoxSpace, Space},
    sac::models::PiModel,
};

fn main() {
    type B = Autodiff<NdArray>;

    let train_device = NdArrayDevice::default();

    let action_space = BoxSpace::from(([0.0].to_vec(), [1.0].to_vec()));
    let obs_space = BoxSpace::from(([0.0].to_vec(), [1.0].to_vec()));

    let pi = PiModel::<B>::new(
        obs_space.shape().len(),
        action_space.shape().len(),
        64,
        &train_device,
    );

    let save_dir = PathBuf::from("weights/sac_probe3/");

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let mut trained_pi = pi
        .load_file(save_dir.join("pi_model"), &recorder, &train_device)
        .unwrap();

    let obs: Tensor<B, 2> = Tensor::<B, 1>::from_floats([0.5], &train_device)
        .unsqueeze()
        .require_grad();

    let (act, _) = trained_pi.act_log_prob(obs.clone());
    let grads = act.clone().mean().backward();
    if let Some(grad) = obs.grad(&grads) {
        println!(
            "obs={obs}, act={act}. ∂a/∂s min/mean/max: {:?}",
            (
                grad.clone().min().into_scalar().elem::<f32>(),
                grad.clone().mean().into_scalar().elem::<f32>(),
                grad.max().into_scalar().elem::<f32>()
            )
        );
    } else {
        println!("Houston...");
    }
}
