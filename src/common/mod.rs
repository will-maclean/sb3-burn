pub mod agent;
pub mod algorithm;
pub mod buffer;
pub mod callback;
pub mod distributions;
pub mod eval;
pub mod logger;
pub mod spaces;
pub mod to_tensor;
pub mod utils;

#[cfg(test)]
mod test {
    // burn sanity tests

    // use burn::{backend::{Autodiff, Wgpu}, tensor::Tensor};

    //TODO: tests with wgpu are panicing. re-enable when working again
    // #[test]
    // fn mean_can_debug_wgpu(){
    //     let t: Tensor<Wgpu, 1> = Tensor::from_floats([0.0, 1.0, 2.0], &Default::default());

    //     println!("{}", t);
    //     println!("{}", t.mean());

    //     let t: Tensor<Autodiff<Wgpu>, 1> = Tensor::from_floats([0.0, 1.0, 2.0], &Default::default());

    //     println!("{t}");
    //     println!("{}", t.mean());
    // }

    //TODO: seems to be a burn bug. Disabling for now so tests pass
    // #[test]
    // fn mean_can_debug_libtorch(){
    //     let t: Tensor<LibTorch, 1> = Tensor::from_floats([0.0, 1.0, 2.0], &Default::default());
    //     let mean: Tensor<LibTorch, 1> = t.clone().mean();

    //     println!("{t}");
    //     println!("{}", mean.clone());

    //     // let t: Tensor<Autodiff<LibTorch>, 1> = Tensor::from_floats([0.0, 1.0, 2.0], &Default::default());

    //     // println!("{t}");
    //     // println!("{}", t.mean());
    // }
}
