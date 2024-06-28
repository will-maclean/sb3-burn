use burn::{module::Module, nn::{Linear, LinearConfig}, tensor::{activation::relu, backend::Backend, Tensor}};

use crate::common::agent::Policy;

use super::module_update::update_linear;

#[derive(Debug, Module)]
pub struct MLP<B: Backend>{
    layers: Vec<Linear<B>>
}

impl<B: Backend> MLP<B>{
    pub fn new(sizes: &Vec<usize>, device: &B::Device) -> Self{
        let mut layers = Vec::new();

        for i in 0..sizes.len() - 1 {
            layers.push(
                LinearConfig::new(sizes[i], sizes[i+1]).init(device)
            )
        }

        Self { 
            layers 
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2>{
        let mut x_ = x;

        if self.layers.len() > 1 {
            for i in 0..self.layers.len()-1{
                x_ = self.layers[i].forward(x_);
                x_ = relu(x_);
            }
        }

        self.layers[self.layers.len()-1].forward(x_)
    }
}

impl<B: Backend> Policy<B> for MLP<B>{
    fn update(&mut self, from: &Self, tau: Option<f32>) {
        for i in 0..self.layers.len(){
            self.layers[i] = update_linear(&from.layers[i], self.layers[i].clone(), tau);
        }
    }
}

#[cfg(test)]
mod test{
    use burn::{backend::NdArray, tensor::{Distribution, Shape, Tensor}};

    use crate::common::agent::Policy;

    use super::MLP;

    #[test]
    fn test_create_mlp_one_layer(){
        type Backend = NdArray;
        
        let mut model = MLP::<NdArray>::new(&vec![5, 6], &Default::default());
        let dummy_forward: Tensor<Backend, 2> = Tensor::random(
            Shape::new([3, 5]), 
            Distribution::Normal(0.0, 1.0), 
            &Default::default(),
        );

        model.forward(dummy_forward);

        model.update(&model.clone(), None);
    }

    #[test]
    fn test_create_mlp_multi_layer(){
        type Backend = NdArray;
        
        let mut model = MLP::<NdArray>::new(&vec![5, 6, 7], &Default::default());
        let dummy_forward: Tensor<Backend, 2> = Tensor::random(
            Shape::new([3, 5]), 
            Distribution::Normal(0.0, 1.0), 
            &Default::default(),
        );

        model.forward(dummy_forward);

        model.update(&model.clone(), None);
    }
}