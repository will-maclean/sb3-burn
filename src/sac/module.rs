use burn::{module::Module, nn, tensor::{backend::Backend, Tensor}};


pub struct QVals<B: Backend> {
    pub q1: Tensor<B, 2>,
    pub q2: Tensor<B, 2>,
}

pub trait SACNet<B, O, A> : Module<B>
where
    B: Backend,
    O: Clone,
{
    fn q_vals(&self, obs: Vec<O>, acts: Vec<A>, device: &B::Device) -> QVals<B>;
    fn pi(&self, obs: Vec<O>, device: &B::Device) -> Tensor<B, 2>;
}

#[derive(Debug, Module)]
pub struct MLP<B: Backend> {
    l1: nn::Linear<B>,
    l2: nn::Linear<B>,
    activ: nn::Relu,
}

impl<B: Backend> MLP<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.l1.forward(x);
        let x = self.activ.forward(x);
        self.l2.forward(x)
    }
}