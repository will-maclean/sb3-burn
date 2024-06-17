use burn::prelude::*;

pub trait ToTensorF<const D: usize>: Clone {
    fn to_tensor<B: Backend>(self, device: &B::Device) -> Tensor<B, D>;
}

impl ToTensorF<1> for f32 {
    fn to_tensor<B: Backend>(self, device: &<B as Backend>::Device) -> Tensor<B, 1> {
        Tensor::from_floats([self], device)
    }
}

impl ToTensorF<1> for Vec<f32> {
    fn to_tensor<B: Backend>(self, device: &<B as Backend>::Device) -> Tensor<B, 1> {
        let n = self.len();

        Tensor::from_data(Data::new(self, Shape::new([n])).convert(), device)
    }
}

impl ToTensorF<2> for Vec<Vec<f32>> {
    fn to_tensor<B: Backend>(self, device: &<B as Backend>::Device) -> Tensor<B, 2> {
        let n0 = self.len();
        let n1 = self[0].len();
        let data: Vec<f32> = self.concat();

        Tensor::from_data(Data::new(data, Shape::new([n0, n1])).convert(), device)
    }
}

pub trait ToTensorI<const D: usize>: Clone {
    fn to_tensor<B: Backend>(self, device: &B::Device) -> Tensor<B, D, Int>;
}

impl ToTensorI<1> for usize {
    fn to_tensor<B: Backend>(self, device: &<B as Backend>::Device) -> Tensor<B, 1, Int> {
        Tensor::from_ints([self as i32], device)
    }
}

impl ToTensorI<1> for Vec<usize> {
    fn to_tensor<B: Backend>(self, device: &<B as Backend>::Device) -> Tensor<B, 1, Int> {
        let n = self.len();
        let data: Vec<i32> = self.into_iter().map(|x| x as i32).collect();

        Tensor::from_data(Data::new(data, Shape::new([n])).convert(), device)
    }
}

pub trait ToTensorB<const D: usize>: Clone {
    fn to_tensor<B: Backend>(self, device: &B::Device) -> Tensor<B, D, Bool>;
}

impl ToTensorB<1> for bool {
    fn to_tensor<B: Backend>(self, device: &<B as Backend>::Device) -> Tensor<B, 1, Bool> {
        Tensor::<B, 1, Int>::from_data(
            Data::new(vec![self as i32], Shape::new([1])).convert(),
            device,
        )
        .bool()
    }
}

impl ToTensorB<1> for Vec<bool> {
    fn to_tensor<B: Backend>(self, device: &<B as Backend>::Device) -> Tensor<B, 1, Bool> {
        let n = self.len();

        Tensor::<B, 1, Int>::from_data(Data::new(self, Shape::new([n])).convert(), device).bool()
    }
}
