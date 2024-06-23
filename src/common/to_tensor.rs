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

#[cfg(test)]
mod test {
    use burn::{backend::NdArray, tensor::{Bool, Int, Tensor}};

    use crate::common::to_tensor::{ToTensorB, ToTensorI};

    use super::ToTensorF;

    #[test]
    fn test_to_tensor_f32(){
        let d: f32 = 1.1;
        let t: Tensor<NdArray, 1> = d.to_tensor(&Default::default());
        
        assert_eq!(t.shape().dims.len(), 1);
        assert_eq!(t.shape().dims, [1]);
        assert_eq!(t.into_scalar(), d);
    }

    #[test]
    fn test_to_tensor_vec_f32(){
        let d: Vec<f32> = vec![1.1, 2.2];
        let t: Tensor<NdArray, 1> = d.to_tensor(&Default::default());
        
        assert_eq!(t.shape().dims.len(), 1);
        assert_eq!(t.shape().dims, [2]);
    }

    #[test]
    fn test_to_tensor_vec_vec_f32(){
        let d: Vec<Vec<f32>> = vec![vec![1.1, 2.2], vec![3.3, 4.4], vec![1.0, 0.0]];
        let t: Tensor<NdArray, 2> = d.to_tensor(&Default::default());
        
        assert_eq!(t.shape().dims.len(), 2);
        assert_eq!(t.shape().dims, [3, 2]);
    }

    #[test]
    fn test_to_tensor_usize(){
        let d: usize = 1;
        let t: Tensor<NdArray, 1, Int> = d.to_tensor(&Default::default());
        
        assert_eq!(t.shape().dims.len(), 1);
        assert_eq!(t.shape().dims, [1]);
        assert_eq!(t.into_scalar() as usize, d);
    }

    #[test]
    fn test_to_tensor_vec_usize(){
        let d: Vec<usize> = vec![1, 2];
        let t: Tensor<NdArray, 1, Int> = d.to_tensor(&Default::default());
        
        assert_eq!(t.shape().dims.len(), 1);
        assert_eq!(t.shape().dims, [2]);
    }

    #[test]
    fn test_to_tensor_bool(){
        let d: bool = true;
        let t: Tensor<NdArray, 1, Bool> = d.to_tensor(&Default::default());
        
        assert_eq!(t.shape().dims.len(), 1);
        assert_eq!(t.shape().dims, [1]);
        assert_eq!(t.into_scalar(), d);
    }

    #[test]
    fn test_to_tensor_vec_bool(){
        let d: Vec<bool> = vec![false, true];
        let t: Tensor<NdArray, 1, Bool> = d.to_tensor(&Default::default());
        
        assert_eq!(t.shape().dims.len(), 1);
        assert_eq!(t.shape().dims, [2]);
    }
}