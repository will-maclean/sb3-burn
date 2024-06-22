use burn::tensor::{backend::Backend, Distribution, Float, Tensor};

pub trait SimpleDistribution<B: Backend, const D: usize> {
    fn sample(&mut self, device: &B::Device) -> Tensor<B, D>;
    fn seed(&mut self, _seed: [u8; 32]){
        todo!();
    }
}

pub struct UniformDistribution<B: Backend, const D: usize>{
    lows: Tensor<B, D>,
    highs: Tensor<B, D>,
}

impl<B: Backend, const D: usize> UniformDistribution<B, D>{
    pub fn new(lows: Tensor<B, D>, highs: Tensor<B, D>) -> Self {
        assert!(lows.clone().lower_equal(highs.clone()).all().into_scalar());

        Self {
            lows,
            highs
        }
    }
}

impl<B: Backend, const D: usize> SimpleDistribution<B, D> for UniformDistribution<B, D>{
    fn sample(&mut self, device: &B::Device) -> Tensor<B, D>{
        let shape = self.lows.shape();
        let sample: Tensor<B, D, Float> = Tensor::random(shape, Distribution::Uniform(0.0, 1.0), device);
        let range = self.highs.clone().sub(self.lows.clone());
        let sample = sample.mul(range).add(self.lows.clone());

        sample
    }
}

pub struct NormalDistribution<B: Backend, const D: usize>{
    means: Tensor<B, D>,
    stds: Tensor<B, D>,
}

impl<B: Backend, const D: usize> NormalDistribution<B, D>{
    pub fn new(means: Tensor<B, D>, stds: Tensor<B, D>) -> Self {
        assert_eq!(means.shape(), stds.shape());
        assert!(stds.clone().greater_elem(0.0).all().into_scalar());

        Self {
            means,
            stds
        }
    }
}

impl<B: Backend, const D: usize> SimpleDistribution<B, D> for NormalDistribution<B, D>{
    fn sample(&mut self, device: &B::Device) -> Tensor<B, D>{
        let shape = self.means.shape();
        let sample: Tensor<B, D, Float> = Tensor::random(shape, Distribution::Normal(0.0, 1.0), device);
        let sample = sample.mul(self.stds.clone()).add(self.means.clone());

        sample
    }
}


#[cfg(test)]
mod test{
    use burn::{backend::{ndarray::NdArrayDevice, NdArray}, tensor::Tensor};

    use super::{NormalDistribution, SimpleDistribution, UniformDistribution};

    #[test]
    fn test_uniform_distribution(){
        let device = NdArrayDevice::default();
        let lows: Tensor<NdArray, 1> = Tensor::from_floats([-1.0, 0.0, 1.0], &device);
        let highs: Tensor<NdArray, 1> = Tensor::from_floats([0.0, 0.0, 3.0], &device);

        let mut dist = UniformDistribution::new(lows.clone(), highs.clone());
        let sample = dist.sample(&device);

        assert!(highs.greater_equal(sample.clone()).all().into_scalar());
        assert!(lows.lower_equal(sample.clone()).all().into_scalar());
    }

    #[test]
    fn test_normal_distribution(){
        let device = NdArrayDevice::default();
        let means: Tensor<NdArray, 1> = Tensor::from_floats([-1.0, 1.0], &device);
        let stds: Tensor<NdArray, 1> = Tensor::from_floats([0.1, 3.0], &device);

        let mut dist = NormalDistribution::new(means.clone(), stds.clone());
        dist.sample(&device);
    }

    #[should_panic]
    #[test]
    fn bad_uniform_init1(){
        let device = NdArrayDevice::default();
        let lows: Tensor<NdArray, 1> = Tensor::from_floats([-1.0, 0.0, 1.0], &device);
        let highs: Tensor<NdArray, 1> = Tensor::from_floats([0.0, 0.0, 3.0], &device);

        UniformDistribution::new(highs.clone(), lows.clone());
    }

    #[should_panic]
    #[test]
    fn bad_uniform_init2(){
        let device = NdArrayDevice::default();
        let lows: Tensor<NdArray, 1> = Tensor::from_floats([-1.0, 0.0, 1.0, 2.0], &device);
        let highs: Tensor<NdArray, 1> = Tensor::from_floats([0.0, 0.0, 3.0, 4.0], &device);

        UniformDistribution::new(highs.clone(), lows.clone());
    }

    #[should_panic]
    #[test]
    fn bad_normal_init1(){
        let device = NdArrayDevice::default();
        let means: Tensor<NdArray, 1> = Tensor::from_floats([-1.0, 0.0, 1.0], &device);
        let stds: Tensor<NdArray, 1> = Tensor::from_floats([1.0, 0.1, 3.0, 2.0], &device);

        NormalDistribution::new(means, stds);
    }

    #[should_panic]
    #[test]
    fn bad_normal_init2(){
        let device = NdArrayDevice::default();
        let means: Tensor<NdArray, 1> = Tensor::from_floats([-1.0, 0.0, 1.0], &device);
        let stds: Tensor<NdArray, 1> = Tensor::from_floats([1.0, 0.1, 0.0], &device);

        NormalDistribution::new(means, stds);
    }
}