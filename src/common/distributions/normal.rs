use std::f32::consts::{E, PI};

use burn::{
    module::Module,
    tensor::{backend::Backend, Distribution, Tensor},
};

use super::{
    distribution::BaseDistribution,
    exp_family::{ExpFamily, NaturalParams},
};

#[derive(Debug, Module)]
pub struct Normal<B: Backend, const D: usize> {
    loc: Tensor<B, D>,
    scale: Tensor<B, D>,
}

impl<B: Backend, const D: usize> Normal<B, D> {
    pub fn new(loc: Tensor<B, D>, scale: Tensor<B, D>) -> Self {
        assert!(
            scale.clone().greater_elem(0.0).all().into_scalar(),
            "scale>0 check failed. scale: {scale}"
        );

        Self {
            loc: loc.no_grad(),
            scale: scale.no_grad(),
        }
    }
}

impl<B: Backend, const D: usize> ExpFamily<B, D> for Normal<B, D> {
    fn log_normaliser(&self, natural_params: NaturalParams<B, D>) -> Tensor<B, D> {
        let x = natural_params.params[0].clone();
        let y = natural_params.params[1].clone();

        x.powi_scalar(2).div(y.clone()).mul_scalar(-0.25)
            + y.powi_scalar(-1).mul_scalar(-PI).log().mul_scalar(0.5)
    }

    fn mean_carrier_measure(&self) -> f32 {
        0.0
    }

    fn natural_params(&self) -> NaturalParams<B, D> {
        let np = vec![
            self.loc.clone().div(self.variance()),
            -self.variance().powi_scalar(-1).mul_scalar(0.5),
        ];

        NaturalParams { params: np }
    }
}

impl<B: Backend, const D: usize> BaseDistribution<B, D> for Normal<B, D> {
    fn mean(&self) -> Tensor<B, D> {
        self.loc.clone()
    }

    fn mode(&self) -> Tensor<B, D> {
        self.loc.clone()
    }

    fn variance(&self) -> Tensor<B, D> {
        self.scale.clone().powi_scalar(2)
    }

    fn stdev(&self) -> Tensor<B, D> {
        self.scale.clone()
    }

    fn sample(&mut self) -> Tensor<B, D> {
        self.rsample()
    }

    fn rsample(&self) -> Tensor<B, D> {
        let s = Tensor::random_like(&self.loc, Distribution::Normal(0.0, 1.0));

        s.mul(self.scale.clone()) + self.loc.clone()
    }

    fn log_prob(&self, value: Tensor<B, D>) -> Tensor<B, D> {
        let log_scale = self.scale.clone().log();

        log_scale.add_scalar((2.0 * PI).sqrt().log(E)).sub(
            (value - self.loc.clone())
                .powi_scalar(2)
                .div_scalar(2)
                .div(self.variance()),
        )
    }

    fn cdf(&self, _value: Tensor<B, D>) -> Tensor<B, D> {
        todo!()
    }

    fn icdf(&self, _value: Tensor<B, D>) -> Tensor<B, D> {
        todo!()
    }

    fn entropy(&self) -> Tensor<B, D> {
        self.scale
            .clone()
            .log()
            .add_scalar(0.5 + 0.5 * (2.0 * PI).log(E))
    }
}

#[cfg(test)]
mod test {
    use burn::{
        backend::Wgpu,
        tensor::{ElementConversion, Tensor},
    };

    use crate::common::distributions::{
        distribution::BaseDistribution, exp_family::ExpFamily, normal::Normal,
    };

    #[test]
    fn test_normal_distribution() {
        type Backend = Wgpu;
        let loc = Tensor::<Backend, 2>::from_floats([[1.0, 0.0], [2.0, -2.0]], &Default::default());
        let scale =
            Tensor::<Backend, 2>::from_floats([[1.0, 0.1], [2.0, 2.0]], &Default::default());

        let mut dist = Normal::new(loc.clone(), scale.clone());

        assert_eq!(dist.mean_carrier_measure(), 0.0);

        // just test that the ones run
        dist.log_normaliser(dist.natural_params());
        dist.sample();
        let s = dist.rsample();
        dist.log_prob(s);
        dist.entropy();
        dist.perplexity();

        assert!(dist.mean().equal(loc.clone()).all().into_scalar());
        assert!(dist.mode().equal(loc.clone()).all().into_scalar());
        assert!(dist
            .variance()
            .equal(scale.clone().powi_scalar(2).clone())
            .all()
            .into_scalar());
        assert!(dist.stdev().equal(scale.clone()).all().into_scalar());
    }

    #[should_panic]
    #[test]
    fn test_bad_normal_init1() {
        type Backend = Wgpu;
        let loc = Tensor::<Backend, 1>::from_floats([1.0], &Default::default());
        let scale = Tensor::<Backend, 1>::from_floats([0.0], &Default::default());

        Normal::new(loc, scale);
    }

    #[should_panic]
    #[test]
    fn test_bad_normal_init2() {
        type Backend = Wgpu;
        let loc = Tensor::<Backend, 1>::from_floats([1.0], &Default::default());
        let scale = Tensor::<Backend, 1>::from_floats([-1.0], &Default::default());

        Normal::new(loc, scale);
    }

    #[test]
    fn normal_dist_calc_verification() {
        type Backend = Wgpu;

        // calculated with PyTorch
        // dist = Normal(mean=0.0, std=1.0)
        // sample = 0.4225
        // log_prob = -1.0082

        let loc = Tensor::<Backend, 1>::from_floats([0.0], &Default::default());
        let scale = Tensor::<Backend, 1>::from_floats([1.0], &Default::default());

        let dist = Normal::new(loc, scale);

        let sample = Tensor::<Backend, 1>::from_floats([0.4225], &Default::default());
        let expected_log_prob = Tensor::<Backend, 1>::from_floats([-1.0082], &Default::default());

        let calculated_log_prob = dist.log_prob(sample);

        assert!(
            (expected_log_prob - calculated_log_prob)
                .sum()
                .into_scalar()
                .elem::<f32>()
                < 1e-12
        );
    }
}
