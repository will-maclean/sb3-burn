#[derive(Clone)]
pub enum SpaceSample {
    Discrete (f32),
    Continuous (Vec<f32>)
}

#[derive(Clone)]
pub enum Space {
    Discrete {size: usize},
    Continuous {lows: Vec<f32>, highs: Vec<f32>}
}

impl Space {
    pub fn sample(&self) -> SpaceSample {
        todo!()
    }

    pub fn size(&self) -> usize {
        match self {
            Space::Discrete { size } => *size,
            Space::Continuous { lows, highs: _ } => lows.len(),
        }
    }
}