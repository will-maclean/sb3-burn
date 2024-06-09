use rand::{seq::SliceRandom, thread_rng, Rng};

pub mod module_update;

pub fn generate_rand_bool_vector(size: usize, num_true: usize) -> Vec<bool> {
    // Create a vector of specified size initialized with false
    let mut vec = vec![false; size];

    // Generate a list of indices and shuffle them
    let mut indices: Vec<usize> = (0..size).collect();
    let mut rng = thread_rng();
    indices.shuffle(&mut rng);

    // Set the first num_true elements to true
    for i in 0..num_true {
        vec[indices[i]] = true;
    }

    vec
}

pub fn linear_decay(curr_frac: f32, start: f32, end: f32, end_frac: f32) -> f32 {
    if curr_frac > end_frac {
        end
    } else {
        start + curr_frac * (end - start) / end_frac
    }
}

pub fn mean(data: &[f32]) -> f32 {
    data.iter().fold(0.0, |acc, x| acc + x) / (data.len() as f32)
}

pub fn generate_random_vector(lows: Vec<f32>, highs: Vec<f32>) -> Vec<f32> {
    if lows.len() != highs.len() {
        panic!("Vectors of lows and highs must have the same length");
    }

    let mut rng = rand::thread_rng();
    let mut random_vector = Vec::with_capacity(lows.len());

    for (low, high) in lows.iter().zip(highs.iter()) {
        if low > high {
            panic!("Each low value must be less than or equal to its corresponding high value");
        }
        random_vector.push(rng.gen_range(*low..=*high));
    }

    random_vector
}

#[cfg(test)]
mod test {
    use crate::utils::mean;

    #[test]
    fn test_mean() {
        let v = [0.0, 1.0, 2.0];

        assert_eq!(mean(&v), 1.0);

        let v = [];

        assert!(mean(&v).is_nan());
    }
}