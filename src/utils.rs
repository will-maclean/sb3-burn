use rand::{seq::SliceRandom, thread_rng};

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
