use rand::{thread_rng, seq::SliceRandom};

pub fn generate_1_0_vector(size: usize, num_true: usize) -> Vec<i32> {
    // Create a vector of specified size initialized with false
    let mut vec = vec![0; size];

    // Generate a list of indices and shuffle them
    let mut indices: Vec<usize> = (0..size).collect();
    let mut rng = thread_rng();
    indices.shuffle(&mut rng);

    // Set the first num_true elements to true
    for i in 0..num_true {
        vec[indices[i]] = 1;
    }

    vec
}