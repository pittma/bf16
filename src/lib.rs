use std::ffi::c_float;

unsafe extern "C" {
    pub fn cosine_sim_asm(a: *const c_float, b: *const c_float, n: usize) -> c_float;
}

pub fn semantic_impl<const N: usize>(a: [f32; N], b: [f32; N]) -> f32 {
    let mag_f = |acc: f32, v: &f32| v.mul_add(*v, acc);
    let mag_a = a.iter().fold(0_f32, mag_f).sqrt();
    let mag_b = b.iter().fold(0_f32, mag_f).sqrt();
    let dp = a
        .iter()
        .zip(b.iter())
        .fold(0_f32, |acc, (a, b)| a.mul_add(*b, acc));
    dp / (mag_a * mag_b)
}

pub fn one_pass_impl<const N: usize>(a: [f32; N], b: [f32; N]) -> f32 {
    let mut dp = 0.0;
    let mut mag_a = 0.0;
    let mut mag_b = 0.0;
    for i in 0..N {
        mag_a = a[i].mul_add(a[i], mag_a);
        mag_b = b[i].mul_add(b[i], mag_b);
        dp = a[i].mul_add(b[i], dp);
    }
    dp / (mag_a * mag_b).sqrt()
}
