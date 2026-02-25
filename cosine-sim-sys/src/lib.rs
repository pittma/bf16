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

#[cfg(test)]
mod test {
    const ONE_TO_16: [f32; 16] = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    ];
    const RHS: [f32; 16] = [
        8.938679, 3.701036, 8.291033, 9.767157, 7.0634584, 6.1626635, 3.344915, 4.749503, 8.938679,
        3.701036, 8.291033, 9.767157, 7.0634584, 6.1626635, 3.344915, 4.749503,
    ];

    const EPSILON: f32 = 0.0009;

    fn is_within_range(a: f32, b: f32) -> bool {
        (a - b).abs() <= EPSILON
    }

    #[test]
    fn cosine_sim_impls() {
        assert_eq!(
            super::semantic_impl(ONE_TO_16, RHS),
            super::one_pass_impl(ONE_TO_16, RHS),
        );
        assert!(is_within_range(
            super::one_pass_impl(ONE_TO_16, RHS),
            unsafe { super::cosine_sim_asm(ONE_TO_16.as_ptr(), RHS.as_ptr(), 16) },
        ));
    }
}
