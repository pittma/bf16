use std::{
    error, fs, io,
    path::Path,
    time::{Duration, Instant},
};

use thiserror::Error;

use bf16;

#[derive(Debug, Error)]
#[error("reading binary file failed: {0}")]
struct IoError(#[from] io::Error);

struct Comp {
    semantic: Duration,
    one_pass: Duration,
    asm: Duration,
}

fn read_vec_bytes<const N: usize, P>(path: P) -> Result<Vec<[f32; N]>, IoError>
where
    P: AsRef<Path>,
{
    let v = fs::read(path)?;
    unsafe {
        let (p, len, _) = v.into_raw_parts();
        let ptr = p as *mut [f32; N];
        Ok(Vec::from_raw_parts(ptr, len / 4 / N, len / 4 / N))
    }
}

fn comp<const N: usize>(left: Vec<[f32; N]>, right: Vec<[f32; N]>) -> Comp {
    let b4 = Instant::now();
    for i in 0..right.len() {
        bf16::semantic_impl(left[i], right[i]);
    }
    let semantic = b4.elapsed();
    let b4 = Instant::now();
    for i in 0..right.len() {
        bf16::one_pass_impl(left[i], right[i]);
    }
    let one_pass = b4.elapsed();
    let b4 = Instant::now();
    for i in 0..right.len() {
        unsafe {
            bf16::cosine_sim_asm(left[i].as_ptr(), right[i].as_ptr(), N);
        };
    }
    let asm = b4.elapsed();
    Comp {
        semantic,
        one_pass,
        asm,
    }
}

fn main() -> Result<(), Box<dyn error::Error>> {
    let left = read_vec_bytes::<768, _>("sources/left.le")?;
    let right = read_vec_bytes::<768, _>("sources/right.le")?;
    println!("processing {} vectors", right.len());
    let c = comp(left, right);
    println!("semantic implementation:      {}ms", c.semantic.as_millis());
    println!("single pass implementation:   {}ms", c.one_pass.as_millis());
    println!("AVX-512F BF16 implementation: {}ms", c.asm.as_millis());
    Ok(())
}

#[cfg(test)]
mod test {

    const EPSILON: f32 = 0.0004;

    fn is_within_range(a: f32, b: f32) -> bool {
        (a - b).abs() <= EPSILON
    }

    #[test]
    fn sts_benchmark_cmp() {
        let left = super::read_vec_bytes::<768, _>("sources/left.le").unwrap();
        let right = super::read_vec_bytes::<768, _>("sources/right.le").unwrap();
        assert_eq!(left.len(), right.len());
        for i in 0..right.len() {
            let op = bf16::one_pass_impl(left[i], right[i]);
            let asm =
                unsafe { bf16::cosine_sim_asm(left[i].as_ptr(), right[i].as_ptr(), 768) };
            assert!(is_within_range(op, asm));
        }
    }
}
