use std::ffi::c_long;

unsafe extern "C" {
    fn dot_product_asm(a: c_long, b: c_long) -> c_long;
}

fn main() {
    let a = 33;
    let b = 34;
    let c = unsafe { dot_product_asm(a, b) };
    println!("{} + {} = {}", a, b, c);
}
