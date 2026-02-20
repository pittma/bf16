use std::{env, path::PathBuf};

use cc::Build;

fn main() {
    let mut root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    root.push("src");

    println!("cargo:rerun-if-changed=asm/cosine-sim-asm.h");
    println!("cargo:rerun-if-changed=asm/cosine-sim-asm.c");
    println!("cargo:rerun-if-changed=asm/x86_64/cosine-sim-avx512.S");

    let mut build = Build::new();
    build
        .include(&root)
        .file(root.join("asm/cosine-sim-asm.c"))
        .file(root.join("asm/x86_64/cosine-sim-avx512.S"))
        .compile("dp");
}
