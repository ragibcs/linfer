fn main() {
    println!("cargo:rerun-if-changed=../../kernels/cpu/gemv_q4_avx2.c");
    println!("cargo:rerun-if-changed=../../kernels/cpu/gemv_q8_avx2.c");
    println!("cargo:rerun-if-changed=../../kernels/cpu/gemm_f32_avx2.c");
    println!("cargo:rerun-if-changed=../../kernels/cpu/rms_norm_avx2.c");
    println!("cargo:rerun-if-changed=../../kernels/cpu/rope_f32.c");
    println!("cargo:rerun-if-changed=../../kernels/cpu/sdpa_f32.c");
    println!("cargo:rerun-if-changed=../../kernels/cpu/swiglu_fused.c");

    cc::Build::new()
        .files([
            "../../kernels/cpu/gemv_q4_avx2.c",
            "../../kernels/cpu/gemv_q8_avx2.c",
            "../../kernels/cpu/gemm_f32_avx2.c",
            "../../kernels/cpu/rms_norm_avx2.c",
            "../../kernels/cpu/rope_f32.c",
            "../../kernels/cpu/sdpa_f32.c",
            "../../kernels/cpu/swiglu_fused.c",
        ])
        .include("../../kernels/cpu/utils")
        .flag("-O3")
        .flag("-march=native")
        .flag("-mavx2")
        .flag("-mfma")
        .flag("-fopenmp")
        .flag("-funroll-loops")
        .compile("linfer_kernels");
}
