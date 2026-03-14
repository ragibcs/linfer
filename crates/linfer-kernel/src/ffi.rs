unsafe extern "C" {
    pub fn gemv_q4_avx2(
        weights: *const u8,
        scales: *const f32,
        input: *const f32,
        output: *mut f32,
        rows: i32,
        cols: i32,
    );

    pub fn gemv_q8_avx2(
        weights: *const i8,
        scales: *const f32,
        input: *const f32,
        output: *mut f32,
        rows: i32,
        cols: i32,
    );

    pub fn gemm_f32_avx2(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    );

    pub fn rms_norm_avx2(input: *const f32, weight: *const f32, output: *mut f32, len: i32, eps: f32);
    pub fn rope_f32(q: *mut f32, k: *mut f32, head_dim: i32, pos: i32, theta: f32);
    pub fn sdpa_f32(q: *const f32, k: *const f32, v: *const f32, out: *mut f32, seq: i32, dim: i32);
    pub fn swiglu_fused(gate: *const f32, up: *const f32, out: *mut f32, len: i32);
}
