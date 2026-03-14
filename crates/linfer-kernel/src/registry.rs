use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelKind {
    GemvQ4,
    GemvQ8,
    GemmF32,
    RmsNorm,
    Rope,
    Sdpa,
    Swiglu,
}

#[derive(Debug, Clone)]
pub struct KernelRegistry {
    map: HashMap<KernelKind, &'static str>,
}

impl Default for KernelRegistry {
    fn default() -> Self {
        let mut map = HashMap::new();
        map.insert(KernelKind::GemvQ4, "gemv_q4_avx2");
        map.insert(KernelKind::GemvQ8, "gemv_q8_avx2");
        map.insert(KernelKind::GemmF32, "gemm_f32_avx2");
        map.insert(KernelKind::RmsNorm, "rms_norm_avx2");
        map.insert(KernelKind::Rope, "rope_f32");
        map.insert(KernelKind::Sdpa, "sdpa_f32");
        map.insert(KernelKind::Swiglu, "swiglu_fused");
        Self { map }
    }
}

impl KernelRegistry {
    pub fn resolve(&self, kind: KernelKind) -> Option<&'static str> {
        self.map.get(&kind).copied()
    }
}
