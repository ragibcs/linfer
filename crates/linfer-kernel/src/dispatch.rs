use crate::registry::KernelKind;

#[derive(Debug, Clone, Copy)]
pub enum QuantType {
    Q4,
    Q8,
    F32,
}

#[derive(Debug, Clone, Copy)]
pub struct KernelDispatch;

impl KernelDispatch {
    pub fn matvec_for(quant: QuantType) -> KernelKind {
        match quant {
            QuantType::Q4 => KernelKind::GemvQ4,
            QuantType::Q8 => KernelKind::GemvQ8,
            QuantType::F32 => KernelKind::GemmF32,
        }
    }
}
