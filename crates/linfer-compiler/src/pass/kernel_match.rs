use linfer_ir::{Graph, Op};

use crate::CompilerPass;

pub struct KernelMatchPass;

impl CompilerPass for KernelMatchPass {
    fn name(&self) -> &str {
        "kernel_match"
    }

    fn run(&self, graph: &mut Graph) -> usize {
        graph
            .nodes
            .iter()
            .filter(|n| {
                matches!(
                    n.op,
                    Op::MatMul { .. }
                        | Op::QMatVec { .. }
                        | Op::RMSNorm { .. }
                        | Op::RoPE { .. }
                        | Op::SDPA { .. }
                        | Op::SwiGLU { .. }
                        | Op::FusedQKV { .. }
                        | Op::FusedMLP { .. }
                        | Op::FusedAddNorm { .. }
                        | Op::FusedRoPEKV { .. }
                )
            })
            .count()
    }
}
