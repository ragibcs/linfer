use linfer_ir::Graph;

use crate::pass::{
    dce::DeadCodeEliminationPass, dual_path::DualPathSelectionPass, kernel_match::KernelMatchPass,
    mem_plan::MemoryPlannerPass, mlp_fuse::MlpFusionPass, norm_fuse::NormFusionPass,
    qkv_fuse::QkvFusionPass, rope_fuse::RopeFusionPass,
};

pub trait CompilerPass {
    fn name(&self) -> &str;
    fn run(&self, graph: &mut Graph) -> usize;
}

#[derive(Debug, Default)]
pub struct CompileReport {
    pub applied: Vec<(String, usize)>,
}

pub struct Pipeline {
    passes: Vec<Box<dyn CompilerPass + Send + Sync>>,
}

impl Default for Pipeline {
    fn default() -> Self {
        Self {
            passes: vec![
                Box::new(DeadCodeEliminationPass),
                Box::new(QkvFusionPass),
                Box::new(MlpFusionPass),
                Box::new(NormFusionPass),
                Box::new(RopeFusionPass),
                Box::new(DualPathSelectionPass),
                Box::new(MemoryPlannerPass),
                Box::new(KernelMatchPass),
            ],
        }
    }
}

impl Pipeline {
    pub fn compile(&self, graph: &mut Graph) -> CompileReport {
        let mut report = CompileReport::default();
        for pass in &self.passes {
            let n = pass.run(graph);
            report.applied.push((pass.name().to_string(), n));
        }
        report
    }
}
