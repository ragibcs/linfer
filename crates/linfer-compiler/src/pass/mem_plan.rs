use std::collections::HashMap;

use linfer_ir::{Graph, TensorId};

use crate::CompilerPass;

pub struct MemoryPlannerPass;

impl CompilerPass for MemoryPlannerPass {
    fn name(&self) -> &str {
        "mem_plan"
    }

    fn run(&self, graph: &mut Graph) -> usize {
        let mut uses: HashMap<TensorId, usize> = HashMap::new();
        for node in &graph.nodes {
            for t in node.op.input_tensors() {
                *uses.entry(t).or_insert(0) += 1;
            }
        }
        for &t in &graph.outputs {
            *uses.entry(t).or_insert(0) += 1;
        }

        let mut free_pool: Vec<TensorId> = Vec::new();
        let mut alias_opportunities = 0usize;

        let ordered = graph.topo_sort();
        for nid in ordered {
            let Some(node) = graph.node(nid) else {
                continue;
            };

            for t in node.op.input_tensors() {
                if let Some(c) = uses.get_mut(&t) {
                    if *c > 0 {
                        *c -= 1;
                    }
                    if *c == 0 && !graph.is_output_tensor(t) {
                        free_pool.push(t);
                    }
                }
            }

            let outputs = node.op.output_tensors();
            if outputs.len() == 1 && !free_pool.is_empty() {
                alias_opportunities += 1;
                free_pool.pop();
            }
        }

        alias_opportunities
    }
}
