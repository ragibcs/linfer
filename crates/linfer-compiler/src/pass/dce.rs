use std::collections::HashSet;

use linfer_ir::{Graph, NodeId};

use crate::CompilerPass;

pub struct DeadCodeEliminationPass;

impl CompilerPass for DeadCodeEliminationPass {
    fn name(&self) -> &str {
        "dce"
    }

    fn run(&self, graph: &mut Graph) -> usize {
        graph.rebuild_edges();

        let mut live_nodes: HashSet<NodeId> = graph
            .nodes
            .iter()
            .filter(|n| {
                n.op.has_side_effects()
                    || n.op
                        .output_tensors()
                        .iter()
                        .any(|t| graph.is_output_tensor(*t))
            })
            .map(|n| n.id)
            .collect();

        let mut changed = true;
        while changed {
            changed = false;
            let current_live: Vec<NodeId> = live_nodes.iter().copied().collect();
            for nid in current_live {
                let Some(node) = graph.node(nid) else {
                    continue;
                };
                for input in node.op.input_tensors() {
                    if let Some(prod) = graph.producer_of_tensor(input) {
                        if live_nodes.insert(prod) {
                            changed = true;
                        }
                    }
                }
            }
        }

        let all: HashSet<NodeId> = graph.nodes.iter().map(|n| n.id).collect();
        let dead: HashSet<NodeId> = all.difference(&live_nodes).copied().collect();
        let removed = graph.remove_nodes(&dead);
        if removed > 0 {
            graph.rebuild_edges();
        }
        removed
    }
}
