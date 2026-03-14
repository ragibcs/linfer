use std::collections::HashSet;

use linfer_ir::{Graph, Op};

use crate::CompilerPass;

pub struct NormFusionPass;

impl CompilerPass for NormFusionPass {
    fn name(&self) -> &str {
        "norm_fuse"
    }

    fn run(&self, graph: &mut Graph) -> usize {
        graph.rebuild_edges();

        let norm_nodes: Vec<(usize, usize, usize)> = graph
            .nodes
            .iter()
            .filter_map(|n| match n.op {
                Op::RMSNorm {
                    input,
                    weight,
                    out,
                    ..
                } => Some((input, weight, out)),
                _ => None,
            })
            .collect();

        let mut updates: Vec<(usize, Op)> = Vec::new();
        let mut remove_ids: HashSet<usize> = HashSet::new();

        for (norm_input, weight, norm_out) in norm_nodes {
            let Some(residual_node_id) = graph.producer_of_tensor(norm_input) else {
                continue;
            };
            let Some(residual_node) = graph.node(residual_node_id) else {
                continue;
            };

            let Op::Residual { a, b, out } = residual_node.op else {
                continue;
            };

            if out != norm_input {
                continue;
            }
            if graph.consumers_of_tensor(norm_input).len() != 1 {
                continue;
            }

            let ids = [residual_node_id, graph.producer_of_tensor(norm_out).unwrap_or(residual_node_id)];
            if ids.iter().any(|id| remove_ids.contains(id)) {
                continue;
            }

            let keeper = residual_node_id;
            let fused = Op::FusedAddNorm {
                residual: a,
                input: b,
                weight,
                out: norm_out,
            };

            updates.push((keeper, fused));
            for id in ids {
                if id != keeper {
                    remove_ids.insert(id);
                }
            }
        }

        for (id, op) in &updates {
            if let Some(node) = graph.node_mut(*id) {
                node.op = op.clone();
            }
        }

        if remove_ids.is_empty() && updates.is_empty() {
            return 0;
        }

        graph.nodes.retain(|n| !remove_ids.contains(&n.id));
        graph.rebuild_edges();
        updates.len()
    }
}
