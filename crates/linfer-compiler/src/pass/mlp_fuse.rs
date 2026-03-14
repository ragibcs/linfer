use std::collections::HashSet;

use linfer_ir::{Graph, Op};

use crate::CompilerPass;

pub struct MlpFusionPass;

impl CompilerPass for MlpFusionPass {
    fn name(&self) -> &str {
        "mlp_fuse"
    }

    fn run(&self, graph: &mut Graph) -> usize {
        graph.rebuild_edges();

        let down_nodes: Vec<(usize, usize, usize)> = graph
            .nodes
            .iter()
            .filter_map(|n| match n.op {
                Op::MatMul {
                    lhs: swiglu_out,
                    rhs: wdown,
                    out,
                } => Some((swiglu_out, wdown, out)),
                _ => None,
            })
            .collect();

        let mut updates: Vec<(usize, Op)> = Vec::new();
        let mut remove_ids: HashSet<usize> = HashSet::new();

        for (swiglu_out, wdown, out) in down_nodes {
            let Some(swiglu_node_id) = graph.producer_of_tensor(swiglu_out) else {
                continue;
            };
            let Some(swiglu_node) = graph.node(swiglu_node_id) else {
                continue;
            };

            let Op::SwiGLU {
                gate,
                up,
                out: swiglu_out_check,
            } = swiglu_node.op
            else {
                continue;
            };

            if swiglu_out_check != swiglu_out {
                continue;
            }

            let Some(gate_node_id) = graph.producer_of_tensor(gate) else {
                continue;
            };
            let Some(up_node_id) = graph.producer_of_tensor(up) else {
                continue;
            };

            if gate_node_id == up_node_id {
                continue;
            }

            let (input_a, wgate, gate_out) = match graph.node(gate_node_id).map(|n| n.op.clone()) {
                Some(Op::MatMul { lhs, rhs, out }) => (lhs, rhs, out),
                _ => continue,
            };
            let (input_b, wup, up_out) = match graph.node(up_node_id).map(|n| n.op.clone()) {
                Some(Op::MatMul { lhs, rhs, out }) => (lhs, rhs, out),
                _ => continue,
            };

            if input_a != input_b || gate_out != gate || up_out != up {
                continue;
            }
            if graph.consumers_of_tensor(gate).len() != 1
                || graph.consumers_of_tensor(up).len() != 1
                || graph.consumers_of_tensor(swiglu_out).len() != 1
            {
                continue;
            }

            let ids = [gate_node_id, up_node_id, swiglu_node_id];
            if ids.iter().any(|id| remove_ids.contains(id)) {
                continue;
            }

            let keeper = *ids.iter().min().unwrap_or(&gate_node_id);
            let fused = Op::FusedMLP {
                input: input_a,
                wgate,
                wup,
                wdown,
                out,
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
