use std::collections::HashSet;

use linfer_ir::{Graph, Op};

use crate::CompilerPass;

pub struct QkvFusionPass;

impl CompilerPass for QkvFusionPass {
    fn name(&self) -> &str {
        "qkv_fuse"
    }

    fn run(&self, graph: &mut Graph) -> usize {
        graph.rebuild_edges();

        let sdpa_nodes: Vec<(usize, usize, usize)> = graph
            .nodes
            .iter()
            .filter_map(|n| match n.op {
                Op::SDPA { q, k, v, .. } => Some((q, k, v)),
                _ => None,
            })
            .collect();

        let mut updates: Vec<(usize, Op)> = Vec::new();
        let mut remove_ids: HashSet<usize> = HashSet::new();

        for (q, k, v) in sdpa_nodes {
            let Some(q_node) = graph.producer_of_tensor(q) else {
                continue;
            };
            let Some(k_node) = graph.producer_of_tensor(k) else {
                continue;
            };
            let Some(v_node) = graph.producer_of_tensor(v) else {
                continue;
            };

            let (q_lhs, q_rhs, q_out) = match graph.node(q_node).map(|n| n.op.clone()) {
                Some(Op::MatMul { lhs, rhs, out }) => (lhs, rhs, out),
                _ => continue,
            };
            let (k_lhs, k_rhs, k_out) = match graph.node(k_node).map(|n| n.op.clone()) {
                Some(Op::MatMul { lhs, rhs, out }) => (lhs, rhs, out),
                _ => continue,
            };
            let (v_lhs, v_rhs, v_out) = match graph.node(v_node).map(|n| n.op.clone()) {
                Some(Op::MatMul { lhs, rhs, out }) => (lhs, rhs, out),
                _ => continue,
            };

            if q_lhs != k_lhs || q_lhs != v_lhs {
                continue;
            }
            if q_out != q || k_out != k || v_out != v {
                continue;
            }

            let ids = [q_node, k_node, v_node];
            if ids.iter().any(|id| remove_ids.contains(id)) {
                continue;
            }

            let keeper = *ids.iter().min().unwrap_or(&q_node);
            let fused = Op::FusedQKV {
                input: q_lhs,
                wq: q_rhs,
                wk: k_rhs,
                wv: v_rhs,
                out_q: q,
                out_k: k,
                out_v: v,
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
