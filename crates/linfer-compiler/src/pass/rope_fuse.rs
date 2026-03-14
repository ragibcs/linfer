use std::collections::HashSet;

use linfer_ir::{Graph, Op};

use crate::CompilerPass;

pub struct RopeFusionPass;

impl CompilerPass for RopeFusionPass {
    fn name(&self) -> &str {
        "rope_fuse"
    }

    fn run(&self, graph: &mut Graph) -> usize {
        graph.rebuild_edges();

        let kv_nodes: Vec<(usize, usize, usize)> = graph
            .nodes
            .iter()
            .filter_map(|n| match n.op {
                Op::KVAppend { k, v, layer } => Some((k, v, layer)),
                _ => None,
            })
            .collect();

        let mut updates: Vec<(usize, Op)> = Vec::new();
        let mut remove_ids: HashSet<usize> = HashSet::new();

        for (k, v, layer) in kv_nodes {
            let Some(rope_node_id) = graph.producer_of_tensor(k) else {
                continue;
            };
            let Some(rope_node) = graph.node(rope_node_id) else {
                continue;
            };

            let Op::RoPE { q, k: rope_k, pos } = rope_node.op else {
                continue;
            };

            if rope_k != k {
                continue;
            }

            let Some(kv_node_id) = graph
                .nodes
                .iter()
                .find(|n| matches!(n.op, Op::KVAppend { k: kk, v: vv, layer: ll } if kk == k && vv == v && ll == layer))
                .map(|n| n.id)
            else {
                continue;
            };

            if remove_ids.contains(&rope_node_id) || remove_ids.contains(&kv_node_id) {
                continue;
            }

            updates.push((
                rope_node_id,
                Op::FusedRoPEKV {
                    q,
                    k,
                    v,
                    pos,
                    layer,
                },
            ));
            remove_ids.insert(kv_node_id);
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
