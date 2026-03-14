use linfer_ir::{Graph, Op};

use crate::CompilerPass;

pub struct DualPathSelectionPass;

impl CompilerPass for DualPathSelectionPass {
    fn name(&self) -> &str {
        "dual_path"
    }

    fn run(&self, graph: &mut Graph) -> usize {
        graph.rebuild_edges();

        let candidates: Vec<(usize, usize, usize, usize, bool)> = graph
            .nodes
            .iter()
            .filter_map(|n| match n.op {
                Op::MatMul { lhs, rhs, out } => {
                    let likely_decode = !graph.is_output_tensor(out)
                        && graph.consumers_of_tensor(out).len() <= 1
                        && graph.producer_of_tensor(rhs).is_none();
                    Some((n.id, lhs, rhs, out, likely_decode))
                }
                _ => None,
            })
            .collect();

        let mut changed = 0;
        for (id, lhs, rhs, out, likely_decode) in candidates {
            if !likely_decode {
                continue;
            }
            if let Some(node) = graph.node_mut(id) {
                node.op = Op::QMatVec {
                    weight: rhs,
                    input: lhs,
                    out,
                    bits: 4,
                };
                changed += 1;
            }
        }

        if changed > 0 {
            graph.rebuild_edges();
        }
        changed
    }
}
