use std::collections::{HashMap, VecDeque};

use crate::{Graph, NodeId};

pub fn topo_sort(graph: &Graph) -> Vec<NodeId> {
    let mut indegree: HashMap<NodeId, usize> = graph.nodes.iter().map(|n| (n.id, 0)).collect();

    for edge in &graph.edges {
        if let Some(v) = indegree.get_mut(&edge.to) {
            *v += 1;
        }
    }

    let mut queue: VecDeque<NodeId> = indegree
        .iter()
        .filter_map(|(k, v)| if *v == 0 { Some(*k) } else { None })
        .collect();

    let mut out = Vec::with_capacity(graph.nodes.len());
    while let Some(id) = queue.pop_front() {
        out.push(id);
        for edge in graph.edges.iter().filter(|e| e.from == id) {
            if let Some(v) = indegree.get_mut(&edge.to) {
                *v -= 1;
                if *v == 0 {
                    queue.push_back(edge.to);
                }
            }
        }
    }

    if out.len() == graph.nodes.len() {
        out
    } else {
        graph.nodes.iter().map(|n| n.id).collect()
    }
}
