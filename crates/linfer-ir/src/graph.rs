use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::{op::Op, tensor::TensorId, topo};

pub type NodeId = usize;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub op: Op,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
    pub tensor: TensorId,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub outputs: Vec<TensorId>,
    next_node_id: NodeId,
}

impl Graph {
    pub fn add_op(&mut self, op: Op) -> NodeId {
        let id = self.next_node_id;
        self.next_node_id += 1;
        self.nodes.push(Node { id, op });
        id
    }

    pub fn connect(&mut self, from: NodeId, to: NodeId, tensor: TensorId) {
        self.edges.push(Edge { from, to, tensor });
    }

    pub fn mark_output(&mut self, tensor: TensorId) {
        if !self.outputs.contains(&tensor) {
            self.outputs.push(tensor);
        }
    }

    pub fn is_output_tensor(&self, tensor: TensorId) -> bool {
        self.outputs.contains(&tensor)
    }

    pub fn topo_sort(&self) -> Vec<NodeId> {
        topo::topo_sort(self)
    }

    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }

    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.iter_mut().find(|n| n.id == id)
    }

    pub fn producer_of_tensor(&self, tensor: TensorId) -> Option<NodeId> {
        self.nodes
            .iter()
            .rev()
            .find(|n| n.op.output_tensors().contains(&tensor))
            .map(|n| n.id)
    }

    pub fn consumers_of_tensor(&self, tensor: TensorId) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|n| n.op.input_tensors().contains(&tensor))
            .map(|n| n.id)
            .collect()
    }

    pub fn rebuild_edges(&mut self) {
        let mut edges = Vec::new();
        let mut last_producer: std::collections::HashMap<TensorId, NodeId> =
            std::collections::HashMap::new();

        let mut ordered: Vec<&Node> = self.nodes.iter().collect();
        ordered.sort_by_key(|n| n.id);

        for node in ordered {
            for t in node.op.input_tensors() {
                if let Some(from) = last_producer.get(&t).copied() {
                    if from != node.id {
                        edges.push(Edge {
                            from,
                            to: node.id,
                            tensor: t,
                        });
                    }
                }
            }
            for t in node.op.output_tensors() {
                last_producer.insert(t, node.id);
            }
        }

        self.edges = edges;
    }

    pub fn remove_nodes(&mut self, ids: &HashSet<NodeId>) -> usize {
        let before = self.nodes.len();
        self.nodes.retain(|n| !ids.contains(&n.id));
        self.edges
            .retain(|e| !ids.contains(&e.from) && !ids.contains(&e.to));
        before.saturating_sub(self.nodes.len())
    }

    pub fn producer_count(&self, id: NodeId) -> usize {
        self.edges.iter().filter(|e| e.to == id).count()
    }

    pub fn consumer_count(&self, id: NodeId) -> usize {
        self.edges.iter().filter(|e| e.from == id).count()
    }
}
