use serde::{Deserialize, Serialize};

pub type TensorId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    I8,
    I4,
    U32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }
}
