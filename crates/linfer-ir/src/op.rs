use serde::{Deserialize, Serialize};

use crate::tensor::TensorId;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Op {
    MatMul {
        lhs: TensorId,
        rhs: TensorId,
        out: TensorId,
    },
    QMatVec {
        weight: TensorId,
        input: TensorId,
        out: TensorId,
        bits: u8,
    },
    RMSNorm {
        input: TensorId,
        weight: TensorId,
        out: TensorId,
        eps: f32,
    },
    RoPE {
        q: TensorId,
        k: TensorId,
        pos: usize,
    },
    SDPA {
        q: TensorId,
        k: TensorId,
        v: TensorId,
        out: TensorId,
    },
    SwiGLU {
        gate: TensorId,
        up: TensorId,
        out: TensorId,
    },
    KVAppend {
        k: TensorId,
        v: TensorId,
        layer: usize,
    },
    Residual {
        a: TensorId,
        b: TensorId,
        out: TensorId,
    },
    Gather {
        table: TensorId,
        idx: TensorId,
        out: TensorId,
    },
    FusedQKV {
        input: TensorId,
        wq: TensorId,
        wk: TensorId,
        wv: TensorId,
        out_q: TensorId,
        out_k: TensorId,
        out_v: TensorId,
    },
    FusedMLP {
        input: TensorId,
        wgate: TensorId,
        wup: TensorId,
        wdown: TensorId,
        out: TensorId,
    },
    FusedAddNorm {
        residual: TensorId,
        input: TensorId,
        weight: TensorId,
        out: TensorId,
    },
    FusedRoPEKV {
        q: TensorId,
        k: TensorId,
        v: TensorId,
        pos: usize,
        layer: usize,
    },
}

impl Op {
    pub fn input_tensors(&self) -> Vec<TensorId> {
        match self {
            Op::MatMul { lhs, rhs, .. } => vec![*lhs, *rhs],
            Op::QMatVec { weight, input, .. } => vec![*weight, *input],
            Op::RMSNorm { input, weight, .. } => vec![*input, *weight],
            Op::RoPE { q, k, .. } => vec![*q, *k],
            Op::SDPA { q, k, v, .. } => vec![*q, *k, *v],
            Op::SwiGLU { gate, up, .. } => vec![*gate, *up],
            Op::KVAppend { k, v, .. } => vec![*k, *v],
            Op::Residual { a, b, .. } => vec![*a, *b],
            Op::Gather { table, idx, .. } => vec![*table, *idx],
            Op::FusedQKV {
                input, wq, wk, wv, ..
            } => vec![*input, *wq, *wk, *wv],
            Op::FusedMLP {
                input,
                wgate,
                wup,
                wdown,
                ..
            } => vec![*input, *wgate, *wup, *wdown],
            Op::FusedAddNorm {
                residual,
                input,
                weight,
                ..
            } => vec![*residual, *input, *weight],
            Op::FusedRoPEKV { q, k, v, .. } => vec![*q, *k, *v],
        }
    }

    pub fn output_tensors(&self) -> Vec<TensorId> {
        match self {
            Op::MatMul { out, .. }
            | Op::QMatVec { out, .. }
            | Op::RMSNorm { out, .. }
            | Op::SDPA { out, .. }
            | Op::SwiGLU { out, .. }
            | Op::Residual { out, .. }
            | Op::Gather { out, .. }
            | Op::FusedMLP { out, .. }
            | Op::FusedAddNorm { out, .. } => vec![*out],
            Op::FusedQKV {
                out_q,
                out_k,
                out_v,
                ..
            } => vec![*out_q, *out_k, *out_v],
            Op::RoPE { q, k, .. } | Op::FusedRoPEKV { q, k, .. } => vec![*q, *k],
            Op::KVAppend { .. } => vec![],
        }
    }

    pub fn has_side_effects(&self) -> bool {
        matches!(self, Op::KVAppend { .. } | Op::FusedRoPEKV { .. })
    }
}
