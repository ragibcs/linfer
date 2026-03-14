mod gemma;
mod llama;
mod mistral;
mod phi;
mod qwen;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArchType {
    Llama,
    Mistral,
    Phi,
    Qwen,
    Gemma,
    Unknown,
}

pub fn detect_arch(model_type: Option<&str>) -> ArchType {
    let Some(mt) = model_type.map(|s| s.to_ascii_lowercase()) else {
        return ArchType::Unknown;
    };

    if llama::matches(&mt) {
        ArchType::Llama
    } else if mistral::matches(&mt) {
        ArchType::Mistral
    } else if phi::matches(&mt) {
        ArchType::Phi
    } else if qwen::matches(&mt) {
        ArchType::Qwen
    } else if gemma::matches(&mt) {
        ArchType::Gemma
    } else {
        ArchType::Unknown
    }
}
