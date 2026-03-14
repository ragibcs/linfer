use std::{fs, path::Path};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::arch::{detect_arch, ArchType};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub arch: ArchType,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
}

#[derive(Debug, Deserialize)]
struct RawModelConfig {
    #[serde(default)]
    model_type: Option<String>,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    vocab_size: usize,
    #[serde(default)]
    max_position_embeddings: Option<usize>,
    #[serde(default)]
    rope_theta: Option<f32>,
    #[serde(default)]
    rms_norm_eps: Option<f32>,
}

impl ModelConfig {
    pub fn from_config_json(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("failed to read config: {}", path.display()))?;
        let raw: RawModelConfig =
            serde_json::from_str(&content).context("invalid HF config.json format")?;

        let arch = detect_arch(raw.model_type.as_deref());
        Ok(Self {
            arch,
            hidden_size: raw.hidden_size,
            num_layers: raw.num_hidden_layers,
            num_heads: raw.num_attention_heads,
            num_kv_heads: raw.num_key_value_heads.unwrap_or(raw.num_attention_heads),
            vocab_size: raw.vocab_size,
            max_seq_len: raw.max_position_embeddings.unwrap_or(4096),
            rope_theta: raw.rope_theta.unwrap_or(10000.0),
            rms_norm_eps: raw.rms_norm_eps.unwrap_or(1e-5),
        })
    }
}
