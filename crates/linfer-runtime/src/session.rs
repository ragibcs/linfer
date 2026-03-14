use std::path::Path;

use anyhow::{Context, Result};
use linfer_format::CompiledModel;
use serde::Deserialize;

use crate::{decode, kv_cache::KVCache, sampler::Sampler};

#[derive(Debug, Clone, Copy)]
pub struct GenerateOptions {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            temperature: 1.0,
            top_k: None,
            top_p: None,
        }
    }
}

#[derive(Debug, Deserialize)]
struct ConfigRoot {
    model_config: Option<ModelConfigJson>,
}

#[derive(Debug, Deserialize)]
struct ModelConfigJson {
    num_layers: Option<usize>,
    hidden_size: Option<usize>,
    num_heads: Option<usize>,
    max_seq_len: Option<usize>,
    vocab_size: Option<usize>,
}

pub struct InferenceSession {
    model: CompiledModel,
    cache: KVCache,
    sampler: Sampler,
    vocab_size: usize,
}

impl InferenceSession {
    pub fn from_compiled(model: CompiledModel, num_layers: usize, layer_size: usize, max_seq_len: usize) -> Self {
        Self {
            model,
            cache: KVCache::new(num_layers, layer_size, max_seq_len),
            sampler: Sampler::Greedy,
            vocab_size: 32000,
        }
    }

    pub fn load(path: &Path, num_layers: usize, layer_size: usize, max_seq_len: usize) -> Result<Self> {
        let model = linfer_format::load(path)?;

        let mut cfg_layers = num_layers;
        let mut cfg_layer_size = layer_size;
        let mut cfg_max_seq = max_seq_len;
        let mut cfg_vocab = 32000usize;

        if let Ok(root) = serde_json::from_str::<ConfigRoot>(&model.config_json) {
            if let Some(mc) = root.model_config {
                if let Some(v) = mc.num_layers {
                    cfg_layers = v.max(1);
                }
                if let (Some(h), Some(nh)) = (mc.hidden_size, mc.num_heads) {
                    let hd = h / nh.max(1);
                    cfg_layer_size = (nh.max(1) * hd).max(64);
                } else if let Some(h) = mc.hidden_size {
                    cfg_layer_size = h.max(64);
                }
                if let Some(v) = mc.max_seq_len {
                    cfg_max_seq = v.max(16);
                }
                if let Some(v) = mc.vocab_size {
                    cfg_vocab = v.max(128);
                }
            }
        }

        let mut s = Self::from_compiled(model, cfg_layers, cfg_layer_size, cfg_max_seq);
        s.vocab_size = cfg_vocab;
        Ok(s)
    }

    pub fn with_sampler(mut self, sampler: Sampler) -> Self {
        self.sampler = sampler;
        self
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        mut on_token: impl FnMut(&str),
    ) {
        self.generate_with_options(
            prompt,
            GenerateOptions {
                max_tokens,
                ..GenerateOptions::default()
            },
            &mut on_token,
        )
    }

    pub fn generate_with_options(
        &mut self,
        prompt: &str,
        opts: GenerateOptions,
        mut on_token: impl FnMut(&str),
    ) {
        let prompt_tokens: Vec<u32> = prompt.bytes().map(|b| b as u32).collect();

        let sampler = sampler_from_options(opts, &self.sampler);

        decode::generate(
            &prompt_tokens,
            opts.max_tokens,
            self.vocab_size,
            &mut self.cache,
            &sampler,
            opts.temperature.max(0.01),
            |id| {
                let s = format!("{id} ");
                on_token(&s);
            },
        );
    }

    pub fn model(&self) -> &CompiledModel {
        &self.model
    }

    pub fn model_id(&self) -> Result<Option<String>> {
        let v: serde_json::Value =
            serde_json::from_str(&self.model.config_json).context("invalid model config json")?;
        Ok(v.get("hf_model_id")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string()))
    }
}

fn sampler_from_options(opts: GenerateOptions, fallback: &Sampler) -> Sampler {
    if let Some(k) = opts.top_k {
        return Sampler::TopK { k: k.max(1) };
    }
    if let Some(p) = opts.top_p {
        return Sampler::TopP {
            p: p.clamp(0.01, 1.0),
        };
    }
    fallback.clone()
}
