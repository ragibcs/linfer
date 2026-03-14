use std::path::Path;

use anyhow::{Context, Result};
use tokenizers::Tokenizer;

#[derive(Clone)]
pub struct TokenizerWrapper {
    tokenizer: Tokenizer,
}

impl TokenizerWrapper {
    pub fn from_file(path: &Path) -> Result<Self> {
        let tokenizer =
            Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!(e.to_string()))?;
        Ok(Self { tokenizer })
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let tokenizer: Tokenizer =
            serde_json::from_slice(bytes).map_err(|e| anyhow::anyhow!(e.to_string()))?;
        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
            .context("tokenization failed")?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
            .context("detokenization failed")
    }
}
