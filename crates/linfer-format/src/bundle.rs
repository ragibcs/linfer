use std::{fs::File, io::{Read, Write}, path::Path};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::header::Header;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledModel {
    pub config_json: String,
    pub tokenizer_bytes: Vec<u8>,
    pub graph_bytes: Vec<u8>,
    pub quantized_weights: Vec<u8>,
}

pub fn save(model: &CompiledModel, path: &Path) -> Result<()> {
    let mut file = File::create(path)
        .with_context(|| format!("failed to create bundle: {}", path.display()))?;

    let header = Header {
        version: 1,
        flags: 0,
    }
    .encode();

    file.write_all(&header)?;

    write_blob(&mut file, model.config_json.as_bytes())?;
    write_blob(&mut file, &model.tokenizer_bytes)?;
    write_blob(&mut file, &model.graph_bytes)?;
    write_blob(&mut file, &model.quantized_weights)?;

    Ok(())
}

pub fn load(path: &Path) -> Result<CompiledModel> {
    let mut file = File::open(path)
        .with_context(|| format!("failed to open bundle: {}", path.display()))?;

    let mut header = [0_u8; 18];
    file.read_exact(&mut header)?;
    if Header::decode(&header).is_none() {
        bail!("invalid bundle header");
    }

    let config_json = String::from_utf8(read_blob(&mut file)?)
        .context("invalid utf8 in config blob")?;
    let tokenizer_bytes = read_blob(&mut file)?;
    let graph_bytes = read_blob(&mut file)?;
    let quantized_weights = read_blob(&mut file)?;

    Ok(CompiledModel {
        config_json,
        tokenizer_bytes,
        graph_bytes,
        quantized_weights,
    })
}

fn write_blob(mut w: impl Write, blob: &[u8]) -> Result<()> {
    let len = blob.len() as u64;
    w.write_all(&len.to_le_bytes())?;
    w.write_all(blob)?;
    Ok(())
}

fn read_blob(mut r: impl Read) -> Result<Vec<u8>> {
    let mut len_buf = [0_u8; 8];
    r.read_exact(&mut len_buf)?;
    let len = u64::from_le_bytes(len_buf) as usize;
    let mut out = vec![0_u8; len];
    r.read_exact(&mut out)?;
    Ok(out)
}
