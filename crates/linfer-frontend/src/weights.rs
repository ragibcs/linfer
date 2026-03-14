use std::{collections::HashMap, fs::File, path::Path};

use anyhow::{Context, Result};
use memmap2::Mmap;

#[derive(Debug)]
pub struct TensorStore {
    map: Mmap,
    index: HashMap<String, (usize, usize)>,
}

impl TensorStore {
    pub fn mmap(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("failed to open safetensors file: {}", path.display()))?;
        let map = unsafe { Mmap::map(&file) }.context("failed to mmap safetensors file")?;
        Ok(Self {
            map,
            index: HashMap::new(),
        })
    }

    pub fn get(&self, name: &str) -> Option<&[u8]> {
        self.index.get(name).map(|(start, len)| {
            let end = start.saturating_add(*len);
            &self.map[*start..end]
        })
    }

    pub fn register(&mut self, name: impl Into<String>, start: usize, len: usize) {
        self.index.insert(name.into(), (start, len));
    }
}
