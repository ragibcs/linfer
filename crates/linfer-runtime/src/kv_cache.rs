#[derive(Debug, Clone)]
pub struct KVCache {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    pos: usize,
    max: usize,
}

impl KVCache {
    pub fn new(num_layers: usize, layer_size: usize, max_seq_len: usize) -> Self {
        let alloc = || vec![0.0_f32; max_seq_len * layer_size];
        Self {
            keys: (0..num_layers).map(|_| alloc()).collect(),
            values: (0..num_layers).map(|_| alloc()).collect(),
            pos: 0,
            max: max_seq_len,
        }
    }

    pub fn append(&mut self, layer: usize, key: &[f32], value: &[f32]) {
        if self.pos >= self.max || layer >= self.keys.len() {
            return;
        }
        let layer_size = self.keys[layer].len() / self.max;
        let start = self.pos * layer_size;
        let end = start.saturating_add(layer_size);

        let kdst = &mut self.keys[layer][start..end];
        let vdst = &mut self.values[layer][start..end];

        let kcopy = key.len().min(kdst.len());
        let vcopy = value.len().min(vdst.len());
        kdst[..kcopy].copy_from_slice(&key[..kcopy]);
        vdst[..vcopy].copy_from_slice(&value[..vcopy]);
    }

    pub fn step(&mut self) {
        self.pos = (self.pos + 1).min(self.max);
    }

    pub fn position(&self) -> usize {
        self.pos
    }
}
