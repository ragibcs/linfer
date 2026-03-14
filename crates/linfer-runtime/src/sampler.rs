use rand::{rng, Rng};

#[derive(Debug, Clone)]
pub enum Sampler {
    Greedy,
    TopK { k: usize },
    TopP { p: f32 },
}

impl Sampler {
    pub fn sample(&self, logits: &[f32]) -> usize {
        match self {
            Sampler::Greedy => argmax(logits),
            Sampler::TopK { k } => sample_topk(logits, *k),
            Sampler::TopP { p } => sample_topp(logits, *p),
        }
    }
}

fn argmax(xs: &[f32]) -> usize {
    let mut best_i = 0;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in xs.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best_i = i;
        }
    }
    best_i
}

fn sample_topk(logits: &[f32], k: usize) -> usize {
    let mut pairs: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
    let top = &pairs[..k.min(pairs.len())];
    if top.is_empty() {
        return 0;
    }
    let idx = rng().random_range(0..top.len());
    top[idx].0
}

fn sample_topp(logits: &[f32], p: f32) -> usize {
    let mut probs: Vec<(usize, f32)> = softmax(logits).into_iter().enumerate().collect();
    probs.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut acc = 0.0;
    let mut cutoff = probs.len();
    for (i, (_, pr)) in probs.iter().enumerate() {
        acc += pr;
        if acc >= p {
            cutoff = i + 1;
            break;
        }
    }

    let truncated = &probs[..cutoff.max(1).min(probs.len())];
    let idx = rng().random_range(0..truncated.len());
    truncated[idx].0
}

fn softmax(xs: &[f32]) -> Vec<f32> {
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = xs.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|x| x / sum.max(1e-8)).collect()
}
