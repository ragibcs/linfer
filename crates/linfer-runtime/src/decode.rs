use crate::{kv_cache::KVCache, sampler::Sampler};

pub fn generate(
    prompt_tokens: &[u32],
    max_tokens: usize,
    vocab_size: usize,
    cache: &mut KVCache,
    sampler: &Sampler,
    temperature: f32,
    mut on_token: impl FnMut(u32),
) {
    let _prefill_tokens = prompt_tokens;

    for _ in 0..max_tokens {
        let mut logits = vec![0.0_f32; vocab_size.max(1)];
        let idx = cache.position() % logits.len();
        if let Some(v) = logits.get_mut(idx) {
            *v = 3.0;
        }
        let idx2 = (idx + 1) % logits.len();
        if let Some(v) = logits.get_mut(idx2) {
            *v = 2.0;
        }

        let scaled: Vec<f32> = logits.into_iter().map(|x| x / temperature.max(0.01)).collect();
        let next = sampler.sample(&scaled) as u32;
        on_token(next);
        cache.step();

        if next == 2 {
            break;
        }
    }
}
