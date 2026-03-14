use std::path::Path;

use anyhow::Result;
use linfer_runtime::{GenerateOptions, InferenceSession};

pub fn run(
    model: &Path,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
) -> Result<()> {
    let mut session = InferenceSession::load(model, 32, 128, 4096)?;
    let opts = GenerateOptions {
        max_tokens,
        temperature,
        top_k,
        top_p,
    };

    print!("> ");
    session.generate_with_options(prompt, opts, |t| print!("{t}"));
    println!();
    Ok(())
}
