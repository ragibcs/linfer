use std::{path::Path, time::Instant};

use anyhow::Result;
use linfer_runtime::{GenerateOptions, InferenceSession};

pub fn run(model: &Path, tokens: usize, compare: Option<&str>) -> Result<()> {
    let mut session = InferenceSession::load(model, 32, 128, 4096)?;

    let start = Instant::now();
    let mut generated = 0usize;
    session.generate_with_options(
        "benchmark prompt",
        GenerateOptions {
            max_tokens: tokens,
            ..GenerateOptions::default()
        },
        |_| generated += 1,
    );
    let elapsed = start.elapsed().as_secs_f64().max(1e-6);
    let tps = generated as f64 / elapsed;

    println!("═══════════════════════════════════════════════");
    println!("  linfer benchmark");
    println!("═══════════════════════════════════════════════");
    println!("  Tokens    : {}", generated);
    println!("  linfer    : {:.2} tok/s", tps);

    if let Some(engine) = compare {
        println!("  compare   : {engine}");
    }

    println!("═══════════════════════════════════════════════");
    Ok(())
}
