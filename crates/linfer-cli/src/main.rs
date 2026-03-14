mod cmd_bench;
mod cmd_compile;
mod cmd_run;
mod cmd_serve;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "linfer")]
#[command(about = "Compile HF models into optimized local bundles")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Compile {
        hf_model_id: String,
        #[arg(long, default_value = "q4")]
        quant: String,
        #[arg(short, long)]
        output: std::path::PathBuf,
    },
    Pull {
        hf_model_id: String,
        #[arg(long, default_value = "q4")]
        quant: String,
    },
    Run {
        model: String,
        prompt: String,
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,
        #[arg(long, default_value_t = 1.0)]
        temperature: f32,
        #[arg(long)]
        top_k: Option<usize>,
        #[arg(long)]
        top_p: Option<f32>,
    },
    Bench {
        model: String,
        #[arg(long, default_value_t = 200)]
        tokens: usize,
        #[arg(long)]
        compare: Option<String>,
    },
    Info {
        model: String,
    },
    Ps,
    List,
    Rm {
        model: String,
    },
    Serve {
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 11434)]
        port: u16,
        #[arg(long)]
        model: Option<String>,
    },
    ListArchs,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compile {
            hf_model_id,
            quant,
            output,
        } => cmd_compile::run(&hf_model_id, &quant, &output),
        Commands::Pull { hf_model_id, quant } => {
            let out = cmd_compile::pull(&hf_model_id, &quant)?;
            println!("pulled: {}", out.display());
            Ok(())
        }
        Commands::Run {
            model,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
        } => {
            let model_path = cmd_compile::resolve_model_spec(&model)?;
            cmd_run::run(&model_path, &prompt, max_tokens, temperature, top_k, top_p)
        }
        Commands::Bench {
            model,
            tokens,
            compare,
        } => {
            let model_path = cmd_compile::resolve_model_spec(&model)?;
            cmd_bench::run(&model_path, tokens, compare.as_deref())
        }
        Commands::Info { model } => cmd_compile::info_model(&model),
        Commands::Ps | Commands::List => {
            let models = cmd_compile::list_local_models()?;
            if models.is_empty() {
                println!("no local models");
            } else {
                for m in models {
                    println!("{}", m.display());
                }
            }
            Ok(())
        }
        Commands::Rm { model } => {
            let removed = cmd_compile::remove_local_model(&model)?;
            println!("removed: {}", removed.display());
            Ok(())
        }
        Commands::Serve { host, port, model } => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(cmd_serve::run(&host, port, model.as_deref()))
        }
        Commands::ListArchs => {
            println!("llama\nmistral\nphi\nqwen\ngemma");
            Ok(())
        }
    }
}
