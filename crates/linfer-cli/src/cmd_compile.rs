use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use hf_hub::{api::sync::ApiBuilder, Cache};
use linfer_compiler::Pipeline;
use linfer_format::{save, CompiledModel};
use linfer_ir::{Graph, Op};
use safetensors::SafeTensors;

pub fn run(hf_model_id: &str, quant: &str, output: &Path) -> Result<()> {
    validate_quant(quant)?;

    let cache_dir = cache_dir()?;
    fs::create_dir_all(&cache_dir)
        .with_context(|| format!("failed to create HF cache dir: {}", cache_dir.display()))?;

    let api = ApiBuilder::from_cache(Cache::new(cache_dir))
        .with_progress(false)
        .build()
        .context("failed to initialize hf-hub API")?;
    let repo = api.model(hf_model_id.to_owned());

    let config_path = repo
        .get("config.json")
        .with_context(|| format!("failed to fetch config.json for {hf_model_id}"))?;
    let tokenizer_path = repo
        .get("tokenizer.json")
        .with_context(|| format!("failed to fetch tokenizer.json for {hf_model_id}"))?;

    let model_config = linfer_frontend::ModelConfig::from_config_json(&config_path)
        .with_context(|| format!("failed to parse config.json for {hf_model_id}"))?;
    let tokenizer_bytes = fs::read(&tokenizer_path)
        .with_context(|| format!("failed to read tokenizer: {}", tokenizer_path.display()))?;
    let _ = linfer_frontend::TokenizerWrapper::from_bytes(&tokenizer_bytes)
        .context("failed to parse tokenizer.json")?;

    let weight_files = resolve_weight_files(&repo, hf_model_id)?;
    let quantized_weights = collect_weight_payload(&weight_files)?;

    let (compiled_graph, compile_report) = build_and_compile_graph(&model_config, quant);
    let graph_bytes = serde_json::to_vec(&serde_json::json!({
        "graph": compiled_graph,
        "pipeline_report": compile_report.applied,
        "weight_files": weight_files
            .iter()
            .map(|p| p.file_name().and_then(|s| s.to_str()).unwrap_or_default().to_string())
            .collect::<Vec<_>>()
    }))?;

    let config = serde_json::json!({
        "hf_model_id": hf_model_id,
        "quant": quant,
        "compiled_by": "linfer-cli",
        "model_config": model_config,
    })
    .to_string();

    let model = CompiledModel {
        config_json: config,
        tokenizer_bytes,
        graph_bytes,
        quantized_weights,
    };

    let out = if output.extension().is_some() {
        output.to_path_buf()
    } else {
        output.with_extension("lnf")
    };

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }

    save(&model, &out)?;
    println!("compiled: {}", out.display());
    println!("weights files: {}", weight_files.len());
    println!("passes: {:?}", compile_report.applied);
    Ok(())
}

pub fn pull(hf_model_id: &str, quant: &str) -> Result<PathBuf> {
    let out = default_model_path(hf_model_id)?;
    run(hf_model_id, quant, &out)?;
    Ok(out)
}

pub fn info_model(model_spec: &str) -> Result<()> {
    let path = resolve_model_spec(model_spec)?;
    info(&path)
}

pub fn info(model: &Path) -> Result<()> {
    let data = linfer_format::load(model)?;
    println!("bundle: {}", model.display());
    println!("config bytes: {}", data.config_json.len());
    println!("tokenizer bytes: {}", data.tokenizer_bytes.len());
    println!("graph bytes: {}", data.graph_bytes.len());
    println!("weights bytes: {}", data.quantized_weights.len());

    if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&data.graph_bytes) {
        if let Some(graph) = v.get("graph") {
            let nodes = graph
                .get("nodes")
                .and_then(|x| x.as_array())
                .map(|x| x.len())
                .unwrap_or(0);
            let edges = graph
                .get("edges")
                .and_then(|x| x.as_array())
                .map(|x| x.len())
                .unwrap_or(0);
            let outputs = graph
                .get("outputs")
                .and_then(|x| x.as_array())
                .map(|x| x.len())
                .unwrap_or(0);

            println!("graph summary:");
            println!("  nodes: {}", nodes);
            println!("  edges: {}", edges);
            println!("  outputs: {}", outputs);

            let mut op_hist: BTreeMap<String, usize> = BTreeMap::new();
            if let Some(ns) = graph.get("nodes").and_then(|x| x.as_array()) {
                for n in ns {
                    if let Some(op_obj) = n.get("op").and_then(|x| x.as_object()) {
                        for k in op_obj.keys() {
                            *op_hist.entry(k.clone()).or_insert(0) += 1;
                        }
                    }
                }
            }

            if !op_hist.is_empty() {
                println!("  op histogram:");
                for (k, c) in op_hist {
                    println!("    {}: {}", k, c);
                }
            }
        }

        if let Some(report) = v.get("pipeline_report").and_then(|x| x.as_array()) {
            println!("pipeline report:");
            for entry in report {
                if let Some(arr) = entry.as_array() {
                    if arr.len() == 2 {
                        let name = arr[0].as_str().unwrap_or("<unknown>");
                        let cnt = arr[1].as_u64().unwrap_or(0);
                        println!("  {}: {}", name, cnt);
                    }
                }
            }
        }

        if let Some(files) = v.get("weight_files").and_then(|x| x.as_array()) {
            println!("weight files:");
            for f in files {
                if let Some(name) = f.as_str() {
                    println!("  {}", name);
                }
            }
        }
    }

    Ok(())
}

pub fn resolve_model_spec(model_spec: &str) -> Result<PathBuf> {
    let as_path = PathBuf::from(model_spec);
    if as_path.exists() {
        return Ok(as_path);
    }

    let models = models_dir()?;
    let mut candidates = Vec::new();

    if model_spec.ends_with(".lnf") {
        candidates.push(models.join(model_spec));
    } else {
        candidates.push(models.join(format!("{model_spec}.lnf")));
        candidates.push(models.join(format!("{}.lnf", canonical_model_name(model_spec))));
    }

    for p in candidates {
        if p.exists() {
            return Ok(p);
        }
    }

    anyhow::bail!(
        "model '{}' not found. Use a local .lnf path or run `linfer pull {}`",
        model_spec,
        model_spec
    )
}

pub fn list_local_models() -> Result<Vec<PathBuf>> {
    let dir = models_dir()?;
    fs::create_dir_all(&dir)?;
    let mut out = Vec::new();

    for entry in fs::read_dir(&dir)? {
        let path = entry?.path();
        if path.extension().and_then(|s| s.to_str()) == Some("lnf") {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

pub fn remove_local_model(model_spec: &str) -> Result<PathBuf> {
    let path = resolve_model_spec(model_spec)?;
    fs::remove_file(&path).with_context(|| format!("failed to remove {}", path.display()))?;
    Ok(path)
}

fn validate_quant(quant: &str) -> Result<()> {
    match quant {
        "q4" | "q8" => Ok(()),
        other => anyhow::bail!("unsupported quant '{}'; expected 'q4' or 'q8'", other),
    }
}

fn resolve_weight_files(repo: &hf_hub::api::sync::ApiRepo, hf_model_id: &str) -> Result<Vec<PathBuf>> {
    if let Ok(index_path) = repo.get("model.safetensors.index.json") {
        let raw = fs::read_to_string(&index_path)
            .with_context(|| format!("failed to read index: {}", index_path.display()))?;
        let v: serde_json::Value =
            serde_json::from_str(&raw).context("invalid model.safetensors.index.json")?;
        let weight_map = v
            .get("weight_map")
            .and_then(|m| m.as_object())
            .context("index missing weight_map object")?;

        let mut names = BTreeSet::new();
        for name in weight_map.values().filter_map(|x| x.as_str()) {
            names.insert(name.to_string());
        }

        let mut paths = Vec::with_capacity(names.len());
        for name in names {
            let path = repo
                .get(&name)
                .with_context(|| format!("failed to fetch weight shard '{name}' for {hf_model_id}"))?;
            paths.push(path);
        }
        if !paths.is_empty() {
            return Ok(paths);
        }
    }

    if let Ok(single) = repo.get("model.safetensors") {
        return Ok(vec![single]);
    }

    let info = repo
        .info()
        .with_context(|| format!("failed to inspect model files for {hf_model_id}"))?;

    let mut names: Vec<String> = info
        .siblings
        .into_iter()
        .map(|s| s.rfilename)
        .filter(|name| name.ends_with(".safetensors"))
        .collect();
    names.sort();
    names.dedup();

    if names.is_empty() {
        anyhow::bail!("no .safetensors files found in model repo {hf_model_id}");
    }

    let mut paths = Vec::with_capacity(names.len());
    for name in names {
        let path = repo
            .get(&name)
            .with_context(|| format!("failed to fetch weight file '{name}' for {hf_model_id}"))?;
        paths.push(path);
    }

    Ok(paths)
}

fn collect_weight_payload(files: &[PathBuf]) -> Result<Vec<u8>> {
    let mut out = Vec::new();

    for path in files {
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read safetensors file: {}", path.display()))?;
        let _ = SafeTensors::deserialize(&bytes)
            .with_context(|| format!("invalid safetensors file: {}", path.display()))?;

        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .context("invalid shard filename")?;
        let name_bytes = name.as_bytes();
        let name_len = name_bytes.len() as u32;
        let blob_len = bytes.len() as u64;

        out.extend_from_slice(&name_len.to_le_bytes());
        out.extend_from_slice(name_bytes);
        out.extend_from_slice(&blob_len.to_le_bytes());
        out.extend_from_slice(&bytes);
    }

    Ok(out)
}

fn build_and_compile_graph(
    config: &linfer_frontend::ModelConfig,
    quant: &str,
) -> (Graph, linfer_compiler::CompileReport) {
    let mut g = Graph::default();

    let t_input = 0usize;
    let t_wq = 1usize;
    let t_wk = 2usize;
    let t_wv = 3usize;
    let t_q = 4usize;
    let t_k = 5usize;
    let t_v = 6usize;
    let t_attn = 7usize;
    let t_residual = 8usize;
    let t_norm = 9usize;
    let t_wgate = 10usize;
    let t_wup = 11usize;
    let t_gate = 12usize;
    let t_up = 13usize;
    let t_swiglu = 14usize;
    let t_wdown = 15usize;
    let t_mlp = 16usize;

    let n_q = g.add_op(Op::MatMul {
        lhs: t_input,
        rhs: t_wq,
        out: t_q,
    });
    let n_k = g.add_op(Op::MatMul {
        lhs: t_input,
        rhs: t_wk,
        out: t_k,
    });
    let n_v = g.add_op(Op::MatMul {
        lhs: t_input,
        rhs: t_wv,
        out: t_v,
    });
    let n_sdpa = g.add_op(Op::SDPA {
        q: t_q,
        k: t_k,
        v: t_v,
        out: t_attn,
    });

    let n_res = g.add_op(Op::Residual {
        a: t_input,
        b: t_attn,
        out: t_residual,
    });
    let n_norm = g.add_op(Op::RMSNorm {
        input: t_residual,
        weight: 17usize,
        out: t_norm,
        eps: config.rms_norm_eps,
    });

    let n_gate = g.add_op(Op::MatMul {
        lhs: t_norm,
        rhs: t_wgate,
        out: t_gate,
    });
    let n_up = g.add_op(Op::MatMul {
        lhs: t_norm,
        rhs: t_wup,
        out: t_up,
    });
    let n_swiglu = g.add_op(Op::SwiGLU {
        gate: t_gate,
        up: t_up,
        out: t_swiglu,
    });
    let n_mlp = g.add_op(Op::MatMul {
        lhs: t_swiglu,
        rhs: t_wdown,
        out: t_mlp,
    });

    let n_rope = g.add_op(Op::RoPE {
        q: t_q,
        k: t_k,
        pos: 0,
    });
    let n_kv = g.add_op(Op::KVAppend {
        k: t_k,
        v: t_v,
        layer: 0,
    });

    g.connect(n_q, n_sdpa, t_q);
    g.connect(n_k, n_sdpa, t_k);
    g.connect(n_v, n_sdpa, t_v);
    g.connect(n_sdpa, n_res, t_attn);
    g.connect(n_res, n_norm, t_residual);
    g.connect(n_norm, n_gate, t_norm);
    g.connect(n_norm, n_up, t_norm);
    g.connect(n_gate, n_swiglu, t_gate);
    g.connect(n_up, n_swiglu, t_up);
    g.connect(n_swiglu, n_mlp, t_swiglu);
    g.connect(n_rope, n_kv, t_k);

    g.mark_output(t_mlp);

    if quant == "q8" {
        g.mark_output(t_attn);
    }

    let pipeline = Pipeline::default();
    let report = pipeline.compile(&mut g);
    (g, report)
}

fn default_model_path(hf_model_id: &str) -> Result<PathBuf> {
    Ok(models_dir()?.join(format!("{}.lnf", canonical_model_name(hf_model_id))))
}

fn canonical_model_name(hf_model_id: &str) -> String {
    hf_model_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn app_data_dir() -> Result<PathBuf> {
    if let Ok(explicit) = std::env::var("LINFER_HOME") {
        return Ok(PathBuf::from(explicit));
    }

    let mut probe = Vec::new();
    if let Some(mut d) = dirs::data_local_dir().or_else(dirs::data_dir) {
        d.push("linfer");
        probe.push(d);
    }

    let mut cwd = std::env::current_dir().context("failed to determine current directory")?;
    cwd.push(".linfer");
    probe.push(cwd);

    for p in probe {
        if fs::create_dir_all(&p).is_ok() {
            return Ok(p);
        }
    }

    anyhow::bail!(
        "failed to initialize linfer data directory; set LINFER_HOME to a writable path"
    )
}

fn cache_dir() -> Result<PathBuf> {
    Ok(app_data_dir()?.join("cache"))
}

fn models_dir() -> Result<PathBuf> {
    Ok(app_data_dir()?.join("models"))
}
