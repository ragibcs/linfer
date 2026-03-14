use std::{net::SocketAddr, sync::Arc};

use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use chrono::Utc;
use linfer_runtime::{GenerateOptions, InferenceSession};
use serde::{Deserialize, Serialize};

use crate::cmd_compile;

#[derive(Clone)]
struct AppState {
    default_model: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GenerateRequest {
    model: Option<String>,
    prompt: String,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    top_p: Option<f32>,
}

#[derive(Debug, Serialize)]
struct GenerateResponse {
    model: String,
    created_at: String,
    response: String,
    done: bool,
}

#[derive(Debug, Serialize)]
struct TagItem {
    name: String,
    model: String,
    size: u64,
}

#[derive(Debug, Serialize)]
struct TagsResponse {
    models: Vec<TagItem>,
}

pub async fn run(host: &str, port: u16, default_model: Option<&str>) -> Result<()> {
    let state = Arc::new(AppState {
        default_model: default_model.map(|s| s.to_string()),
    });

    let app = Router::new()
        .route("/", get(root))
        .route("/api/tags", get(tags))
        .route("/api/generate", post(generate))
        .with_state(state);

    let addr: SocketAddr = format!("{host}:{port}").parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("linfer serve listening on http://{}", listener.local_addr()?);
    axum::serve(listener, app).await?;
    Ok(())
}

async fn root() -> impl IntoResponse {
    Json(serde_json::json!({"service": "linfer", "status": "ok"}))
}

async fn tags() -> impl IntoResponse {
    match cmd_compile::list_local_models() {
        Ok(models) => {
            let mut out = Vec::new();
            for p in models {
                let size = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
                let name = p
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or_default()
                    .to_string();
                out.push(TagItem {
                    name: name.clone(),
                    model: name,
                    size,
                });
            }
            (StatusCode::OK, Json(TagsResponse { models: out })).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

async fn generate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GenerateRequest>,
) -> impl IntoResponse {
    if req.stream.unwrap_or(false) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "stream=true not supported yet"})),
        )
            .into_response();
    }

    let model_name = req
        .model
        .clone()
        .or_else(|| state.default_model.clone())
        .unwrap_or_default();

    if model_name.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "model is required"})),
        )
            .into_response();
    }

    let model_path = match cmd_compile::resolve_model_spec(&model_name) {
        Ok(p) => p,
        Err(e) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response()
        }
    };

    let mut session = match InferenceSession::load(&model_path, 32, 128, 4096) {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response()
        }
    };

    let opts = GenerateOptions {
        max_tokens: req.max_tokens.unwrap_or(128),
        temperature: req.temperature.unwrap_or(1.0),
        top_k: req.top_k,
        top_p: req.top_p,
    };

    let mut text = String::new();
    session.generate_with_options(&req.prompt, opts, |tok| text.push_str(tok));

    (
        StatusCode::OK,
        Json(GenerateResponse {
            model: model_name,
            created_at: Utc::now().to_rfc3339(),
            response: text.trim().to_string(),
            done: true,
        }),
    )
        .into_response()
}
