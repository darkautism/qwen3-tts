use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;

use crate::config::Config;
use crate::pipeline::{InlineVoiceData, Pipeline, SynthesisParams};

static INDEX_HTML: &str = include_str!("static/index.html");

pub struct AppState {
    pub pipeline: Arc<Mutex<Pipeline>>,
    pub config: Mutex<Config>,
    pub config_path: PathBuf,
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(index_page))
        .route("/v1/audio/speech", post(create_speech))
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
        .route("/api/config", get(get_config).post(update_config))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

#[derive(Debug, Deserialize)]
pub struct CreateSpeechRequest {
    pub model: Option<String>,
    pub input: String,
    #[serde(default = "default_voice")]
    pub voice: String,
    #[serde(default = "default_format")]
    pub response_format: String,
    pub language: Option<String>,
    /// Inline voice data (codec_tokens + ref_text) from web UI
    pub voice_data: Option<VoiceDataPayload>,
}

#[derive(Debug, Deserialize)]
pub struct VoiceDataPayload {
    pub codec_tokens: Vec<Vec<i64>>,
    pub ref_text: Option<String>,
}

fn default_voice() -> String {
    "default".into()
}
fn default_format() -> String {
    "wav".into()
}

#[derive(Serialize)]
struct ModelObject {
    id: String,
    object: String,
    owned_by: String,
}

#[derive(Serialize)]
struct ModelsResponse {
    object: String,
    data: Vec<ModelObject>,
}

async fn create_speech(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateSpeechRequest>,
) -> impl IntoResponse {
    let config = state.config.lock().await.clone();
    let language = req
        .language
        .unwrap_or_else(|| config.defaults.language.clone());

    // Inline voice data takes priority over voice filename
    let (voice, voice_data) = if let Some(vd) = req.voice_data {
        (
            None,
            Some(InlineVoiceData {
                codec_tokens: vd.codec_tokens,
                ref_text: vd.ref_text,
            }),
        )
    } else if req.voice != "default" {
        match crate::voices::resolve_voice(&req.voice) {
            Ok(p) => (Some(p.to_string_lossy().to_string()), None),
            Err(e) => {
                let body = serde_json::json!({
                    "error": {
                        "message": format!("Voice '{}' not found: {}", req.voice, e),
                        "type": "invalid_request_error",
                    }
                });
                return (StatusCode::BAD_REQUEST, Json(body)).into_response();
            }
        }
    } else {
        (None, None)
    };

    let params = SynthesisParams {
        text: req.input,
        language,
        voice,
        voice_data,
        max_tokens: config.defaults.max_tokens,
        temperature: config.defaults.temperature,
        cp_temperature: config.defaults.cp_temperature,
        repetition_penalty: config.defaults.repetition_penalty,
    };

    let mut pipeline = state.pipeline.lock().await;
    match pipeline.synthesize(&params).await {
        Ok(result) => {
            // Encode as WAV in memory
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: result.sample_rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut buf = std::io::Cursor::new(Vec::new());
            {
                let mut writer = hound::WavWriter::new(&mut buf, spec).unwrap();
                for &s in &result.audio_samples {
                    writer.write_sample(s).unwrap();
                }
                writer.finalize().unwrap();
            }

            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "audio/wav")],
                buf.into_inner(),
            )
                .into_response()
        }
        Err(e) => {
            let body = serde_json::json!({
                "error": {
                    "message": e.to_string(),
                    "type": "server_error",
                }
            });
            (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
        }
    }
}

async fn list_models() -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list".into(),
        data: vec![ModelObject {
            id: "qwen3-tts".into(),
            object: "model".into(),
            owned_by: "local".into(),
        }],
    })
}

async fn health() -> &'static str {
    "ok"
}

#[derive(Debug, Serialize, Deserialize)]
struct WorkerConfigPayload {
    talker_host: String,
    talker_port: u16,
    predictor_host: String,
    predictor_port: u16,
    vocoder_host: String,
    vocoder_port: u16,
}

async fn get_config(State(state): State<Arc<AppState>>) -> Json<WorkerConfigPayload> {
    let config = state.config.lock().await;
    let voc = config
        .workers
        .vocoder
        .clone()
        .unwrap_or_else(|| config.workers.predictor.clone());
    Json(WorkerConfigPayload {
        talker_host: config.workers.talker.host.clone(),
        talker_port: config.workers.talker.port,
        predictor_host: config.workers.predictor.host.clone(),
        predictor_port: config.workers.predictor.port,
        vocoder_host: voc.host.clone(),
        vocoder_port: voc.port,
    })
}

async fn update_config(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<WorkerConfigPayload>,
) -> impl IntoResponse {
    use crate::config::WorkerEndpoint;

    // Update config
    {
        let mut config = state.config.lock().await;
        config.workers.talker = WorkerEndpoint {
            host: payload.talker_host.clone(),
            port: payload.talker_port,
        };
        config.workers.predictor = WorkerEndpoint {
            host: payload.predictor_host.clone(),
            port: payload.predictor_port,
        };
        config.workers.vocoder = Some(WorkerEndpoint {
            host: payload.vocoder_host.clone(),
            port: payload.vocoder_port,
        });

        // Save to TOML
        if let Err(e) = config.save(&state.config_path) {
            let body = serde_json::json!({
                "error": format!("Failed to save config: {}", e)
            });
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response();
        }
    }

    // Reconnect pipeline with new config
    let new_config = state.config.lock().await.clone();
    match Pipeline::new(new_config).await {
        Ok(new_pipeline) => {
            let mut pipeline = state.pipeline.lock().await;
            *pipeline = new_pipeline;
            let body = serde_json::json!({"status": "ok", "message": "Config saved and workers reconnected"});
            (StatusCode::OK, Json(body)).into_response()
        }
        Err(e) => {
            let body = serde_json::json!({
                "error": format!("Config saved but reconnect failed: {}. Workers may be offline.", e)
            });
            (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response()
        }
    }
}

async fn index_page() -> Html<&'static str> {
    Html(INDEX_HTML)
}
