use axum::{
    extract::State,
    http::{header, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::config::Config;
use crate::pipeline::{Pipeline, SynthesisParams};

pub struct AppState {
    pub pipeline: Arc<Mutex<Pipeline>>,
    pub config: Config,
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/audio/speech", post(create_speech))
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
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
    let language = req
        .language
        .unwrap_or_else(|| state.config.defaults.language.clone());

    let voice = if req.voice != "default" {
        match crate::voices::resolve_voice(&req.voice) {
            Ok(p) => Some(p.to_string_lossy().to_string()),
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
        None
    };

    let params = SynthesisParams {
        text: req.input,
        language,
        voice,
        max_tokens: state.config.defaults.max_tokens,
        temperature: state.config.defaults.temperature,
        cp_temperature: state.config.defaults.cp_temperature,
        repetition_penalty: state.config.defaults.repetition_penalty,
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
