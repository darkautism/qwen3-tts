use axum::{
    extract::{Multipart, State},
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
use crate::pipeline::{ChunkMode, InlineVoiceData, Pipeline, SynthesisParams};

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
        .route("/api/encode-voice", post(encode_voice))
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
    /// Text chunking mode: "none", "2" (default), "4"
    pub chunk_mode: Option<String>,
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

    let chunk_mode = req
        .chunk_mode
        .as_deref()
        .unwrap_or("2")
        .parse::<ChunkMode>()
        .unwrap_or_default();

    let params = SynthesisParams {
        text: req.input,
        language,
        voice,
        voice_data,
        max_tokens: config.defaults.max_tokens,
        temperature: config.defaults.temperature,
        cp_temperature: config.defaults.cp_temperature,
        repetition_penalty: config.defaults.repetition_penalty,
        chunk_mode,
    };

    let mut pipeline = state.pipeline.lock().await;
    match pipeline.synthesize(&params).await {
        Ok(result) => {
            let audio_duration = result.audio_samples.len() as f32 / result.sample_rate as f32;

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
                [
                    (header::CONTENT_TYPE, "audio/wav".to_string()),
                    (
                        header::HeaderName::from_static("x-audio-duration"),
                        format!("{:.2}", audio_duration),
                    ),
                ],
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

async fn encode_voice(
    State(_state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    let mut audio_data: Option<Vec<u8>> = None;
    let mut ref_text: Option<String> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "audio" => {
                audio_data = field.bytes().await.ok().map(|b| b.to_vec());
            }
            "ref_text" => {
                ref_text = field.text().await.ok();
            }
            _ => {}
        }
    }

    let audio_bytes = match audio_data {
        Some(d) if !d.is_empty() => d,
        _ => {
            let body = serde_json::json!({"error": "Missing 'audio' field (WAV file)"});
            return (StatusCode::BAD_REQUEST, Json(body)).into_response();
        }
    };
    let ref_text = match ref_text {
        Some(t) if !t.is_empty() => t,
        _ => {
            let body = serde_json::json!({"error": "Missing 'ref_text' field"});
            return (StatusCode::BAD_REQUEST, Json(body)).into_response();
        }
    };

    // Write to temp file, encode, clean up
    let tmp_path = format!("/tmp/qwen3_encode_{}.wav", std::process::id());
    if let Err(e) = std::fs::write(&tmp_path, &audio_bytes) {
        let body = serde_json::json!({"error": format!("Failed to write temp file: {}", e)});
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response();
    }

    let result = tokio::task::spawn_blocking(move || {
        let model_path = crate::speech_tokenizer::resolve_model_path("kautism/qwen3-tts-rk3588")?;
        let tokenizer = crate::speech_tokenizer::SpeechTokenizer::load(&model_path)?;
        let codes = tokenizer.encode_wav(&tmp_path)?;
        let _ = std::fs::remove_file(&tmp_path);
        Ok::<_, anyhow::Error>(codes)
    })
    .await;

    match result {
        Ok(Ok(codes)) => {
            let body = serde_json::json!({
                "ref_text": ref_text,
                "codec_tokens": codes,
            });
            (StatusCode::OK, Json(body)).into_response()
        }
        Ok(Err(e)) => {
            let body = serde_json::json!({"error": format!("Encode failed: {}", e)});
            (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
        }
        Err(e) => {
            let body = serde_json::json!({"error": format!("Task failed: {}", e)});
            (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
        }
    }
}

async fn index_page() -> Html<&'static str> {
    Html(INDEX_HTML)
}
