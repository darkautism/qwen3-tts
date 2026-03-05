pub mod candle_qwen3;
pub mod code_predictor;
pub mod code_predictor_candle;
#[cfg(any(feature = "ggml-backend", feature = "ggml-predictor"))]
pub mod code_predictor_ggml;
pub mod embedder;
pub mod sampling;
pub mod talker;
#[cfg(feature = "ggml-backend")]
pub mod talker_llamacpp;
pub mod vocoder;

use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

use crate::protocol::{RefCodecTokens, Request, Response, ResponseData, HIDDEN_SIZE};

pub struct Worker {
    state: WorkerState,
    #[allow(dead_code)]
    models_dir: PathBuf,
}

enum WorkerState {
    Talker(TalkerState),
    Predictor(PredictorState),
    Vocoder(VocoderState),
}

struct TalkerState {
    tokenizer: tokenizers::Tokenizer,
    embedder: embedder::TextEmbedder,
    #[cfg(feature = "ggml-backend")]
    talker: talker_llamacpp::TalkerLlamaCpp,
    #[cfg(not(feature = "ggml-backend"))]
    talker: talker::TalkerLlamaCpp,
    last_hidden: Option<Array1<f32>>,
    ref_codec_tokens: Option<Array2<i64>>,
    ref_text: Option<String>,
    past_tokens: Vec<i32>,
    feedback_decode_buf: Vec<f32>,
}

struct PredictorState {
    code_predictor: PredictorBackend,
    codec_embedding: Array2<f32>,
    tts_pad_embed: Array1<f32>,
    feedback_buf: Array1<f32>,
    hidden_decode_buf: Vec<f32>,
}

/// Abstraction over Candle, GGML, or ONNX code predictor
enum PredictorBackend {
    Candle(code_predictor_candle::CodePredictorCandle),
    #[cfg(any(feature = "ggml-backend", feature = "ggml-predictor"))]
    Ggml(code_predictor_ggml::CodePredictorGgml),
    Onnx(code_predictor::CodePredictor),
}

impl PredictorBackend {
    fn predict(
        &mut self,
        hidden: &[f32],
        _code_0: i32,
        code_0_embed: &[f32],
        temperature: f32,
    ) -> Result<Vec<i32>> {
        match self {
            PredictorBackend::Candle(cp) => cp.predict(hidden, code_0_embed, temperature),
            #[cfg(any(feature = "ggml-backend", feature = "ggml-predictor"))]
            PredictorBackend::Ggml(cp) => {
                let hidden_arr = Array1::from_vec(hidden.to_vec());
                cp.predict(&hidden_arr, _code_0, temperature)
            }
            PredictorBackend::Onnx(cp) => {
                let hidden_arr = Array1::from_vec(hidden.to_vec());
                let code_0_embed_arr = Array1::from_vec(code_0_embed.to_vec());
                cp.predict(&hidden_arr, &code_0_embed_arr, temperature)
            }
        }
    }

    fn codec_embeddings(&self) -> &Vec<Array2<f32>> {
        match self {
            PredictorBackend::Candle(cp) => &cp.codec_embeddings,
            #[cfg(any(feature = "ggml-backend", feature = "ggml-predictor"))]
            PredictorBackend::Ggml(cp) => &cp.codec_embeddings,
            PredictorBackend::Onnx(cp) => &cp.codec_embeddings,
        }
    }
}

struct VocoderState {
    vocoder: vocoder::Vocoder,
}

impl Worker {
    pub fn new(role: &str, models_dir: &Path) -> Result<Self> {
        let state = match role {
            "talker" => {
                let tok_path = find_tokenizer(models_dir)?;
                info!("Loading tokenizer: {}", tok_path.display());
                let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
                    .map_err(|e| anyhow::anyhow!("Load tokenizer: {}", e))?;

                let emb_dir = models_dir.join("embeddings");
                let embedder = embedder::TextEmbedder::load(&emb_dir)?;

                let talker_path = find_talker_model(models_dir)?;
                let n_ctx = 4096i32;
                let n_threads = num_cpus().min(4) as i32;
                info!(
                    "Loading talker: {} (threads={})",
                    talker_path.display(),
                    n_threads
                );

                #[cfg(feature = "ggml-backend")]
                let talker = {
                    info!("Using llama.cpp talker (ggml-backend feature)");
                    talker_llamacpp::TalkerLlamaCpp::load(&talker_path, n_ctx, n_threads)?
                };
                #[cfg(not(feature = "ggml-backend"))]
                let talker = talker::TalkerLlamaCpp::load(&talker_path, n_ctx, n_threads)?;

                WorkerState::Talker(TalkerState {
                    tokenizer,
                    embedder,
                    talker,
                    last_hidden: None,
                    ref_codec_tokens: None,
                    ref_text: None,
                    past_tokens: Vec::new(),
                    feedback_decode_buf: Vec::with_capacity(HIDDEN_SIZE),
                })
            }
            "predictor" => {
                let cp_dir = models_dir.join("code_predictor");

                // Try GGML C (ggml-backend feature) → Candle GGUF → ONNX (fallback)
                let gguf_exists = [
                    "code-predictor-q8_0.gguf",
                    "code-predictor-q4_0.gguf",
                    "qwen3-tts-0.6b-q8_0.gguf",
                    "qwen3-tts-0.6b-q4_0.gguf",
                    "qwen3-tts-0.6b-f16.gguf",
                ]
                .iter()
                .any(|f| cp_dir.join(f).exists());

                let code_pred = if gguf_exists {
                    #[cfg(any(feature = "ggml-backend", feature = "ggml-predictor"))]
                    {
                        info!("Using GGML code predictor (optimized C backend)");
                        let ggml = code_predictor_ggml::CodePredictorGgml::load(&cp_dir)?;
                        PredictorBackend::Ggml(ggml)
                    }
                    #[cfg(not(any(feature = "ggml-backend", feature = "ggml-predictor")))]
                    {
                        info!("Using Candle code predictor (quantized GGUF)");
                        let candle = code_predictor_candle::CodePredictorCandle::load(&cp_dir)?;
                        PredictorBackend::Candle(candle)
                    }
                } else {
                    let onnx_path = cp_dir.join("code_predictor_core.onnx");
                    let onnx_data_path = cp_dir.join("code_predictor_core.onnx.data");
                    let weights_path = cp_dir.join("code_predictor_weights.npz");
                    if !onnx_path.exists() || !onnx_data_path.exists() || !weights_path.exists() {
                        anyhow::bail!(
                            "No GGUF predictor model found in {} and ONNX fallback assets are missing. \
Expected GGUF: code-predictor-q8_0.gguf / code-predictor-q4_0.gguf. \
Expected ONNX fallback: code_predictor_core.onnx + code_predictor_core.onnx.data + code_predictor_weights.npz",
                            cp_dir.display()
                        );
                    }
                    info!("Falling back to ONNX code predictor");
                    detect_ort_lib();
                    let onnx = code_predictor::CodePredictor::load(&cp_dir)?;
                    PredictorBackend::Onnx(onnx)
                };

                // Load codec_embedding and tts_pad_embed for feedback computation
                let emb_dir = models_dir.join("embeddings");
                let codec_embedding: Array2<f32> =
                    read_npy(emb_dir.join("codec_embedding.npy")).context("codec_embedding.npy")?;

                let tts_pad_path = emb_dir.join("tts_pad_embed.npy");
                let tts_pad_embed: Array1<f32> = if tts_pad_path.exists() {
                    read_npy(&tts_pad_path).context("tts_pad_embed.npy")?
                } else {
                    warn!("tts_pad_embed.npy not found, computing from TextEmbedder...");
                    let embedder = embedder::TextEmbedder::load(&emb_dir)?;
                    let embed = embedder.tts_pad_embed.clone();
                    if let Err(e) = ndarray_npy::write_npy(&tts_pad_path, &embed) {
                        warn!("Failed to save tts_pad_embed.npy: {}", e);
                    }
                    embed
                };

                WorkerState::Predictor(PredictorState {
                    code_predictor: code_pred,
                    codec_embedding,
                    feedback_buf: tts_pad_embed.clone(),
                    tts_pad_embed,
                    hidden_decode_buf: Vec::with_capacity(HIDDEN_SIZE),
                })
            }
            "vocoder" => {
                detect_ort_lib();
                let voc_path = find_vocoder_model(models_dir)?;
                let voc = vocoder::Vocoder::load(&voc_path)?;
                WorkerState::Vocoder(VocoderState { vocoder: voc })
            }
            _ => anyhow::bail!("Unknown role: {} (expected: talker, predictor)", role),
        };

        info!("{} worker initialized", role);
        Ok(Self {
            state,
            models_dir: models_dir.to_path_buf(),
        })
    }

    pub async fn handle_request(&mut self, req: Request) -> Response {
        match self.dispatch(req) {
            Ok(resp) => resp,
            Err(e) => {
                error!("Request error: {:#}", e);
                Response {
                    status: "error".into(),
                    data: ResponseData::Init,
                    error: Some(format!("{:#}", e)),
                }
            }
        }
    }

    fn dispatch(&mut self, req: Request) -> Result<Response> {
        match req {
            Request::Ping => ok(ResponseData::Ping),
            Request::Init { .. } => ok(ResponseData::Init),
            Request::TokenizeAndEmbed {
                text,
                language,
                ref_codec_tokens,
            } => self.handle_tokenize_embed(&text, &language, ref_codec_tokens.as_ref()),
            Request::TalkerPrefill { prefix_embeddings } => {
                self.handle_talker_prefill(&prefix_embeddings)
            }
            Request::TalkerStep {
                feedback_embedding,
                temperature,
                repetition_penalty,
                eos_boost,
            } => self.handle_talker_step(
                feedback_embedding.as_deref(),
                temperature,
                repetition_penalty,
                eos_boost,
            ),
            Request::CodePredict {
                hidden_state,
                code_0,
                temperature,
            } => self.handle_code_predict(&hidden_state, code_0, temperature),
            Request::Vocode { codes, n_tokens } => self.handle_vocode(&codes, n_tokens),
            Request::LoadVoice {
                codec_tokens,
                n_tokens,
                ref_text,
            } => self.handle_load_voice(&codec_tokens, n_tokens, ref_text),
        }
    }

    // ── Talker handlers ──────────────────────────────────────────────

    fn handle_tokenize_embed(
        &mut self,
        text: &str,
        language: &str,
        ref_tokens: Option<&RefCodecTokens>,
    ) -> Result<Response> {
        let ts = self.talker_state()?;

        // Tokenize target text
        let encoding = ts
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenize: {}", e))?;
        let target_ids: Vec<usize> = encoding.get_ids().iter().map(|&id| id as usize).collect();
        info!("Tokenized {} → {} tokens", text.len(), target_ids.len());

        let prefix = if let Some(ref_input) = ref_tokens {
            match ref_input {
                RefCodecTokens::Loaded => {
                    let ref_toks = ts
                        .ref_codec_tokens
                        .as_ref()
                        .context("No voice reference loaded")?;
                    // Prepend ref_text tokens for ICL if available
                    let all_ids = if let Some(ref ref_text) = ts.ref_text {
                        let ref_enc = ts
                            .tokenizer
                            .encode(ref_text.as_str(), false)
                            .map_err(|e| anyhow::anyhow!("Tokenize ref_text: {}", e))?;
                        let ref_ids: Vec<usize> =
                            ref_enc.get_ids().iter().map(|&id| id as usize).collect();
                        info!("Prepending ref_text ({} tokens) for ICL", ref_ids.len());
                        let mut combined = ref_ids;
                        combined.extend_from_slice(&target_ids);
                        combined
                    } else {
                        target_ids.clone()
                    };
                    ts.embedder
                        .build_prefix_with_ref(&all_ids, ref_toks, language)
                }
                RefCodecTokens::Inline(raw) => {
                    let ref_toks = decode_i64_array2_bytes(raw, 16)?;
                    ts.embedder
                        .build_prefix_with_ref(&target_ids, &ref_toks, language)
                }
            }
        } else {
            ts.embedder.build_prefix(&target_ids, language)
        };

        let n_prefix = prefix.nrows();
        let prefix_bytes = encode_f32_slice(prefix.as_slice().unwrap());
        let expected = estimate_tokens(text, language);

        ok(ResponseData::TokenizeAndEmbed {
            prefix_embeddings: prefix_bytes,
            n_prefix,
            expected_output_tokens: expected,
        })
    }

    fn handle_talker_prefill(&mut self, prefix_bytes: &[u8]) -> Result<Response> {
        let ts = self.talker_state()?;
        let prefix_flat = decode_f32_vec(prefix_bytes)?;
        let n_tokens = prefix_flat.len() / HIDDEN_SIZE;
        info!("Prefilling {} tokens", n_tokens);

        let hidden = ts.talker.get_hidden(&prefix_flat, n_tokens, false)?;
        ts.last_hidden = Some(hidden);

        ok(ResponseData::TalkerPrefill {
            prefilled: n_tokens,
        })
    }

    fn handle_talker_step(
        &mut self,
        feedback_bytes: Option<&[u8]>,
        temperature: f32,
        repetition_penalty: f32,
        eos_boost: f32,
    ) -> Result<Response> {
        let ts = self.talker_state()?;

        let hidden = if feedback_bytes.is_none() {
            ts.past_tokens.clear();
            ts.last_hidden
                .clone()
                .context("No hidden state from prefill")?
        } else {
            decode_f32_into(feedback_bytes.unwrap(), &mut ts.feedback_decode_buf)?;
            ts.talker.get_hidden(&ts.feedback_decode_buf, 1, true)?
        };

        let past = if ts.past_tokens.is_empty() {
            None
        } else {
            Some(ts.past_tokens.as_slice())
        };
        let code_0 =
            ts.embedder
                .sample_token(&hidden, temperature, past, repetition_penalty, eos_boost);

        let is_eos = code_0 == 2150 || code_0 >= 2048;
        let hidden_bytes = encode_f32_slice(hidden.as_slice().unwrap());

        // Track past tokens for repetition penalty
        ts.past_tokens.push(code_0);
        // Save hidden for potential reuse
        ts.last_hidden = Some(hidden);

        ok(ResponseData::TalkerStep {
            hidden_state: hidden_bytes,
            code_0,
            is_eos,
        })
    }

    fn handle_load_voice(
        &mut self,
        codec_tokens_bytes: &[u8],
        n_tokens: usize,
        ref_text: Option<String>,
    ) -> Result<Response> {
        let ts = self.talker_state()?;
        let flat = decode_i64_vec(codec_tokens_bytes)?;
        let ref_tokens = Array2::from_shape_vec((n_tokens, 16), flat)
            .context("Shape mismatch in codec_tokens")?;
        info!("Loaded voice: {} tokens", ref_tokens.nrows());
        ts.ref_codec_tokens = Some(ref_tokens);
        ts.ref_text = ref_text;
        ok(ResponseData::LoadVoice)
    }

    // ── Predictor handlers ───────────────────────────────────────────

    fn handle_code_predict(
        &mut self,
        hidden_bytes: &[u8],
        code_0: i32,
        temperature: f32,
    ) -> Result<Response> {
        let ps = self.predictor_state()?;

        decode_f32_into(hidden_bytes, &mut ps.hidden_decode_buf)?;
        let code_0_embed = ps.codec_embedding.row(code_0 as usize);
        let code_0_embed_slice = code_0_embed
            .as_slice()
            .context("codec_embedding row is not contiguous")?;

        let t0 = std::time::Instant::now();
        let predicted = ps.code_predictor.predict(
            &ps.hidden_decode_buf,
            code_0,
            code_0_embed_slice,
            temperature,
        )?;
        let pred_ms = t0.elapsed().as_millis();

        // Feedback = tts_pad + talker_codec_embed[code_0] + sum(cp_codec_embed[gi][code_i] for i=1..15)
        ps.feedback_buf.assign(&ps.tts_pad_embed);
        ps.feedback_buf += &ps.codec_embedding.row(code_0 as usize);
        for (gi, &c) in predicted.iter().enumerate() {
            ps.feedback_buf += &ps.code_predictor.codec_embeddings()[gi].row(c as usize);
        }
        debug!("code_predict: {}ms (code_0={})", pred_ms, code_0);

        let codes_json: Vec<i64> = predicted.iter().map(|&c| c as i64).collect();
        let feedback_bytes = encode_f32_slice(ps.feedback_buf.as_slice().unwrap());

        ok(ResponseData::CodePredict {
            codes: codes_json,
            feedback_embedding: feedback_bytes,
        })
    }

    fn handle_vocode(&mut self, codes_bytes: &[u8], n_tokens: usize) -> Result<Response> {
        let voc = match &mut self.state {
            WorkerState::Vocoder(vs) => &mut vs.vocoder,
            _ => anyhow::bail!("This worker does not have a vocoder"),
        };

        let codes: Vec<i64> = codes_bytes
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect();

        let audio_f32 = voc.synthesize(&codes, n_tokens)?;

        // Convert f32 → i16 PCM
        let audio_i16: Vec<i16> = audio_f32
            .iter()
            .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
            .collect();
        let audio_bytes: Vec<u8> = audio_i16.iter().flat_map(|s| s.to_le_bytes()).collect();

        ok(ResponseData::Vocode {
            audio: audio_bytes,
            n_samples: audio_i16.len(),
        })
    }

    // ── Helpers ──────────────────────────────────────────────────────

    fn talker_state(&mut self) -> Result<&mut TalkerState> {
        match &mut self.state {
            WorkerState::Talker(s) => Ok(s),
            _ => anyhow::bail!("This worker is not a talker"),
        }
    }

    fn predictor_state(&mut self) -> Result<&mut PredictorState> {
        match &mut self.state {
            WorkerState::Predictor(s) => Ok(s),
            _ => anyhow::bail!("This worker is not a predictor"),
        }
    }
}

// ── TCP server ───────────────────────────────────────────────────────

pub async fn run_worker(bind: &str, role: &str, models_dir: &str) -> Result<()> {
    info!("Initializing {} worker (models: {})...", role, models_dir);
    let worker = Worker::new(role, Path::new(models_dir))?;
    info!("Worker ready, listening on {}", bind);

    let listener = TcpListener::bind(bind).await?;

    if role == "talker" {
        // Talker keeps per-connection autoregressive state; keep single active client for correctness.
        let mut worker = worker;
        loop {
            let (stream, addr) = listener.accept().await?;
            info!("Client connected: {}", addr);
            handle_client_single(stream, addr, &mut worker).await;
        }
    } else {
        // Predictor/Vocoder are safe to share per request with a mutex.
        let shared = Arc::new(Mutex::new(worker));
        loop {
            let (stream, addr) = listener.accept().await?;
            info!("Client connected: {}", addr);
            let shared = Arc::clone(&shared);
            tokio::spawn(async move {
                handle_client_shared(stream, addr, shared).await;
            });
        }
    }
}

async fn handle_client_single(
    mut stream: tokio::net::TcpStream,
    addr: SocketAddr,
    worker: &mut Worker,
) {
    let _ = stream.set_nodelay(true);
    let mut msg_buf = Vec::with_capacity(16 * 1024);

    loop {
        match read_request(&mut stream, &mut msg_buf, addr).await {
            Ok(Some(req)) => {
                let resp = worker.handle_request(req).await;
                if let Err(e) = send_response(&mut stream, &resp).await {
                    warn!("Send error: {}", e);
                    info!("Client disconnected: {}", addr);
                    break;
                }
            }
            Ok(None) => {
                info!("Client disconnected: {}", addr);
                break;
            }
            Err(e) => {
                warn!("Client {} read failed: {}", addr, e);
                let resp = Response {
                    status: "error".into(),
                    data: ResponseData::Init,
                    error: Some(format!("Read: {}", e)),
                };
                let _ = send_response(&mut stream, &resp).await;
                break;
            }
        }
    }
}

async fn handle_client_shared(
    mut stream: tokio::net::TcpStream,
    addr: SocketAddr,
    worker: Arc<Mutex<Worker>>,
) {
    let _ = stream.set_nodelay(true);
    let mut msg_buf = Vec::with_capacity(16 * 1024);

    loop {
        match read_request(&mut stream, &mut msg_buf, addr).await {
            Ok(Some(req)) => {
                let resp = {
                    let mut locked = worker.lock().await;
                    locked.handle_request(req).await
                };
                if let Err(e) = send_response(&mut stream, &resp).await {
                    warn!("Send error: {}", e);
                    info!("Client disconnected: {}", addr);
                    break;
                }
            }
            Ok(None) => {
                info!("Client disconnected: {}", addr);
                break;
            }
            Err(e) => {
                warn!("Client {} read failed: {}", addr, e);
                let resp = Response {
                    status: "error".into(),
                    data: ResponseData::Init,
                    error: Some(format!("Read: {}", e)),
                };
                let _ = send_response(&mut stream, &resp).await;
                break;
            }
        }
    }
}

async fn read_request(
    stream: &mut tokio::net::TcpStream,
    msg_buf: &mut Vec<u8>,
    addr: SocketAddr,
) -> Result<Option<Request>> {
    let mut len_buf = [0u8; 4];
    if stream.read_exact(&mut len_buf).await.is_err() {
        return Ok(None);
    }
    let msg_len = u32::from_be_bytes(len_buf) as usize;
    if msg_len > 100_000_000 {
        anyhow::bail!("Message too large: {} bytes from {}", msg_len, addr);
    }

    msg_buf.resize(msg_len, 0);
    stream.read_exact(msg_buf).await?;

    let req: Request = rmp_serde::from_slice(msg_buf)
        .map_err(|e| anyhow::anyhow!("Deserialize request from {} failed: {}", addr, e))?;

    let method_name = match &req {
        Request::Ping => "ping",
        Request::Init { .. } => "init",
        Request::TokenizeAndEmbed { .. } => "tokenize_and_embed",
        Request::TalkerPrefill { .. } => "talker_prefill",
        Request::TalkerStep { .. } => "talker_step",
        Request::CodePredict { .. } => "code_predict",
        Request::Vocode { .. } => "vocode",
        Request::LoadVoice { .. } => "load_voice",
    };
    debug!("Handling request from {}: {}", addr, method_name);

    Ok(Some(req))
}

async fn send_response(stream: &mut tokio::net::TcpStream, resp: &Response) -> Result<()> {
    let bytes = rmp_serde::to_vec(resp)?;
    stream
        .write_all(&(bytes.len() as u32).to_be_bytes())
        .await?;
    stream.write_all(&bytes).await?;
    Ok(())
}

// ── Utility functions ────────────────────────────────────────────────

fn ok(data: ResponseData) -> Result<Response> {
    Ok(Response {
        status: "ok".into(),
        data,
        error: None,
    })
}

fn encode_f32_slice(data: &[f32]) -> Vec<u8> {
    // Safety: f32 slice has compatible alignment and layout for u8 view
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    bytes.to_vec()
}

fn decode_f32_vec(bytes: &[u8]) -> Result<Vec<f32>> {
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn decode_f32_into(bytes: &[u8], out: &mut Vec<f32>) -> Result<()> {
    if bytes.len() % 4 != 0 {
        anyhow::bail!("Invalid f32 byte length: {}", bytes.len());
    }
    let n = bytes.len() / 4;
    out.resize(n, 0.0);
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(())
}

fn decode_i64_vec(bytes: &[u8]) -> Result<Vec<i64>> {
    Ok(bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn decode_i64_array2_bytes(bytes: &[u8], cols: usize) -> Result<Array2<i64>> {
    let data: Vec<i64> = bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect();
    let rows = data.len() / cols;
    Ok(Array2::from_shape_vec((rows, cols), data)?)
}

fn estimate_tokens(text: &str, language: &str) -> usize {
    let chars = text.chars().count();
    let estimate = match language {
        "chinese" | "zh" => chars as f32 * 3.0,
        "english" | "en" => chars as f32 * 0.7,
        _ => chars as f32 * 2.0,
    };
    estimate.ceil().max(10.0) as usize
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

fn find_tokenizer(models_dir: &Path) -> Result<PathBuf> {
    let candidates = [
        models_dir.join("tokenizer.json"),
        models_dir.join("tokenizer/tokenizer.json"),
    ];
    candidates
        .iter()
        .find(|p| p.exists())
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("tokenizer.json not found in {}", models_dir.display()))
}

fn find_talker_model(models_dir: &Path) -> Result<PathBuf> {
    let candidates = [
        models_dir.join("talker.gguf"),
        models_dir.join("talker-q4_0.gguf"),
        models_dir.join("talker-q8_0.gguf"),
    ];
    candidates
        .iter()
        .find(|p| p.exists())
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("talker.gguf not found in {}", models_dir.display()))
}

fn find_vocoder_model(models_dir: &Path) -> Result<PathBuf> {
    let mut candidates = Vec::new();

    // RKNN models (only when feature enabled)
    #[cfg(feature = "rknn-vocoder")]
    {
        candidates.push(models_dir.join("vocoder.rknn"));
        candidates.push(models_dir.join("vocoder_64.rknn"));
        candidates.push(models_dir.join("vocoder_256.rknn"));
    }

    // ONNX models (default, noise-free)
    candidates.push(models_dir.join("vocoder.onnx"));
    candidates.push(models_dir.join("vocoder_64.onnx"));
    candidates.push(models_dir.join("vocoder_256.onnx"));

    candidates
        .iter()
        .find(|p| p.exists())
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("No vocoder model found in {}", models_dir.display()))
}

fn detect_ort_lib() {
    if std::env::var("ORT_DYLIB_PATH").is_ok() {
        return;
    }
    // Try common locations
    let paths = [
        "/usr/lib/libonnxruntime.so",
        "/usr/local/lib/libonnxruntime.so",
        "/usr/lib/aarch64-linux-gnu/libonnxruntime.so",
    ];

    for p in &paths {
        if Path::new(p).exists() {
            std::env::set_var("ORT_DYLIB_PATH", p);
            info!("Auto-detected ORT: {}", p);
            return;
        }
    }
}
