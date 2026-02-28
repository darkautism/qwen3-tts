use anyhow::{Context, Result};
use tracing::{debug, info, warn};

use crate::config::Config;
use crate::protocol::*;
use crate::worker_client::{decode_i16, encode_f32, encode_i64, WorkerClient};

/// Orchestrates the distributed TTS pipeline across workers
pub struct Pipeline {
    talker: WorkerClient,
    predictor: WorkerClient,
    vocoder: WorkerClient,
    config: Config,
}

/// Parameters for a single TTS request
pub struct SynthesisParams {
    pub text: String,
    pub language: String,
    pub voice: Option<String>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub cp_temperature: f32,
    pub repetition_penalty: f32,
}

/// Result of synthesis
pub struct SynthesisResult {
    pub audio_samples: Vec<i16>,
    pub sample_rate: u32,
    pub n_tokens: usize,
    pub generation_time_ms: u64,
}

impl Pipeline {
    /// Connect to workers and initialize
    pub async fn new(config: Config) -> Result<Self> {
        let talker = WorkerClient::connect(&config.workers.talker.host, config.workers.talker.port)
            .await
            .context("Failed to connect to talker worker")?;

        let predictor = WorkerClient::connect(
            &config.workers.predictor.host,
            config.workers.predictor.port,
        )
        .await
        .context("Failed to connect to predictor worker")?;

        // Vocoder: use dedicated worker if configured, otherwise fall back to predictor
        let vocoder = if let Some(ref voc) = config.workers.vocoder {
            WorkerClient::connect(&voc.host, voc.port)
                .await
                .context("Failed to connect to vocoder worker")?
        } else {
            // Legacy mode: vocoder on same endpoint as predictor
            WorkerClient::connect(
                &config.workers.predictor.host,
                config.workers.predictor.port,
            )
            .await
            .context("Failed to connect to vocoder (via predictor)")?
        };

        Ok(Self {
            talker,
            predictor,
            vocoder,
            config,
        })
    }

    /// Run full TTS pipeline: text → audio
    pub async fn synthesize(&mut self, params: &SynthesisParams) -> Result<SynthesisResult> {
        let start = std::time::Instant::now();

        // Phase 1: Tokenize and build prefix embeddings (on talker worker)
        info!("Tokenizing and embedding text...");
        let ref_tokens = if let Some(ref voice) = params.voice {
            // Read voice file locally and send data to worker
            let (codec_tokens_i64, ref_text) = crate::voice_loader::load_voice_file(voice)?;
            let tokens_b64 = encode_i64(&codec_tokens_i64);
            let n_tokens = codec_tokens_i64.len() / 16;
            self.talker
                .call(&Request::LoadVoice {
                    codec_tokens: tokens_b64,
                    n_tokens,
                    ref_text,
                })
                .await?;
            // Then tokenize+embed with __loaded__
            self.talker
                .call(&Request::TokenizeAndEmbed {
                    text: params.text.clone(),
                    language: params.language.clone(),
                    ref_codec_tokens: Some("__loaded__".into()),
                })
                .await?
        } else {
            self.talker
                .call(&Request::TokenizeAndEmbed {
                    text: params.text.clone(),
                    language: params.language.clone(),
                    ref_codec_tokens: None,
                })
                .await?
        };

        let prefix_b64 = ref_tokens.data["prefix_embeddings"]
            .as_str()
            .context("Missing prefix_embeddings in response")?;
        let n_prefix = ref_tokens.data["n_prefix"].as_u64().unwrap_or(0) as usize;
        let expected_tokens = ref_tokens.data["expected_output_tokens"]
            .as_u64()
            .unwrap_or(params.max_tokens as u64) as usize;

        info!(
            "Prefix built: {} tokens, expecting ~{} output tokens",
            n_prefix, expected_tokens
        );

        // Phase 2: Prefill talker
        info!("Prefilling talker...");
        self.talker
            .call(&Request::TalkerPrefill {
                prefix_embeddings: prefix_b64.to_string(),
            })
            .await?;

        // Phase 3: Autoregressive generation
        info!("Generating tokens...");
        let d = &self.config.defaults;
        let eos_start = (expected_tokens as f32 * d.eos_start_ratio) as usize;
        let eos_max_step = (expected_tokens as f32 * d.eos_max_ratio) as usize;
        let eos_force = (expected_tokens as f32 * d.eos_force_ratio) as usize;
        let eos_max_boost: f32 = d.eos_max_boost;

        let mut all_codes: Vec<Vec<i64>> = Vec::new();
        // First step uses empty feedback (talker uses its own last hidden)
        let mut feedback_b64 = encode_f32(&[0.0f32; HIDDEN_SIZE]);
        let mut first_step = true;

        for i in 0..params.max_tokens {
            if i >= eos_force {
                warn!("Force EOS at step {} (exceeded 2x expected)", i);
                break;
            }

            // Adaptive EOS boost
            let eos_boost = if i >= eos_start {
                let progress =
                    ((i - eos_start) as f32 / (eos_max_step - eos_start).max(1) as f32).min(1.0);
                eos_max_boost * progress
            } else {
                0.0
            };

            // Talker step → hidden_state + code_0
            let t0 = std::time::Instant::now();
            let fb = if first_step {
                "__first__".to_string()
            } else {
                std::mem::take(&mut feedback_b64)
            };
            let talker_resp = self
                .talker
                .call(&Request::TalkerStep {
                    feedback_embedding: fb,
                    temperature: params.temperature,
                    repetition_penalty: params.repetition_penalty,
                    eos_boost,
                })
                .await?;
            let talker_ms = t0.elapsed().as_millis();

            first_step = false;

            let code_0 = talker_resp.data["code_0"]
                .as_i64()
                .context("Missing code_0")? as i32;
            let is_eos = talker_resp.data["is_eos"].as_bool().unwrap_or(false);

            if i < 5 || (i + 1) % 20 == 0 {
                debug!("Step {}: code_0={} talker={}ms", i, code_0, talker_ms);
            }

            if is_eos || code_0 == CODEC_EOS_ID || code_0 >= 2048 {
                info!(
                    "EOS at step {} (token={}, boost={:.1})",
                    i, code_0, eos_boost
                );
                break;
            }

            let hidden_b64 = talker_resp.data["hidden_state"]
                .as_str()
                .context("Missing hidden_state")?
                .to_string();

            // CodePredictor → codes[1-15] + feedback embedding
            let t1 = std::time::Instant::now();
            let cp_resp = self
                .predictor
                .call(&Request::CodePredict {
                    hidden_state: hidden_b64,
                    code_0,
                    temperature: params.cp_temperature,
                })
                .await?;
            let cp_ms = t1.elapsed().as_millis();

            let codes_1_15: Vec<i64> = cp_resp.data["codes"]
                .as_array()
                .context("Missing codes array")?
                .iter()
                .map(|v| v.as_i64().unwrap_or(0))
                .collect();

            feedback_b64 = cp_resp.data["feedback_embedding"]
                .as_str()
                .context("Missing feedback_embedding")?
                .to_string();

            // Collect full 16-group code token
            let mut token_codes = vec![code_0 as i64];
            token_codes.extend_from_slice(&codes_1_15);
            all_codes.push(token_codes);

            if (i + 1) % 10 == 0 {
                let elapsed = start.elapsed().as_secs_f32();
                let rate = (i + 1) as f32 / elapsed;
                info!(
                    "[{}/{}] rate={:.1} tok/s, talker={}ms, cp={}ms",
                    i + 1,
                    params.max_tokens,
                    rate,
                    talker_ms,
                    cp_ms
                );
            }
        }

        let n_tokens = all_codes.len();
        let gen_time = start.elapsed();
        info!(
            "Generated {} tokens in {:.1}s ({:.1} tok/s)",
            n_tokens,
            gen_time.as_secs_f32(),
            n_tokens as f32 / gen_time.as_secs_f32()
        );

        if n_tokens == 0 {
            anyhow::bail!("No tokens generated");
        }

        // Phase 4: Vocoder → audio
        info!("Running vocoder...");
        let flat_codes: Vec<i64> = all_codes.iter().flatten().copied().collect();
        let voc_resp = self
            .vocoder
            .call(&Request::Vocode {
                codes: encode_i64(&flat_codes),
                n_tokens,
            })
            .await?;

        let audio_b64 = voc_resp.data["audio"]
            .as_str()
            .context("Missing audio data")?;
        let audio_i16 = decode_i16(audio_b64)?;

        let total_time = start.elapsed();
        let audio_duration = audio_i16.len() as f32 / SAMPLE_RATE as f32;
        info!(
            "Done: {:.1}s audio in {:.1}s (RTF={:.2}x)",
            audio_duration,
            total_time.as_secs_f32(),
            total_time.as_secs_f32() / audio_duration
        );

        Ok(SynthesisResult {
            audio_samples: audio_i16,
            sample_rate: SAMPLE_RATE,
            n_tokens,
            generation_time_ms: total_time.as_millis() as u64,
        })
    }
}
