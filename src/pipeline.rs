use anyhow::{Context, Result};
use tracing::{debug, info, warn};

use crate::config::Config;
use crate::protocol::*;
use crate::worker_client::{decode_i16, encode_f32, encode_i64, WorkerClient};

/// Text chunking mode for long text synthesis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkMode {
    /// No splitting — synthesize entire text at once
    None,
    /// Split every 2 punctuation marks
    Punct2,
    /// Split every 4 punctuation marks
    Punct4,
}

impl Default for ChunkMode {
    fn default() -> Self {
        ChunkMode::None
    }
}

impl std::str::FromStr for ChunkMode {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "none" | "off" | "0" => Ok(ChunkMode::None),
            "2" | "punct2" => Ok(ChunkMode::Punct2),
            "4" | "punct4" => Ok(ChunkMode::Punct4),
            _ => Err(format!("Invalid chunk mode: '{}' (use: none, 2, 4)", s)),
        }
    }
}

impl std::fmt::Display for ChunkMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChunkMode::None => write!(f, "none"),
            ChunkMode::Punct2 => write!(f, "2"),
            ChunkMode::Punct4 => write!(f, "4"),
        }
    }
}

/// Split text by counting punctuation marks
fn split_by_punct(text: &str, n: usize) -> Vec<String> {
    const PUNCT: &[char] = &[
        '。', '！', '？', '；', '，', '、', '：', '.', '!', '?', ';', ',', ':', '\n',
    ];
    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut punct_count = 0;

    for ch in text.chars() {
        current.push(ch);
        if PUNCT.contains(&ch) {
            punct_count += 1;
            if punct_count >= n {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    chunks.push(trimmed);
                }
                current.clear();
                punct_count = 0;
            }
        }
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        chunks.push(trimmed);
    }

    if chunks.is_empty() {
        chunks.push(text.to_string());
    }
    chunks
}

/// Split text according to chunk mode
pub fn split_text(text: &str, mode: ChunkMode) -> Vec<String> {
    match mode {
        ChunkMode::None => vec![text.to_string()],
        ChunkMode::Punct2 => split_by_punct(text, 2),
        ChunkMode::Punct4 => split_by_punct(text, 4),
    }
}

/// Orchestrates the distributed TTS pipeline across workers
pub struct Pipeline {
    talker: WorkerClient,
    predictor: WorkerClient,
    vocoder: WorkerClient,
    config: Config,
}

/// Inline voice data (from web UI / API)
pub struct InlineVoiceData {
    pub codec_tokens: Vec<Vec<i64>>,
    pub ref_text: Option<String>,
}

/// Parameters for a single TTS request
pub struct SynthesisParams {
    pub text: String,
    pub language: String,
    pub voice: Option<String>,
    pub voice_data: Option<InlineVoiceData>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub cp_temperature: f32,
    pub repetition_penalty: f32,
    pub chunk_mode: ChunkMode,
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

    /// Run full TTS pipeline: text → audio (with optional chunking)
    pub async fn synthesize(&mut self, params: &SynthesisParams) -> Result<SynthesisResult> {
        let chunks = split_text(&params.text, params.chunk_mode);

        if chunks.len() <= 1 {
            return self.synthesize_single(params).await;
        }

        info!(
            "Chunked synthesis: {} chunks (mode={})",
            chunks.len(),
            params.chunk_mode
        );
        self.synthesize_pipelined(params, &chunks).await
    }

    /// Pipelined multi-chunk synthesis: overlap vocode(N) with generation(N+1)
    async fn synthesize_pipelined(
        &mut self,
        params: &SynthesisParams,
        chunks: &[String],
    ) -> Result<SynthesisResult> {
        let start = std::time::Instant::now();

        // Load voice once (persists in TalkerState for all chunks)
        let has_voice = if let Some(ref vd) = params.voice_data {
            let flat: Vec<i64> = vd.codec_tokens.iter().flatten().copied().collect();
            self.talker
                .call(&Request::LoadVoice {
                    codec_tokens: encode_i64(&flat),
                    n_tokens: vd.codec_tokens.len(),
                    ref_text: vd.ref_text.clone(),
                })
                .await?;
            true
        } else if let Some(ref voice) = params.voice {
            let (codec_tokens_i64, ref_text) = crate::voice_loader::load_voice_file(voice)?;
            self.talker
                .call(&Request::LoadVoice {
                    codec_tokens: encode_i64(&codec_tokens_i64),
                    n_tokens: codec_tokens_i64.len() / 16,
                    ref_text,
                })
                .await?;
            true
        } else {
            false
        };

        let mut all_audio: Vec<i16> = Vec::new();
        let mut total_tokens = 0usize;
        let mut pending_vocode = 0usize;

        for (i, chunk_text) in chunks.iter().enumerate() {
            let preview: String = chunk_text.chars().take(30).collect();
            info!("Chunk {}/{}: \"{}...\"", i + 1, chunks.len(), preview);

            // Phase 1: TokenizeAndEmbed
            let ref_codec = if has_voice {
                Some(RefCodecTokens::Loaded)
            } else {
                None
            };
            let tok_resp = self
                .talker
                .call(&Request::TokenizeAndEmbed {
                    text: chunk_text.clone(),
                    language: params.language.clone(),
                    ref_codec_tokens: ref_codec,
                })
                .await?;

            let (prefix_bytes, _n_prefix, expected_tokens) = match tok_resp.data {
                ResponseData::TokenizeAndEmbed {
                    prefix_embeddings,
                    n_prefix,
                    expected_output_tokens,
                } => (prefix_embeddings, n_prefix, expected_output_tokens),
                _ => anyhow::bail!("Unexpected response from tokenize_and_embed"),
            };

            // Voice clone voices may speak slower → need more tokens per character
            let expected_tokens = if has_voice {
                (expected_tokens as f32 * 1.5) as usize
            } else {
                expected_tokens
            };

            // Phase 2: TalkerPrefill
            self.talker
                .call(&Request::TalkerPrefill {
                    prefix_embeddings: prefix_bytes,
                })
                .await?;

            // Phase 3: Generate tokens (no streaming vocode — we vocode entire chunk at once)
            let d = &self.config.defaults;
            let eos_start = (expected_tokens as f32 * d.eos_start_ratio) as usize;
            let eos_max_step = (expected_tokens as f32 * d.eos_max_ratio) as usize;
            let eos_force = (expected_tokens as f32 * d.eos_force_ratio) as usize;
            let eos_max_boost: f32 = d.eos_max_boost;

            let mut codes: Vec<Vec<i64>> = Vec::new();
            let mut feedback_bytes = encode_f32(&[0.0f32; HIDDEN_SIZE]);
            let mut first_step = true;

            for step in 0..params.max_tokens {
                if step >= eos_force {
                    warn!("Force EOS at step {} in chunk {}", step, i + 1);
                    break;
                }

                let eos_boost = if step >= eos_start {
                    let progress = ((step - eos_start) as f32
                        / (eos_max_step - eos_start).max(1) as f32)
                        .min(1.0);
                    eos_max_boost * progress
                } else {
                    0.0
                };

                let fb = if first_step {
                    None
                } else {
                    Some(std::mem::take(&mut feedback_bytes))
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
                first_step = false;

                let (hidden_bytes, code_0, is_eos) = match talker_resp.data {
                    ResponseData::TalkerStep {
                        hidden_state,
                        code_0,
                        is_eos,
                    } => (hidden_state, code_0, is_eos),
                    _ => anyhow::bail!("Unexpected response from talker_step"),
                };

                if is_eos || code_0 == CODEC_EOS_ID || code_0 >= 2048 {
                    debug!("EOS at step {} in chunk {}", step, i + 1);
                    break;
                }

                let cp_resp = self
                    .predictor
                    .call(&Request::CodePredict {
                        hidden_state: hidden_bytes,
                        code_0,
                        temperature: params.cp_temperature,
                    })
                    .await?;

                let (codes_1_15, fb) = match cp_resp.data {
                    ResponseData::CodePredict {
                        codes: c,
                        feedback_embedding,
                    } => (c, feedback_embedding),
                    _ => anyhow::bail!("Unexpected response from code_predict"),
                };

                feedback_bytes = fb;
                let mut token = Vec::with_capacity(16);
                token.push(code_0 as i64);
                token.extend_from_slice(&codes_1_15);
                codes.push(token);
            }

            let n_chunk_tokens = codes.len();
            total_tokens += n_chunk_tokens;
            info!("Chunk {} generated {} tokens", i + 1, n_chunk_tokens);

            if n_chunk_tokens == 0 {
                continue;
            }

            // Queue this chunk to vocoder first, then drain if queue depth exceeds 1.
            let flat: Vec<i64> = codes.iter().flatten().copied().collect();
            self.vocoder
                .send_request(&Request::Vocode {
                    codes: encode_i64(&flat),
                    n_tokens: n_chunk_tokens,
                })
                .await?;
            pending_vocode += 1;

            // Keep one vocoder request in flight while next chunk is being generated.
            if pending_vocode > 1 {
                let resp = self.vocoder.read_response().await?;
                let audio_bytes = match &resp.data {
                    ResponseData::Vocode { audio, .. } => audio,
                    _ => anyhow::bail!("Unexpected vocoder response"),
                };
                all_audio.extend_from_slice(&decode_i16(audio_bytes)?);
                pending_vocode -= 1;
            }
        }

        // Collect remaining vocoder responses in order.
        while pending_vocode > 0 {
            let resp = self.vocoder.read_response().await?;
            let audio_bytes = match &resp.data {
                ResponseData::Vocode { audio, .. } => audio,
                _ => anyhow::bail!("Unexpected vocoder response"),
            };
            all_audio.extend_from_slice(&decode_i16(audio_bytes)?);
            pending_vocode -= 1;
        }

        if all_audio.is_empty() {
            anyhow::bail!("No audio generated from {} chunks", chunks.len());
        }

        let total_time = start.elapsed();
        let audio_duration = all_audio.len() as f32 / SAMPLE_RATE as f32;
        info!(
            "Done: {:.1}s audio in {:.1}s (RTF={:.2}x) [{} chunks, {} tokens]",
            audio_duration,
            total_time.as_secs_f32(),
            total_time.as_secs_f32() / audio_duration,
            chunks.len(),
            total_tokens
        );

        Ok(SynthesisResult {
            audio_samples: all_audio,
            sample_rate: SAMPLE_RATE,
            n_tokens: total_tokens,
            generation_time_ms: total_time.as_millis() as u64,
        })
    }

    /// Run full TTS pipeline for a single chunk (with streaming vocode)
    async fn synthesize_single(&mut self, params: &SynthesisParams) -> Result<SynthesisResult> {
        let start = std::time::Instant::now();

        // Phase 1: Tokenize and build prefix embeddings (on talker worker)
        info!("Tokenizing and embedding text...");
        let ref_tokens = if let Some(ref vd) = params.voice_data {
            // Inline voice data (from web UI)
            let flat: Vec<i64> = vd.codec_tokens.iter().flatten().copied().collect();
            let tokens_bytes = encode_i64(&flat);
            let n_tokens = vd.codec_tokens.len();
            self.talker
                .call(&Request::LoadVoice {
                    codec_tokens: tokens_bytes,
                    n_tokens,
                    ref_text: vd.ref_text.clone(),
                })
                .await?;
            self.talker
                .call(&Request::TokenizeAndEmbed {
                    text: params.text.clone(),
                    language: params.language.clone(),
                    ref_codec_tokens: Some(RefCodecTokens::Loaded),
                })
                .await?
        } else if let Some(ref voice) = params.voice {
            // Read voice file locally and send data to worker
            let (codec_tokens_i64, ref_text) = crate::voice_loader::load_voice_file(voice)?;
            let tokens_bytes = encode_i64(&codec_tokens_i64);
            let n_tokens = codec_tokens_i64.len() / 16;
            self.talker
                .call(&Request::LoadVoice {
                    codec_tokens: tokens_bytes,
                    n_tokens,
                    ref_text,
                })
                .await?;
            // Then tokenize+embed with __loaded__
            self.talker
                .call(&Request::TokenizeAndEmbed {
                    text: params.text.clone(),
                    language: params.language.clone(),
                    ref_codec_tokens: Some(RefCodecTokens::Loaded),
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

        let (prefix_bytes, n_prefix, expected_tokens) = match ref_tokens.data {
            ResponseData::TokenizeAndEmbed {
                prefix_embeddings,
                n_prefix,
                expected_output_tokens,
            } => (prefix_embeddings, n_prefix, expected_output_tokens),
            _ => anyhow::bail!("Unexpected response from tokenize_and_embed"),
        };

        // Voice clone voices may speak slower → need more tokens per character
        let has_voice = params.voice_data.is_some() || params.voice.is_some();
        let expected_tokens = if has_voice {
            (expected_tokens as f32 * 1.5) as usize
        } else {
            expected_tokens
        };

        info!(
            "Prefix built: {} tokens, expecting ~{} output tokens",
            n_prefix, expected_tokens
        );

        // Phase 2: Prefill talker
        info!("Prefilling talker...");
        self.talker
            .call(&Request::TalkerPrefill {
                prefix_embeddings: prefix_bytes,
            })
            .await?;

        // Phase 3: Autoregressive generation
        info!("Generating tokens...");
        let d = &self.config.defaults;
        let eos_start = (expected_tokens as f32 * d.eos_start_ratio) as usize;
        let eos_max_step = (expected_tokens as f32 * d.eos_max_ratio) as usize;
        let eos_force = (expected_tokens as f32 * d.eos_force_ratio) as usize;
        let eos_max_boost: f32 = d.eos_max_boost;

        let mut all_codes: Vec<Vec<i64>> = Vec::with_capacity(params.max_tokens);
        let mut feedback_bytes = encode_f32(&[0.0f32; HIDDEN_SIZE]);
        let mut first_step = true;
        let vocode_chunk_size = 64usize;
        let mut vocode_batches_sent = 0usize;
        let mut token_codes = Vec::with_capacity(16);

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
                None
            } else {
                Some(std::mem::take(&mut feedback_bytes))
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

            let (hidden_bytes, code_0, is_eos) = match talker_resp.data {
                ResponseData::TalkerStep {
                    hidden_state,
                    code_0,
                    is_eos,
                } => (hidden_state, code_0, is_eos),
                _ => anyhow::bail!("Unexpected response from talker_step"),
            };

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

            // CodePredictor → codes[1-15] + feedback embedding
            let t1 = std::time::Instant::now();
            let cp_resp = self
                .predictor
                .call(&Request::CodePredict {
                    hidden_state: hidden_bytes,
                    code_0,
                    temperature: params.cp_temperature,
                })
                .await?;
            let cp_ms = t1.elapsed().as_millis();

            let (codes_1_15, fb) = match cp_resp.data {
                ResponseData::CodePredict {
                    codes,
                    feedback_embedding,
                } => (codes, feedback_embedding),
                _ => anyhow::bail!("Unexpected response from code_predict"),
            };

            feedback_bytes = fb;

            // Collect full 16-group code token
            token_codes.clear();
            token_codes.push(code_0 as i64);
            token_codes.extend_from_slice(&codes_1_15);
            all_codes.push(token_codes.clone());

            // Vocoder streaming: send partial batch when we have enough tokens
            if all_codes.len() >= (vocode_batches_sent + 1) * vocode_chunk_size {
                let batch_start = vocode_batches_sent * vocode_chunk_size;
                let batch_end = (vocode_batches_sent + 1) * vocode_chunk_size;
                let batch_codes: Vec<i64> = all_codes[batch_start..batch_end]
                    .iter()
                    .flatten()
                    .copied()
                    .collect();
                let batch_n = batch_end - batch_start;
                debug!(
                    "Vocoder streaming: sending batch {} ({} tokens)",
                    vocode_batches_sent, batch_n
                );
                self.vocoder
                    .send_request(&Request::Vocode {
                        codes: encode_i64(&batch_codes),
                        n_tokens: batch_n,
                    })
                    .await?;
                vocode_batches_sent += 1;
            }

            if (i + 1) % 10 == 0 {
                let elapsed = start.elapsed().as_secs_f32();
                let rate = (i + 1) as f32 / elapsed;
                debug!(
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

        // Phase 4: Vocoder — collect streamed batches + send remainder
        info!(
            "Running vocoder ({} batches pre-sent)...",
            vocode_batches_sent
        );
        let mut all_audio_i16: Vec<i16> = Vec::new();

        // Queue remaining tail first so it can run while earlier batch responses are draining.
        let sent_tokens = vocode_batches_sent * vocode_chunk_size;
        let mut pending_batches = vocode_batches_sent;
        if sent_tokens < n_tokens {
            let remaining_codes: Vec<i64> = all_codes[sent_tokens..].iter().flatten().copied().collect();
            let remaining_n = n_tokens - sent_tokens;
            debug!("Vocoder: queue remaining {} tokens", remaining_n);
            self.vocoder
                .send_request(&Request::Vocode {
                    codes: encode_i64(&remaining_codes),
                    n_tokens: remaining_n,
                })
                .await?;
            pending_batches += 1;
        }

        // Collect responses in submission order.
        for batch_idx in 0..pending_batches {
            let voc_resp = self.vocoder.read_response().await?;
            let audio_bytes = match &voc_resp.data {
                ResponseData::Vocode { audio, .. } => audio,
                _ => anyhow::bail!("Unexpected response from vocoder batch"),
            };
            let batch_audio = decode_i16(audio_bytes)?;
            debug!(
                "Vocoder batch {} received: {} samples",
                batch_idx,
                batch_audio.len()
            );
            all_audio_i16.extend_from_slice(&batch_audio);
        }

        let audio_i16 = all_audio_i16;

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
