//! Candle-based code predictor using quantized GGUF model.
//! Replaces the C GGML FFI wrapper with pure Rust.

use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use std::path::Path;

use super::candle_qwen3::Qwen3Model;
use candle_core::quantized::gguf_file;
use candle_core::quantized::QMatMul;
use candle_core::{Device, Module, Tensor};

const NUM_GROUPS: usize = 15;

/// Candle-based code predictor using quantized GGUF model.
pub struct CodePredictorCandle {
    model: Qwen3Model,
    codec_embeddings_tensors: Vec<Tensor>, // [NUM_GROUPS] each [2048, 1024]
    lm_heads: Vec<QMatMul>,                // [NUM_GROUPS] each [1024, 2048]
    pub codec_embeddings: Vec<Array2<f32>>, // ndarray version for feedback computation
    device: Device,
}

impl CodePredictorCandle {
    pub fn load(cp_dir: &Path) -> Result<Self> {
        let device = Device::Cpu;

        // Find GGUF file
        let gguf_path = find_gguf(cp_dir)?;
        tracing::info!("Loading Candle code predictor: {}", gguf_path.display());

        let mut file = std::fs::File::open(&gguf_path)
            .with_context(|| format!("Open {}", gguf_path.display()))?;
        let ct =
            gguf_file::Content::read(&mut file).map_err(|e| anyhow::anyhow!("Read GGUF: {}", e))?;

        // Load the 5-layer code predictor transformer
        let model = Qwen3Model::from_gguf(
            &ct,
            &mut file,
            &device,
            "qwen3-tts",
            "code_pred.blk",
            "code_pred.output_norm.weight",
        )
        .map_err(|e| anyhow::anyhow!("Load code predictor model: {}", e))?;

        // Load codec embeddings and LM heads for each of the 15 groups
        let mut codec_embeddings_tensors = Vec::with_capacity(NUM_GROUPS);
        let mut lm_heads = Vec::with_capacity(NUM_GROUPS);
        let mut codec_embeddings = Vec::with_capacity(NUM_GROUPS);

        for i in 0..NUM_GROUPS {
            let emb = ct
                .tensor(
                    &mut file,
                    &format!("code_pred.codec_embd.{}.weight", i),
                    &device,
                )
                .map_err(|e| anyhow::anyhow!("Load codec_embd.{}: {}", i, e))?;
            let emb_f32 = emb
                .dequantize(&device)
                .map_err(|e| anyhow::anyhow!("Dequantize codec_embd.{}: {}", i, e))?;

            // Convert to ndarray for feedback computation
            let shape = emb_f32
                .dims2()
                .map_err(|e| anyhow::anyhow!("dims2 codec_embd.{}: {}", i, e))?;
            let data = emb_f32
                .to_vec2::<f32>()
                .map_err(|e| anyhow::anyhow!("to_vec2 codec_embd.{}: {}", i, e))?;
            let flat: Vec<f32> = data.into_iter().flatten().collect();
            codec_embeddings.push(Array2::from_shape_vec((shape.0, shape.1), flat)?);

            codec_embeddings_tensors.push(emb_f32);

            let head = ct
                .tensor(
                    &mut file,
                    &format!("code_pred.lm_head.{}.weight", i),
                    &device,
                )
                .map_err(|e| anyhow::anyhow!("Load lm_head.{}: {}", i, e))?;
            lm_heads.push(
                QMatMul::from_qtensor(head)
                    .map_err(|e| anyhow::anyhow!("QMatMul lm_head.{}: {}", i, e))?,
            );
        }

        tracing::info!("Candle code predictor ready ({} groups)", NUM_GROUPS);
        Ok(Self {
            model,
            codec_embeddings_tensors,
            lm_heads,
            codec_embeddings,
            device,
        })
    }

    /// Predict codes 1-15 from hidden state and code_0 embedding (from TALKER).
    pub fn predict(
        &mut self,
        hidden: &Array1<f32>,
        code_0_embed: &Array1<f32>,
        temperature: f32,
    ) -> Result<Vec<i32>> {
        self.model.clear_kv();

        let hidden_tensor = Tensor::from_slice(
            hidden.as_slice().unwrap(),
            (1, 1, hidden.len()),
            &self.device,
        )
        .map_err(|e| anyhow::anyhow!("hidden tensor: {}", e))?;

        // Use TALKER's codec embedding for code_0 (not code predictor's own)
        let code_0_emb = Tensor::from_slice(
            code_0_embed.as_slice().unwrap(),
            (1, 1, code_0_embed.len()),
            &self.device,
        )
        .map_err(|e| anyhow::anyhow!("code_0_embed tensor: {}", e))?;

        // Prefill: hidden + code_0_embed as 2-token sequence
        let prefill_input = Tensor::cat(&[&hidden_tensor, &code_0_emb], 1)
            .map_err(|e| anyhow::anyhow!("prefill cat: {}", e))?;

        let h = self
            .model
            .forward_embeddings(&prefill_input, 0)
            .map_err(|e| anyhow::anyhow!("prefill forward: {}", e))?;

        // Get last hidden from prefill
        let last_h = h
            .narrow(1, 1, 1)
            .map_err(|e| anyhow::anyhow!("narrow prefill: {}", e))?;

        let mut codes = Vec::with_capacity(NUM_GROUPS);
        let mut current_h = last_h;
        let mut offset = 2usize; // already consumed 2 tokens in prefill

        for gi in 0..NUM_GROUPS {
            // Project to logits
            let logits = self.lm_heads[gi]
                .forward(&current_h)
                .map_err(|e| anyhow::anyhow!("lm_head[{}]: {}", gi, e))?;
            let logits = logits
                .squeeze(0)
                .and_then(|t| t.squeeze(0))
                .map_err(|e| anyhow::anyhow!("squeeze logits: {}", e))?;

            // Sample
            let code = sample_top_k(&logits, temperature, 50)?;
            codes.push(code);

            if gi < NUM_GROUPS - 1 {
                // Embed predicted code with this group's embedding table
                let next_emb = self.codec_embeddings_tensors[gi]
                    .get(code as usize)
                    .map_err(|e| anyhow::anyhow!("codec_embd[{}][{}]: {}", gi, code, e))?
                    .unsqueeze(0)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| anyhow::anyhow!("unsqueeze step: {}", e))?;

                current_h = self
                    .model
                    .forward_embeddings(&next_emb, offset)
                    .map_err(|e| anyhow::anyhow!("step[{}] forward: {}", gi, e))?;
                offset += 1;
            }
        }

        Ok(codes)
    }
}

fn sample_top_k(logits: &Tensor, temperature: f32, top_k: usize) -> Result<i32> {
    let logits_vec = logits
        .to_vec1::<f32>()
        .map_err(|e| anyhow::anyhow!("logits to_vec1: {}", e))?;

    // Apply temperature
    let scaled: Vec<f32> = logits_vec.iter().map(|&x| x / temperature).collect();

    // Top-k filtering
    let mut indexed: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(top_k);

    // Softmax over top-k
    let max_val = indexed[0].1;
    let exps: Vec<f32> = indexed.iter().map(|(_, v)| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

    // Sample
    let r: f32 = rand::random();
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if cumulative >= r {
            return Ok(indexed[i].0 as i32);
        }
    }
    Ok(indexed.last().unwrap().0 as i32)
}

fn find_gguf(cp_dir: &Path) -> Result<std::path::PathBuf> {
    let candidates = [
        "qwen3-tts-0.6b-q8_0.gguf",
        "qwen3-tts-0.6b-q4_0.gguf",
        "qwen3-tts-0.6b-f16.gguf",
    ];
    for name in &candidates {
        let p = cp_dir.join(name);
        if p.exists() {
            return Ok(p);
        }
    }
    anyhow::bail!(
        "No GGUF model found in {}. Expected one of: {:?}",
        cp_dir.display(),
        candidates
    )
}
