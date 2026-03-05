use anyhow::{Context, Result};
#[cfg(feature = "rknn-vocoder")]
use rknn_rs::prelude::{Rknn, RknnCoreMask, RknnTensorFormat, RknnTensorType};
use std::path::Path;

const SAMPLES_PER_TOKEN: usize = 1920;

enum VocoderBackend {
    Onnx(OnnxVocoder),
    #[cfg(feature = "rknn-vocoder")]
    Rknn(RknnVocoder),
}

/// Vocoder: converts codec tokens → audio waveform
pub struct Vocoder {
    backend: VocoderBackend,
    pub max_tokens: usize,
}

impl Vocoder {
    pub fn load(model_path: &Path) -> Result<Self> {
        let path_str = model_path.to_str().unwrap_or("");

        #[cfg(feature = "rknn-vocoder")]
        if path_str.ends_with(".rknn") {
            let max_tokens = if path_str.contains("256") { 256 } else { 64 };
            let rknn = RknnVocoder::load(model_path)?;
            tracing::info!("RKNN INT8 vocoder ready: max_tokens={}", max_tokens);
            return Ok(Self {
                backend: VocoderBackend::Rknn(rknn),
                max_tokens,
            });
        }

        #[cfg(not(feature = "rknn-vocoder"))]
        if path_str.ends_with(".rknn") {
            anyhow::bail!(
                "RKNN vocoder requires --features rknn-vocoder. Rebuild with:\n  \
                 cargo build --release --features rknn-vocoder"
            );
        }

        if !path_str.ends_with(".onnx") {
            anyhow::bail!("Unsupported vocoder format: {}", model_path.display());
        }

        let onnx = OnnxVocoder::load(model_path)?;
        let max_tokens = onnx.max_tokens;
        tracing::info!(
            "ONNX FP32 vocoder ready: max_tokens={} (noise-free)",
            max_tokens
        );
        Ok(Self {
            backend: VocoderBackend::Onnx(onnx),
            max_tokens,
        })
    }

    /// Convert codec tokens [n_tokens, 16] → audio f32 samples
    pub fn synthesize(&mut self, codes: &[i64], n_tokens: usize) -> Result<Vec<f32>> {
        let mut audio_chunks = Vec::with_capacity(n_tokens * SAMPLES_PER_TOKEN);
        let mut padded = vec![0i64; self.max_tokens * 16];
        let mut chunk_start = 0;

        while chunk_start < n_tokens {
            let chunk_end = (chunk_start + self.max_tokens).min(n_tokens);
            let chunk_len = chunk_end - chunk_start;

            // Pad to max_tokens
            padded.fill(0);
            let src = &codes[chunk_start * 16..chunk_end * 16];
            padded[..src.len()].copy_from_slice(src);

            let chunk_audio = match &mut self.backend {
                VocoderBackend::Onnx(onnx) => onnx.run(&padded, self.max_tokens)?,
                #[cfg(feature = "rknn-vocoder")]
                VocoderBackend::Rknn(rknn) => rknn.run(&padded, self.max_tokens)?,
            };
            let actual_samples = chunk_len * SAMPLES_PER_TOKEN;
            audio_chunks.extend_from_slice(&chunk_audio[..actual_samples.min(chunk_audio.len())]);
            chunk_start = chunk_end;
        }

        Ok(audio_chunks)
    }
}

// ============================================================
// RKNN (via darkautism/rknn-rs) — only with rknn-vocoder feature
// ============================================================
#[cfg(feature = "rknn-vocoder")]
struct RknnVocoder {
    rknn: Rknn,
}

#[cfg(feature = "rknn-vocoder")]
impl RknnVocoder {
    fn load(model_path: &Path) -> Result<Self> {
        let rknn = Rknn::new(model_path)
            .with_context(|| format!("Initialize RKNN model: {}", model_path.display()))?;
        if let Err(e) = rknn.set_core_mask(RknnCoreMask::Core0_1_2) {
            tracing::warn!(
                "rknn_set_core_mask(Core0_1_2) failed: {} (continuing with default)",
                e
            );
        }
        tracing::info!("RKNN vocoder initialized");
        Ok(Self { rknn })
    }

    fn run(&self, codes: &[i64], _max_tokens: usize) -> Result<Vec<f32>> {
        // Keep fmt/type aligned with previous implementation (INT64 + fmt value 0).
        self.rknn
            .input_set_slice(
                0,
                codes,
                false,
                RknnTensorType::Int64,
                RknnTensorFormat::NCHW,
            )
            .context("rknn_inputs_set failed")?;
        self.rknn.run().context("rknn_run failed")?;
        let output = self.rknn.outputs_get::<f32>().context("rknn_outputs_get failed")?;
        Ok(output.to_vec())
    }
}

// ============================================================
// ONNX FP32 Vocoder (noise-free, CPU-based) — default
// ============================================================
struct OnnxVocoder {
    session: ort::session::Session,
    max_tokens: usize,
}

impl OnnxVocoder {
    fn load(model_path: &Path) -> Result<Self> {
        let intra_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .clamp(1, 4);
        let session = ort::session::Session::builder()?
            .with_parallel_execution(false)?
            .with_inter_threads(1)?
            .with_intra_threads(intra_threads)?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::All)?
            .with_memory_pattern(true)?
            .commit_from_file(model_path)
            .with_context(|| format!("Load ONNX vocoder: {}", model_path.display()))?;

        // Extract max_tokens from input shape [1, max_tokens, 16]
        let max_tokens = session.inputs()[0]
            .dtype()
            .tensor_shape()
            .and_then(|s| s.get(1).map(|&d| d as usize))
            .unwrap_or(64);

        Ok(Self {
            session,
            max_tokens,
        })
    }

    fn run(&mut self, codes: &[i64], max_tokens: usize) -> Result<Vec<f32>> {
        let input_tensor =
            ort::value::TensorRef::from_array_view(([1usize, max_tokens, 16usize], codes))?;
        let outputs = self.session.run(ort::inputs![input_tensor])?;
        let audio = outputs[0]
            .try_extract_array::<f32>()
            .context("Extract vocoder output")?;
        Ok(audio.as_slice().unwrap_or(&[]).to_vec())
    }
}
