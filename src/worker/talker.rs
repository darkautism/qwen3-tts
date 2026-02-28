use anyhow::{Context, Result};
use ndarray::Array1;
use std::path::Path;

use super::candle_qwen3::Qwen3Model;
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};

/// Talker LLM via Candle quantized GGUF (pure Rust, no llama.cpp)
pub struct TalkerLlamaCpp {
    model: Qwen3Model,
    n_embd: usize,
    pos: usize,
}

// Single-threaded usage within worker
unsafe impl Send for TalkerLlamaCpp {}
unsafe impl Sync for TalkerLlamaCpp {}

impl TalkerLlamaCpp {
    pub fn load(model_path: &Path, _n_ctx: i32, _n_threads: i32) -> Result<Self> {
        let device = Device::Cpu;

        let mut file = std::fs::File::open(model_path)
            .with_context(|| format!("Open {}", model_path.display()))?;
        let ct =
            gguf_file::Content::read(&mut file).map_err(|e| anyhow::anyhow!("Read GGUF: {}", e))?;

        // Detect architecture: "qwen3" (talker-only) or "qwen3-tts" (full model)
        let arch = ct
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok().map(|s| s.clone()))
            .unwrap_or_default();

        let (arch_prefix, block_prefix, norm_name) = if arch.contains("qwen3-tts") {
            ("qwen3-tts", "talker.blk", "talker.output_norm.weight")
        } else {
            ("qwen3", "blk", "output_norm.weight")
        };

        tracing::info!(
            "Loading Candle Qwen3 talker from {} (arch={})",
            model_path.display(),
            arch
        );
        let model = Qwen3Model::from_gguf(
            &ct,
            &mut file,
            &device,
            arch_prefix,
            block_prefix,
            norm_name,
        )
        .map_err(|e| anyhow::anyhow!("Load model: {}", e))?;

        let n_embd = model.n_embd();
        tracing::info!("TalkerLlamaCpp ready: n_embd={} (Candle GGUF)", n_embd);

        Ok(Self {
            model,
            n_embd,
            pos: 0,
        })
    }

    /// Feed embeddings [n_tokens, n_embd] and return last hidden state [n_embd]
    pub fn get_hidden(
        &mut self,
        embeddings: &[f32],
        n_tokens: usize,
        keep_history: bool,
    ) -> Result<Array1<f32>> {
        if !keep_history {
            self.model.clear_kv();
            self.pos = 0;
        }

        assert_eq!(embeddings.len(), n_tokens * self.n_embd);

        let input = Tensor::from_slice(embeddings, (1, n_tokens, self.n_embd), &Device::Cpu)
            .map_err(|e| anyhow::anyhow!("Create input tensor: {}", e))?;

        let hidden = self
            .model
            .forward_embeddings(&input, self.pos)
            .map_err(|e| anyhow::anyhow!("Forward: {}", e))?;

        self.pos += n_tokens;

        // Extract last token's hidden state: [1, seq, hidden] -> [hidden]
        let last = hidden
            .squeeze(0)
            .and_then(|t| {
                let l = t.dim(0)?;
                t.narrow(0, l - 1, 1)?.squeeze(0)
            })
            .map_err(|e| anyhow::anyhow!("Extract hidden: {}", e))?;

        let data = last
            .to_vec1::<f32>()
            .map_err(|e| anyhow::anyhow!("To vec: {}", e))?;
        Ok(Array1::from_vec(data))
    }

    pub fn n_embd(&self) -> usize {
        self.n_embd
    }
}
