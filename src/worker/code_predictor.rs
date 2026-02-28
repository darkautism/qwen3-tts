use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Array3};
use ndarray_npy::NpzReader;
use ort::session::Session;
use ort::value::Tensor;
use std::fs::File;
use std::path::Path;

use super::sampling;

const HIDDEN_SIZE: usize = 1024;
const NUM_GROUPS: usize = 15;

/// CodePredictor using ONNX Runtime for the transformer core (no KV-cache).
/// Runs full transformer for each group with growing sequence.
pub struct CodePredictor {
    session: Session,
    pub codec_embeddings: Vec<Array2<f32>>, // [2048, 1024] × 15
    lm_heads: Vec<Array2<f32>>,             // [2048, 1024] × 15
}

impl CodePredictor {
    pub fn load(cp_dir: &Path) -> Result<Self> {
        tracing::info!("Loading CodePredictor from {}...", cp_dir.display());

        // Load weights
        let weights_path = cp_dir.join("code_predictor_weights.npz");
        let mut npz = NpzReader::new(
            File::open(&weights_path)
                .with_context(|| format!("Open {}", weights_path.display()))?,
        )?;

        let mut codec_embeddings = Vec::with_capacity(NUM_GROUPS);
        let mut lm_heads = Vec::with_capacity(NUM_GROUPS);
        for i in 0..NUM_GROUPS {
            let emb: Array2<f32> = npz
                .by_name(&format!("codec_emb_{}", i))
                .with_context(|| format!("codec_emb_{}", i))?;
            let head: Array2<f32> = npz
                .by_name(&format!("lm_head_{}", i))
                .with_context(|| format!("lm_head_{}", i))?;
            codec_embeddings.push(emb);
            lm_heads.push(head);
        }

        // Load ONNX model (code_predictor_core.onnx - dynamic seq_len, no KV cache)
        let onnx_path = cp_dir.join("code_predictor_core.onnx");
        let n_threads = std::env::var("CP_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4usize);
        tracing::info!(
            "Loading ONNX model: {} (threads={})",
            onnx_path.display(),
            n_threads
        );

        let session = Session::builder()
            .context("Create ONNX session builder")?
            .with_intra_threads(n_threads)
            .context("Set intra threads")?
            .with_inter_threads(1)
            .context("Set inter threads")?
            .commit_from_file(&onnx_path)
            .with_context(|| format!("Load ONNX: {}", onnx_path.display()))?;

        tracing::info!("CodePredictor ready: {} groups", NUM_GROUPS);

        Ok(Self {
            session,
            codec_embeddings,
            lm_heads,
        })
    }

    /// Predict codes 1-15 from hidden state and code_0 embedding.
    /// Runs full transformer each step with growing sequence (no KV cache).
    pub fn predict(
        &mut self,
        past_hidden: &Array1<f32>,
        code_0_embed: &Array1<f32>,
        temperature: f32,
    ) -> Result<Vec<i32>> {
        let top_k = 50usize;

        // Build initial sequence: [hidden_state, code_0_embed]
        let mut sequence: Vec<Vec<f32>> = vec![past_hidden.to_vec(), code_0_embed.to_vec()];

        let mut predicted = Vec::with_capacity(NUM_GROUPS);

        for step in 0..NUM_GROUPS {
            let seq_len = sequence.len();

            // Build [1, seq_len, hidden_size] input
            let flat: Vec<f32> = sequence.iter().flat_map(|v| v.iter().copied()).collect();
            let hidden = Array3::from_shape_vec((1, seq_len, HIDDEN_SIZE), flat)?;
            let positions: Vec<i64> = (0..seq_len as i64).collect();
            let pos_ids = ndarray::Array2::from_shape_vec((1, seq_len), positions)?;

            // Run transformer
            let hidden_tensor = Tensor::<f32>::from_array(hidden)?;
            let pos_tensor = Tensor::<i64>::from_array(pos_ids)?;
            let outputs = self.session.run(ort::inputs![hidden_tensor, pos_tensor])?;

            let output_view = outputs[0]
                .try_extract_array::<f32>()
                .context("Extract CP output")?;

            // Take last position output → project through lm_head
            let out_shape = output_view.shape();
            let last_offset = (out_shape[1] - 1) * out_shape[2];
            let last_hidden: Vec<f32> =
                output_view.as_slice().unwrap()[last_offset..last_offset + HIDDEN_SIZE].to_vec();
            let last_h = Array1::from_vec(last_hidden.clone());

            let logits: Array1<f32> = self.lm_heads[step].dot(&last_h);
            let token = sampling::sample_simple(logits.as_slice().unwrap(), temperature, top_k);
            predicted.push(token);

            // Append the embedding for this token to the sequence (for next step)
            if step < NUM_GROUPS - 1 {
                let embed_row = self.codec_embeddings[step].row(token as usize);
                sequence.push(embed_row.to_vec());
            }
        }

        Ok(predicted)
    }
}
