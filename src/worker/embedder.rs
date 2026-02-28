use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;
use std::path::Path;

use super::sampling;

const HIDDEN_SIZE: usize = 1024;
const TTS_PAD_TOKEN_ID: usize = 151671;
const TTS_BOS_TOKEN_ID: usize = 151672;
const TTS_EOS_TOKEN_ID: usize = 151673;
const IM_START_TOKEN_ID: usize = 151644;

const CODEC_BOS_ID: usize = 2149;
const CODEC_PAD_ID: usize = 2148;
const CODEC_NOTHINK_ID: usize = 2155;
const CODEC_THINK_BOS_ID: usize = 2156;
const CODEC_THINK_EOS_ID: usize = 2157;

/// CPU-based text/codec embedding and token sampling
pub struct TextEmbedder {
    text_embedding: Array2<f32>,      // [151936, 2048]
    pub codec_embedding: Array2<f32>, // [3072, 1024]
    codec_head: Array2<f32>,          // [3072, 1024]
    proj_fc1_w: Array2<f32>,          // [2048, 2048]
    proj_fc1_b: Array1<f32>,          // [2048]
    proj_fc2_w: Array2<f32>,          // [1024, 2048]
    proj_fc2_b: Array1<f32>,          // [1024]
    /// Code predictor per-group codec embeddings (15 matrices, each [2048, 1024])
    /// Used for ICL reference frame embedding: group 0 uses main codec_embedding,
    /// groups 1-15 use these separate tables.
    cp_codec_embeddings: Vec<Array2<f32>>,
    pub tts_pad_embed: Array1<f32>, // [1024]
    tts_bos_embed: Array1<f32>,     // [1024]
    tts_eos_embed: Array1<f32>,     // [1024]
}

impl TextEmbedder {
    pub fn load(emb_dir: &Path) -> Result<Self> {
        tracing::info!("Loading embeddings from {}...", emb_dir.display());

        let text_embedding: Array2<f32> =
            read_npy(emb_dir.join("text_embedding.npy")).context("text_embedding.npy")?;
        let codec_embedding: Array2<f32> =
            read_npy(emb_dir.join("codec_embedding.npy")).context("codec_embedding.npy")?;
        let codec_head: Array2<f32> =
            read_npy(emb_dir.join("codec_head.npy")).context("codec_head.npy")?;
        let proj_fc1_w: Array2<f32> =
            read_npy(emb_dir.join("text_projection_linear_fc1_weight.npy"))
                .context("fc1_weight")?;
        let proj_fc1_b: Array1<f32> =
            read_npy(emb_dir.join("text_projection_linear_fc1_bias.npy")).context("fc1_bias")?;
        let proj_fc2_w: Array2<f32> =
            read_npy(emb_dir.join("text_projection_linear_fc2_weight.npy"))
                .context("fc2_weight")?;
        let proj_fc2_b: Array1<f32> =
            read_npy(emb_dir.join("text_projection_linear_fc2_bias.npy")).context("fc2_bias")?;

        tracing::info!(
            "Loaded: text_emb={:?}, codec_emb={:?}, codec_head={:?}",
            text_embedding.shape(),
            codec_embedding.shape(),
            codec_head.shape()
        );

        // Load code predictor per-group codec embeddings for ICL
        let mut cp_codec_embeddings = Vec::new();
        for i in 0..15 {
            let path = emb_dir.join(format!("cp_codec_emb_{}.npy", i));
            if path.exists() {
                let emb: Array2<f32> =
                    read_npy(&path).with_context(|| format!("cp_codec_emb_{}.npy", i))?;
                cp_codec_embeddings.push(emb);
            } else {
                break;
            }
        }
        if !cp_codec_embeddings.is_empty() {
            tracing::info!(
                "Loaded {} code predictor codec embeddings for ICL",
                cp_codec_embeddings.len()
            );
        }

        // Pre-compute TTS special embeddings
        let special_ids = [TTS_PAD_TOKEN_ID, TTS_BOS_TOKEN_ID, TTS_EOS_TOKEN_ID];
        let special_embeds = Self::embed_text_static(
            &special_ids,
            &text_embedding,
            &proj_fc1_w,
            &proj_fc1_b,
            &proj_fc2_w,
            &proj_fc2_b,
        );

        Ok(Self {
            text_embedding,
            codec_embedding,
            codec_head,
            proj_fc1_w,
            proj_fc1_b,
            proj_fc2_w,
            proj_fc2_b,
            cp_codec_embeddings,
            tts_pad_embed: special_embeds.row(0).to_owned(),
            tts_bos_embed: special_embeds.row(1).to_owned(),
            tts_eos_embed: special_embeds.row(2).to_owned(),
        })
    }

    fn embed_text_static(
        token_ids: &[usize],
        text_embedding: &Array2<f32>,
        fc1_w: &Array2<f32>,
        fc1_b: &Array1<f32>,
        fc2_w: &Array2<f32>,
        fc2_b: &Array1<f32>,
    ) -> Array2<f32> {
        let n = token_ids.len();
        let dim = text_embedding.ncols();
        let mut embeds = Array2::<f32>::zeros((n, dim));
        for (i, &id) in token_ids.iter().enumerate() {
            embeds.row_mut(i).assign(&text_embedding.row(id));
        }
        // MLP: SiLU(x @ fc1_w.T + fc1_b) @ fc2_w.T + fc2_b
        let h = embeds.dot(&fc1_w.t()) + fc1_b;
        let h = &h * &h.mapv(|x| 1.0 / (1.0 + (-x).exp())); // SiLU
        (h.dot(&fc2_w.t()) + fc2_b).mapv(|x| x as f32)
    }

    /// Embed text token IDs → [N, 1024]
    pub fn embed_text(&self, token_ids: &[usize]) -> Array2<f32> {
        Self::embed_text_static(
            token_ids,
            &self.text_embedding,
            &self.proj_fc1_w,
            &self.proj_fc1_b,
            &self.proj_fc2_w,
            &self.proj_fc2_b,
        )
    }

    /// Build prefix embedding for standard TTS (no voice cloning)
    pub fn build_prefix(&self, text_token_ids: &[usize], _language: &str) -> Array2<f32> {
        // Role: <|im_start|> assistant \n
        let role_embeds = self.embed_text(&[IM_START_TOKEN_ID, 77091, 198]);

        // Codec prefix: [nothink, think_bos, think_eos]
        let codec_special = [CODEC_NOTHINK_ID, CODEC_THINK_BOS_ID, CODEC_THINK_EOS_ID];
        let mut dual_codec = Array2::<f32>::zeros((3, HIDDEN_SIZE));
        for (i, &id) in codec_special.iter().enumerate() {
            let sum = &self.tts_pad_embed + &self.codec_embedding.row(id);
            dual_codec.row_mut(i).assign(&sum);
        }

        // Transition: tts_bos + codec_pad
        let transition = &self.tts_bos_embed + &self.codec_embedding.row(CODEC_PAD_ID);
        let transition = transition.insert_axis(ndarray::Axis(0));

        // Text: text_proj(tokens) + codec_pad, then tts_eos + codec_pad
        let text_embeds = self.embed_text(text_token_ids);
        let n_text = text_token_ids.len();
        let mut dual_text = Array2::<f32>::zeros((n_text + 1, HIDDEN_SIZE));
        let codec_pad = self.codec_embedding.row(CODEC_PAD_ID);
        for i in 0..n_text {
            let sum = &text_embeds.row(i) + &codec_pad;
            dual_text.row_mut(i).assign(&sum);
        }
        let eos_plus_pad = &self.tts_eos_embed + &codec_pad;
        dual_text.row_mut(n_text).assign(&eos_plus_pad);

        // Final: tts_pad + codec_bos
        let final_tok = &self.tts_pad_embed + &self.codec_embedding.row(CODEC_BOS_ID);
        let final_tok = final_tok.insert_axis(ndarray::Axis(0));

        ndarray::concatenate![
            ndarray::Axis(0),
            role_embeds,
            dual_codec,
            transition,
            dual_text,
            final_tok
        ]
    }

    /// Build prefix with reference audio for voice cloning (ICL mode, non-streaming).
    ///
    /// Official structure (non_streaming_mode):
    ///   [role(3)] [codec_prefix(4)] [text_stream+codec_pad(T)] [codec_stream+tts_pad(1+R)]
    ///
    /// - text_stream = ref_text_tokens + target_text_tokens + eos
    /// - codec_stream = codec_bos + sum(per_group_embeddings for each ref frame)
    /// - No final tts_pad+codec_bos (unlike normal TTS)
    pub fn build_prefix_with_ref(
        &self,
        text_token_ids: &[usize],
        ref_codec_tokens: &Array2<i64>,
        _language: &str,
    ) -> Array2<f32> {
        // Phase 1: Role prefix (text-only, 3 positions)
        let role_embeds = self.embed_text(&[IM_START_TOKEN_ID, 77091, 198]);

        // Phase 2: Codec prefix (same as normal TTS, 4 positions)
        let codec_special = [CODEC_NOTHINK_ID, CODEC_THINK_BOS_ID, CODEC_THINK_EOS_ID];
        let mut dual_codec = Array2::<f32>::zeros((3, HIDDEN_SIZE));
        for (i, &id) in codec_special.iter().enumerate() {
            let sum = &self.tts_pad_embed + &self.codec_embedding.row(id);
            dual_codec.row_mut(i).assign(&sum);
        }
        let transition = &self.tts_bos_embed + &self.codec_embedding.row(CODEC_PAD_ID);
        let transition = transition.insert_axis(ndarray::Axis(0));

        // Phase 3: Text stream = (ref_text + target_text + eos) + codec_pad
        let text_embeds = self.embed_text(text_token_ids);
        let n_text = text_token_ids.len();
        let codec_pad = self.codec_embedding.row(CODEC_PAD_ID);
        let mut text_stream = Array2::<f32>::zeros((n_text + 1, HIDDEN_SIZE));
        for i in 0..n_text {
            let sum = &text_embeds.row(i) + &codec_pad;
            text_stream.row_mut(i).assign(&sum);
        }
        let eos_plus_pad = &self.tts_eos_embed + &codec_pad;
        text_stream.row_mut(n_text).assign(&eos_plus_pad);

        // Phase 4: Codec stream = (codec_bos + ref_frame_embeddings) + tts_pad
        let n_ref = ref_codec_tokens.nrows();
        let n_groups = ref_codec_tokens.ncols().min(16);
        let have_cp_embs = self.cp_codec_embeddings.len() >= 15;

        // codec_bos position
        let bos_embed = &self.codec_embedding.row(CODEC_BOS_ID) + &self.tts_pad_embed;

        // Reference frame embeddings: sum per-group embeddings
        let mut ref_embeds = Array2::<f32>::zeros((n_ref, HIDDEN_SIZE));
        for i in 0..n_ref {
            let mut sum = self.tts_pad_embed.clone();
            for j in 0..n_groups {
                let code = ref_codec_tokens[[i, j]] as usize;
                if j == 0 {
                    // Group 0: main codec_embedding
                    sum = sum + &self.codec_embedding.row(code);
                } else if have_cp_embs {
                    // Groups 1-15: code_predictor per-group embeddings
                    sum = sum + &self.cp_codec_embeddings[j - 1].row(code);
                } else {
                    // Fallback: use main codec_embedding (wrong but backward compatible)
                    sum = sum + &self.codec_embedding.row(code);
                }
            }
            ref_embeds.row_mut(i).assign(&sum);
        }

        let bos_row = bos_embed.insert_axis(ndarray::Axis(0));

        ndarray::concatenate![
            ndarray::Axis(0),
            role_embeds,
            dual_codec,
            transition,
            text_stream,
            bos_row,
            ref_embeds
        ]
    }

    /// Sample code_0 from hidden state
    pub fn sample_token(
        &self,
        hidden_state: &Array1<f32>,
        temperature: f32,
        past_tokens: Option<&[i32]>,
        repetition_penalty: f32,
        eos_boost: f32,
    ) -> i32 {
        // logits = hidden @ codec_head.T → [3072]
        let logits = self.codec_head.dot(hidden_state);
        sampling::sample_token(
            &logits,
            temperature,
            50,
            0.95,
            past_tokens,
            repetition_penalty,
            eos_boost,
        )
    }
}
