//! Qwen3 transformer model using Candle's quantized GGUF support.
//! Used for both the Talker (28 layers) and Code Predictor (5 layers).

use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::RmsNorm;

// ── Rotary Embedding ─────────────────────────────────────────────

#[derive(Clone)]
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub fn new(head_dim: usize, max_pos: usize, rope_theta: f64, dev: &Device) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / (rope_theta as f32).powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), dev)?;
        let t = Tensor::arange(0u32, max_pos as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_pos, 1))?;
        let freqs = t.matmul(&inv_freq.reshape((1, inv_freq.elem_count()))?)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q, k))
    }
}

// ── Attention ────────────────────────────────────────────────────

struct Attention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rotary: RotaryEmbedding,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let kv_groups = self.n_heads / self.n_kv_heads;

        let q = self
            .q_proj
            .forward(x)?
            .reshape((b, l, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, l, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, l, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Per-head QK norms (flatten heads into batch for norm)
        let q =
            self.q_norm
                .forward(&q.flatten(0, 2)?)?
                .reshape((b, self.n_heads, l, self.head_dim))?;
        let k = self.k_norm.forward(&k.flatten(0, 2)?)?.reshape((
            b,
            self.n_kv_heads,
            l,
            self.head_dim,
        ))?;

        let (q, k) = self.rotary.apply(&q, &k, offset)?;

        // KV cache
        let (k, v) = match &self.kv_cache {
            Some((kc, vc)) if offset > 0 => {
                (Tensor::cat(&[kc, &k], 2)?, Tensor::cat(&[vc, &v], 2)?)
            }
            _ => (k, v),
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // GQA expand
        let k = if kv_groups > 1 {
            repeat_kv(k, kv_groups)?
        } else {
            k
        };
        let v = if kv_groups > 1 {
            repeat_kv(v, kv_groups)?
        } else {
            v
        };

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = probs.matmul(&v.contiguous()?)?;
        let out = out
            .transpose(1, 2)?
            .reshape((b, l, self.n_heads * self.head_dim))?;
        self.o_proj.forward(&out)
    }

    fn clear_kv(&mut self) {
        self.kv_cache = None;
    }
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b, h, l, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, h, n_rep, l, d))?
        .reshape((b, h * n_rep, l, d))
}

// ── Transformer Block ────────────────────────────────────────────

struct Block {
    attn: Attention,
    mlp_gate: QMatMul,
    mlp_up: QMatMul,
    mlp_down: QMatMul,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl Block {
    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let h = self.attn.forward(&self.ln1.forward(x)?, mask, offset)?;
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x)?;
        let gate = candle_nn::ops::silu(&self.mlp_gate.forward(&h2)?)?;
        let up = self.mlp_up.forward(&h2)?;
        let h2 = self.mlp_down.forward(&(gate * up)?)?;
        x + h2
    }

    fn clear_kv(&mut self) {
        self.attn.clear_kv();
    }
}

// ── Model ────────────────────────────────────────────────────────

pub struct Qwen3Model {
    blocks: Vec<Block>,
    norm: RmsNorm,
    n_embd: usize,
    device: Device,
    dtype: DType,
}

impl Qwen3Model {
    /// Load a Qwen3 model from GGUF file with given tensor name prefix.
    /// `prefix`: "blk" for talker GGUF, "talker.blk" or "code_pred.blk" for full TTS GGUF.
    pub fn from_gguf(
        ct: &gguf_file::Content,
        reader: &mut (impl std::io::Read + std::io::Seek),
        device: &Device,
        arch_prefix: &str,  // e.g. "qwen3" for talker, "qwen3-tts" for full model
        block_prefix: &str, // e.g. "blk" for talker, "code_pred.blk" for code predictor
        norm_name: &str,    // e.g. "output_norm.weight" or "code_pred.output_norm.weight"
    ) -> Result<Self> {
        let md = |key: &str| -> Result<&gguf_file::Value> {
            ct.metadata
                .get(key)
                .ok_or_else(|| candle_core::Error::Msg(format!("missing {key}")))
        };

        let n_heads = md(&format!("{arch_prefix}.attention.head_count"))?.to_u32()? as usize;
        let n_kv_heads = md(&format!("{arch_prefix}.attention.head_count_kv"))?.to_u32()? as usize;
        let n_embd = md(&format!("{arch_prefix}.embedding_length"))?.to_u32()? as usize;
        // head_dim from metadata (key_length), NOT n_embd/n_heads (Qwen3 uses GQA with head_dim != n_embd/n_heads)
        let head_dim = md(&format!("{arch_prefix}.attention.key_length"))
            .and_then(|v| {
                v.to_u32()
                    .map(|x| x as usize)
                    .map_err(|e| candle_core::Error::Msg(format!("{e}")))
            })
            .unwrap_or(n_embd / n_heads);
        let n_layers_key = format!("{arch_prefix}.block_count");
        let code_pred_key = format!("{arch_prefix}.code_predictor.layer_count");
        let n_layers = if block_prefix.contains("code_pred") {
            md(&code_pred_key)
                .map(|v| v.to_u32().unwrap_or(5) as usize)
                .unwrap_or(5)
        } else {
            md(&n_layers_key)?.to_u32()? as usize
        };
        let rms_eps =
            md(&format!("{arch_prefix}.attention.layer_norm_rms_epsilon"))?.to_f32()? as f64;
        let rope_theta = md(&format!("{arch_prefix}.rope.freq_base"))?.to_f32()? as f64;
        let ctx_len = md(&format!("{arch_prefix}.context_length"))
            .and_then(|v| {
                v.to_u32()
                    .map(|x| x as usize)
                    .map_err(|e| candle_core::Error::Msg(format!("{e}")))
            })
            .unwrap_or(4096);

        let rotary = RotaryEmbedding::new(head_dim, ctx_len, rope_theta, device)?;
        tracing::info!(
            "Qwen3Model config: n_heads={}, n_kv_heads={}, n_embd={}, head_dim={}, n_layers={}, rms_eps={}, rope_theta={}, ctx_len={}",
            n_heads, n_kv_heads, n_embd, head_dim, n_layers, rms_eps, rope_theta, ctx_len
        );

        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("{block_prefix}.{i}");
            let q_proj =
                QMatMul::from_qtensor(ct.tensor(reader, &format!("{p}.attn_q.weight"), device)?)?;
            let k_proj =
                QMatMul::from_qtensor(ct.tensor(reader, &format!("{p}.attn_k.weight"), device)?)?;
            let v_proj =
                QMatMul::from_qtensor(ct.tensor(reader, &format!("{p}.attn_v.weight"), device)?)?;
            let o_proj = QMatMul::from_qtensor(ct.tensor(
                reader,
                &format!("{p}.attn_output.weight"),
                device,
            )?)?;
            let q_norm = RmsNorm::new(
                ct.tensor(reader, &format!("{p}.attn_q_norm.weight"), device)?
                    .dequantize(device)?,
                rms_eps,
            );
            let k_norm = RmsNorm::new(
                ct.tensor(reader, &format!("{p}.attn_k_norm.weight"), device)?
                    .dequantize(device)?,
                rms_eps,
            );
            let ln1 = RmsNorm::new(
                ct.tensor(reader, &format!("{p}.attn_norm.weight"), device)?
                    .dequantize(device)?,
                rms_eps,
            );
            let ln2 = RmsNorm::new(
                ct.tensor(reader, &format!("{p}.ffn_norm.weight"), device)?
                    .dequantize(device)?,
                rms_eps,
            );
            let mlp_gate = QMatMul::from_qtensor(ct.tensor(
                reader,
                &format!("{p}.ffn_gate.weight"),
                device,
            )?)?;
            let mlp_up =
                QMatMul::from_qtensor(ct.tensor(reader, &format!("{p}.ffn_up.weight"), device)?)?;
            let mlp_down = QMatMul::from_qtensor(ct.tensor(
                reader,
                &format!("{p}.ffn_down.weight"),
                device,
            )?)?;

            blocks.push(Block {
                attn: Attention {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    q_norm,
                    k_norm,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    rotary: rotary.clone(),
                    kv_cache: None,
                },
                mlp_gate,
                mlp_up,
                mlp_down,
                ln1,
                ln2,
            });
        }

        let norm = RmsNorm::new(
            ct.tensor(reader, norm_name, device)?.dequantize(device)?,
            rms_eps,
        );

        Ok(Self {
            blocks,
            norm,
            n_embd,
            device: device.clone(),
            dtype: DType::F32,
        })
    }

    /// Forward pass accepting raw embeddings [batch, seq, hidden]. Returns normalized hidden states.
    pub fn forward_embeddings(&mut self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, l, _) = x.dims3()?;
        let mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(l, offset)?)
        };
        let mut h = x.clone();
        for block in &mut self.blocks {
            h = block.forward(&h, mask.as_ref(), offset)?;
        }
        self.norm.forward(&h)
    }

    pub fn clear_kv(&mut self) {
        for block in &mut self.blocks {
            block.clear_kv();
        }
    }

    pub fn n_embd(&self) -> usize {
        self.n_embd
    }

    fn causal_mask(&self, tgt: usize, offset: usize) -> Result<Tensor> {
        let total = tgt + offset;
        let mask: Vec<f32> = (0..tgt)
            .flat_map(|i| {
                (0..total).map(move |j| {
                    if j <= i + offset {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();
        Tensor::from_slice(&mask, (1, 1, tgt, total), &self.device)?.to_dtype(self.dtype)
    }
}
