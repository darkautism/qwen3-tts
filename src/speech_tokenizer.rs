//! Candle-based speech tokenizer encoder for Qwen3-TTS voice cloning.
//!
//! Implements the Mimi/Encodec encoder architecture to convert WAV audio into
//! codec tokens (ref_codec_tokens) directly on ARM64, eliminating the need
//! for a separate x86 machine with PyTorch.
//!
//! Architecture: SEANet Conv Encoder → 8-layer Transformer → Downsample → RVQ

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;

// ─── Configuration ───────────────────────────────────────────────────

const SAMPLE_RATE: u32 = 24000;
const FRAME_RATE: f64 = 12.5;
const DIMENSION: usize = 512;
const N_FILTERS: usize = 64;
const RATIOS: [usize; 4] = [8, 6, 5, 4];
const COMPRESS: usize = 2;
const KERNEL_SIZE: usize = 7;
const RESIDUAL_KERNEL_SIZE: usize = 3;
const LAST_KERNEL_SIZE: usize = 3;
const NUM_TRANSFORMER_LAYERS: usize = 8;
const NUM_HEADS: usize = 8;
const DIM_FEEDFORWARD: usize = 2048;
const CONTEXT: usize = 250; // sliding window
const ROPE_MAX_PERIOD: f64 = 10000.0;
const NUM_CODEBOOKS: usize = 16; // only first 16 used for TTS
const CODEBOOK_SIZE: usize = 2048;
const CODEBOOK_DIM: usize = 256;
const NUM_SEMANTIC_QUANTIZERS: usize = 1;
const NUM_ACOUSTIC_QUANTIZERS: usize = 31;

// ─── Causal Conv1d ───────────────────────────────────────────────────

/// Conv1d with causal (left) padding, no weight normalization.
struct CausalConv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    kernel_size: usize,
}

impl CausalConv1d {
    fn load(
        in_c: usize,
        out_c: usize,
        kernel_size: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let weight = vb
            .get((out_c, in_c, kernel_size), "weight")
            .context("conv weight")?;
        let bias = vb.get(out_c, "bias").ok();
        Ok(Self {
            weight,
            bias,
            stride,
            kernel_size,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Causal padding: pad left by (kernel_size - 1) * dilation
        // For stride > 1, extra padding needed: stride - (xs_len % stride) if not aligned
        let padding = self.kernel_size - self.stride;
        let xs = if padding > 0 {
            xs.pad_with_zeros(2, padding, 0)?
        } else {
            xs.clone()
        };
        let out = xs.conv1d(&self.weight, 0, self.stride, 1, 1)?;
        match &self.bias {
            Some(b) => Ok(out.broadcast_add(&b.reshape((1, (), 1))?)?),
            None => Ok(out),
        }
    }
}

// ─── ELU Activation ──────────────────────────────────────────────────

fn elu(xs: &Tensor, alpha: f64) -> Result<Tensor> {
    // ELU(x) = x if x > 0, alpha * (exp(x) - 1) if x <= 0
    // Only compute exp() on negative values to avoid overflow for large positive x
    let zeros = xs.zeros_like()?;
    let pos = xs.maximum(&zeros)?;
    let neg = xs.minimum(&zeros)?; // clamp to <= 0, safe for exp()
    let neg_part = ((neg.exp()? - 1.0)? * alpha)?;
    Ok((&pos + &neg_part)?)
}

// ─── Residual Block ──────────────────────────────────────────────────

struct ResidualBlock {
    conv1: CausalConv1d, // with dilation
    conv2: CausalConv1d, // 1x1 conv
    shortcut: Option<CausalConv1d>,
}

impl ResidualBlock {
    fn load(channels: usize, compress: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        let hidden = channels / compress;
        // block.1 = dilated conv (channels → hidden, kernel=3, dilation)
        let conv1_weight = vb
            .get(
                (hidden, channels, RESIDUAL_KERNEL_SIZE),
                "block.1.conv.weight",
            )
            .context("resblock conv1 weight")?;
        let conv1_bias = vb.get(hidden, "block.1.conv.bias").ok();
        // block.3 = 1x1 conv (hidden → channels)
        let conv2_weight = vb
            .get((channels, hidden, 1), "block.3.conv.weight")
            .context("resblock conv2 weight")?;
        let conv2_bias = vb.get(channels, "block.3.conv.bias").ok();

        let conv1 = CausalConv1d {
            weight: conv1_weight,
            bias: conv1_bias,
            stride: 1,
            kernel_size: RESIDUAL_KERNEL_SIZE * dilation, // effective kernel with dilation
        };
        let conv2 = CausalConv1d {
            weight: conv2_weight,
            bias: conv2_bias,
            stride: 1,
            kernel_size: 1,
        };

        Ok(Self {
            conv1,
            conv2,
            shortcut: None,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = match &self.shortcut {
            Some(s) => s.forward(xs)?,
            None => xs.clone(),
        };
        let h = elu(xs, 1.0)?;
        let h = self.conv1_dilated_forward(&h)?;
        let h = elu(&h, 1.0)?;
        let h = self.conv2.forward(&h)?;
        Ok((&residual + &h)?)
    }

    /// Conv1d with dilation (implemented via dilated padding)
    fn conv1_dilated_forward(&self, xs: &Tensor) -> Result<Tensor> {
        let padding = self.conv1.kernel_size - 1; // causal: full left padding
        let xs = if padding > 0 {
            xs.pad_with_zeros(2, padding, 0)?
        } else {
            xs.clone()
        };
        // Use dilation parameter in conv1d
        let dilation = self.conv1.kernel_size / RESIDUAL_KERNEL_SIZE;
        let out = xs.conv1d(&self.conv1.weight, 0, 1, dilation, 1)?;
        match &self.conv1.bias {
            Some(b) => Ok(out.broadcast_add(&b.reshape((1, (), 1))?)?),
            None => Ok(out),
        }
    }
}

// ─── SEANet Encoder ──────────────────────────────────────────────────

struct SEANetEncoder {
    initial_conv: CausalConv1d,
    blocks: Vec<(ResidualBlock, CausalConv1d)>, // (residual, downsample) pairs
    final_conv: CausalConv1d,
}

impl SEANetEncoder {
    fn load(vb: VarBuilder) -> Result<Self> {
        let initial_conv =
            CausalConv1d::load(1, N_FILTERS, KERNEL_SIZE, 1, vb.pp("layers.0.conv"))?;

        // Encoder reverses the ratios
        let encoder_ratios: Vec<usize> = RATIOS.iter().copied().rev().collect();
        let mut blocks = Vec::new();
        let mut layer_idx = 1usize;

        for (i, &ratio) in encoder_ratios.iter().enumerate() {
            let mult = 1 << i; // 1, 2, 4, 8
            let in_channels = N_FILTERS * mult;
            let out_channels = N_FILTERS * mult * 2;

            // Residual block (n_residual_layers = 1, dilation_base = 2)
            let dilation = 1; // dilation_base^0 = 1 for first residual layer
            let res = ResidualBlock::load(
                in_channels,
                COMPRESS,
                dilation,
                vb.pp(format!("layers.{layer_idx}")),
            )?;
            layer_idx += 1;
            // Skip layer for activation (ELU is applied inline)
            layer_idx += 1;

            // Downsample conv: kernel = 2 * ratio, stride = ratio
            let down = CausalConv1d::load(
                in_channels,
                out_channels,
                2 * ratio,
                ratio,
                vb.pp(format!("layers.{layer_idx}.conv")),
            )?;
            layer_idx += 1;

            blocks.push((res, down));
        }

        // Final ELU + conv
        layer_idx += 1; // skip ELU layer
        let final_conv = CausalConv1d::load(
            N_FILTERS * 16,
            DIMENSION,
            LAST_KERNEL_SIZE,
            1,
            vb.pp(format!("layers.{layer_idx}.conv")),
        )?;

        Ok(Self {
            initial_conv,
            blocks,
            final_conv,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = self.initial_conv.forward(xs)?;
        for (res, down) in &self.blocks {
            h = res.forward(&h)?;
            h = elu(&h, 1.0)?;
            h = down.forward(&h)?;
        }
        h = elu(&h, 1.0)?;
        self.final_conv.forward(&h)
    }
}

// ─── RoPE Attention ──────────────────────────────────────────────────

struct RoPEAttention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    num_heads: usize,
    head_dim: usize,
    context: usize, // sliding window size
}

impl RoPEAttention {
    fn load(dim: usize, num_heads: usize, context: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;
        let q_proj = vb.get((dim, dim), "q_proj.weight")?;
        let k_proj = vb.get((dim, dim), "k_proj.weight")?;
        let v_proj = vb.get((dim, dim), "v_proj.weight")?;
        let o_proj = vb.get((dim, dim), "o_proj.weight")?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            head_dim,
            context,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, t, _d) = xs.dims3()?;

        // Project Q, K, V
        let q = xs.broadcast_matmul(&self.q_proj.t()?)?;
        let k = xs.broadcast_matmul(&self.k_proj.t()?)?;
        let v = xs.broadcast_matmul(&self.v_proj.t()?)?;

        // Reshape to [B, H, T, head_dim]
        let q = q
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply RoPE
        let q = self.apply_rope(&q)?;
        let k = self.apply_rope(&k)?;

        // Scaled dot-product attention with causal + sliding window mask
        let scale = (self.head_dim as f64).sqrt();
        let attn = (q.matmul(&k.transpose(2, 3)?)? / scale)?;
        let attn = self.apply_causal_sliding_mask(&attn, t)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;

        // Reshape back to [B, T, D] and output projection
        let out = out.transpose(1, 2)?.reshape((b, t, ()))?;
        Ok(out.broadcast_matmul(&self.o_proj.t()?)?)
    }

    fn apply_rope(&self, xs: &Tensor) -> Result<Tensor> {
        let (_b, _h, t, hd) = xs.dims4()?;
        let half = hd / 2;
        let dev = xs.device();

        // Compute frequencies: theta_i = 1 / (max_period ^ (2i / dim))
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / ROPE_MAX_PERIOD.powf(2.0 * i as f64 / hd as f64) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, half, dev)?;

        // Positions: [0, 1, 2, ..., t-1]
        let positions: Vec<f32> = (0..t).map(|i| i as f32).collect();
        let positions = Tensor::from_vec(positions, t, dev)?;

        // [T, half] = outer product
        let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        // Split x into first half and second half
        let x1 = xs.narrow(3, 0, half)?;
        let x2 = xs.narrow(3, half, half)?;

        // Broadcast cos/sin to [1, 1, T, half]
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        // Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
        let r1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let r2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;

        Tensor::cat(&[&r1, &r2], 3).map_err(Into::into)
    }

    fn apply_causal_sliding_mask(&self, attn: &Tensor, t: usize) -> Result<Tensor> {
        let dev = attn.device();
        // Create causal mask with sliding window
        let mut mask = vec![f32::NEG_INFINITY; t * t];
        for i in 0..t {
            let start = if i >= self.context {
                i - self.context + 1
            } else {
                0
            };
            for j in start..=i {
                mask[i * t + j] = 0.0;
            }
        }
        let mask = Tensor::from_vec(mask, (1, 1, t, t), dev)?;
        Ok(attn.broadcast_add(&mask)?)
    }
}

// ─── Transformer Layer ───────────────────────────────────────────────

struct TransformerLayer {
    input_ln: LayerNormBias,
    self_attn: RoPEAttention,
    attn_layer_scale: Tensor,
    post_attn_ln: LayerNormBias,
    fc1: Tensor,
    fc2: Tensor,
    mlp_layer_scale: Tensor,
}

struct LayerNormBias {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNormBias {
    fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        let bias = vb.get(dim, "bias")?;
        Ok(Self {
            weight,
            bias,
            eps: 1e-5,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mean = xs.mean_keepdim(2)?;
        let var = xs.broadcast_sub(&mean)?.sqr()?.mean_keepdim(2)?;
        let xs_norm = xs
            .broadcast_sub(&mean)?
            .broadcast_div(&(var + self.eps)?.sqrt()?)?;
        Ok(xs_norm
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)?)
    }
}

impl TransformerLayer {
    fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        let input_ln = LayerNormBias::load(dim, vb.pp("input_layernorm"))?;
        let self_attn = RoPEAttention::load(dim, NUM_HEADS, CONTEXT, vb.pp("self_attn"))?;
        let attn_layer_scale = vb.get(dim, "self_attn_layer_scale.scale")?;
        let post_attn_ln = LayerNormBias::load(dim, vb.pp("post_attention_layernorm"))?;
        let fc1 = vb.get((DIM_FEEDFORWARD, dim), "mlp.fc1.weight")?;
        let fc2 = vb.get((dim, DIM_FEEDFORWARD), "mlp.fc2.weight")?;
        let mlp_layer_scale = vb.get(dim, "mlp_layer_scale.scale")?;
        Ok(Self {
            input_ln,
            self_attn,
            attn_layer_scale,
            post_attn_ln,
            fc1,
            fc2,
            mlp_layer_scale,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Pre-norm self-attention with layer scale
        let residual = xs;
        let h = self.input_ln.forward(xs)?;
        let h = self.self_attn.forward(&h)?;
        let h = h.broadcast_mul(&self.attn_layer_scale)?;
        let h = (residual + &h)?;

        // Pre-norm FFN with GELU and layer scale
        let residual = &h;
        let h2 = self.post_attn_ln.forward(&h)?;
        let h2 = h2.broadcast_matmul(&self.fc1.t()?)?;
        let h2 = h2.gelu()?;
        let h2 = h2.broadcast_matmul(&self.fc2.t()?)?;
        let h2 = h2.broadcast_mul(&self.mlp_layer_scale)?;
        Ok((residual + &h2)?)
    }
}

// ─── Encoder Transformer ─────────────────────────────────────────────

struct EncoderTransformer {
    layers: Vec<TransformerLayer>,
}

impl EncoderTransformer {
    fn load(vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 0..NUM_TRANSFORMER_LAYERS {
            layers.push(TransformerLayer::load(
                DIMENSION,
                vb.pp(format!("layers.{i}")),
            )?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Input is [B, D, T] from SEANet → transpose to [B, T, D]
        let mut h = xs.transpose(1, 2)?;
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }
        // Transpose back to [B, D, T]
        Ok(h.transpose(1, 2)?)
    }
}

// ─── Quantizer ───────────────────────────────────────────────────────

struct EuclideanCodebook {
    embed: Tensor, // [codebook_size, codebook_dim]
}

impl EuclideanCodebook {
    /// Load from embed_sum / cluster_usage (EMA-trained codebook format)
    fn load(vb: VarBuilder) -> Result<Self> {
        let embed_sum = vb.get((CODEBOOK_SIZE, CODEBOOK_DIM), "embed_sum")?;
        let cluster_usage = vb.get(CODEBOOK_SIZE, "cluster_usage")?;
        // Compute actual embeddings: embed = embed_sum / cluster_usage
        let usage = cluster_usage
            .unsqueeze(1)?
            .broadcast_as((CODEBOOK_SIZE, CODEBOOK_DIM))?;
        let embed = embed_sum.div(&usage)?;
        Ok(Self { embed })
    }

    /// Find nearest codebook entry for each vector
    fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        // xs: [B, codebook_dim, T] → transpose to [B*T, codebook_dim]
        let (b, _d, t) = xs.dims3()?;
        let flat = xs.transpose(1, 2)?.reshape((b * t, CODEBOOK_DIM))?;

        // Compute L2 distance: ||x - e||^2 = ||x||^2 - 2*x*e^T + ||e||^2
        let x_sq = flat.sqr()?.sum(1)?; // [B*T]
        let e_sq = self.embed.sqr()?.sum(1)?; // [codebook_size]
        let xe = flat.matmul(&self.embed.t()?)?; // [B*T, codebook_size]

        let dist = (x_sq.unsqueeze(1)?.broadcast_add(&e_sq.unsqueeze(0)?)? - (xe * 2.0)?)?;

        // Argmin → codes
        let codes = dist.argmin(1)?; // [B*T], u32
        codes.reshape((b, t)).map_err(Into::into)
    }
}

struct ResidualVectorQuantizer {
    input_proj: Tensor,   // [codebook_dim, dim, 1]
    _output_proj: Tensor, // [dim, codebook_dim, 1]
    layers: Vec<EuclideanCodebook>,
}

impl ResidualVectorQuantizer {
    fn load(n_layers: usize, vb: VarBuilder) -> Result<Self> {
        let input_proj = vb.get((CODEBOOK_DIM, DIMENSION, 1), "input_proj.weight")?;
        let output_proj = vb.get((DIMENSION, CODEBOOK_DIM, 1), "output_proj.weight")?;
        let mut layers = Vec::new();
        for i in 0..n_layers {
            layers.push(EuclideanCodebook::load(
                vb.pp(format!("layers.{i}.codebook")),
            )?);
        }
        Ok(Self {
            input_proj,
            _output_proj: output_proj,
            layers,
        })
    }

    /// Encode: returns codes [B, n_layers, T]
    fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        // Project input: [B, D, T] → [B, codebook_dim, T]
        let projected = xs.conv1d(&self.input_proj, 0, 1, 1, 1)?;

        let mut residual = projected;
        let mut all_codes = Vec::new();

        for layer in &self.layers {
            let codes = layer.encode(&residual)?; // [B, T]
            all_codes.push(codes.unsqueeze(1)?); // [B, 1, T]

            // Compute quantized value and subtract from residual
            let (b, _d, t) = residual.dims3()?;
            let flat_codes = codes.reshape((b * t,))?;
            let quantized = layer.embed.index_select(&flat_codes, 0)?; // [B*T, codebook_dim]
            let quantized = quantized.reshape((b, t, CODEBOOK_DIM))?.transpose(1, 2)?; // [B, codebook_dim, T]
            residual = (&residual - &quantized)?;
        }

        Tensor::cat(&all_codes, 1).map_err(Into::into) // [B, n_layers, T]
    }
}

struct SplitResidualVectorQuantizer {
    semantic_rvq: ResidualVectorQuantizer,
    acoustic_rvq: ResidualVectorQuantizer,
}

impl SplitResidualVectorQuantizer {
    fn load(vb: VarBuilder) -> Result<Self> {
        let semantic_rvq = ResidualVectorQuantizer::load(
            NUM_SEMANTIC_QUANTIZERS,
            vb.pp("semantic_residual_vector_quantizer"),
        )?;
        let acoustic_rvq = ResidualVectorQuantizer::load(
            NUM_ACOUSTIC_QUANTIZERS,
            vb.pp("acoustic_residual_vector_quantizer"),
        )?;
        Ok(Self {
            semantic_rvq,
            acoustic_rvq,
        })
    }

    /// Encode: returns codes [B, n_q, T] with semantic first then acoustic
    fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        let semantic_codes = self.semantic_rvq.encode(xs)?; // [B, 1, T]
        let acoustic_codes = self.acoustic_rvq.encode(xs)?; // [B, 31, T]
        Tensor::cat(&[&semantic_codes, &acoustic_codes], 1).map_err(Into::into)
    }
}

// ─── Downsample Conv ─────────────────────────────────────────────────

struct ConvDownsample1d {
    conv: CausalConv1d,
}

impl ConvDownsample1d {
    fn load(stride: usize, dim: usize, vb: VarBuilder) -> Result<Self> {
        let kernel_size = 2 * stride;
        let conv = CausalConv1d::load(dim, dim, kernel_size, stride, vb.pp("conv"))?;
        Ok(Self { conv })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.conv.forward(xs)
    }
}

// ─── Public API ──────────────────────────────────────────────────────

pub struct SpeechTokenizer {
    encoder: SEANetEncoder,
    encoder_transformer: EncoderTransformer,
    downsample: ConvDownsample1d,
    quantizer: SplitResidualVectorQuantizer,
}

impl SpeechTokenizer {
    /// Load speech tokenizer from a safetensors model file.
    pub fn load(model_path: &str) -> Result<Self> {
        tracing::info!("Loading speech tokenizer from {}", model_path);
        let dev = Device::Cpu;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &dev)? };

        let encoder = SEANetEncoder::load(vb.pp("encoder.encoder"))?;
        let encoder_transformer = EncoderTransformer::load(vb.pp("encoder.encoder_transformer"))?;

        // Downsample stride = encoder_frame_rate / frame_rate
        let product: usize = RATIOS.iter().product(); // 960
        let encoder_frame_rate = SAMPLE_RATE as f64 / product as f64; // 25 Hz
        let downsample_stride = (encoder_frame_rate / FRAME_RATE) as usize; // 2
        let downsample =
            ConvDownsample1d::load(downsample_stride, DIMENSION, vb.pp("encoder.downsample"))?;

        let quantizer = SplitResidualVectorQuantizer::load(vb.pp("encoder.quantizer"))?;

        tracing::info!("Speech tokenizer loaded successfully");
        Ok(Self {
            encoder,
            encoder_transformer,
            downsample,
            quantizer,
        })
    }

    /// Encode audio samples (24kHz, mono, f32) to codec tokens.
    ///
    /// Returns: `[n_tokens, n_codebooks]` as i64 array (typically 16 codebooks used)
    pub fn encode(&self, audio: &[f32]) -> Result<Vec<Vec<i64>>> {
        let dev = Device::Cpu;
        let len = audio.len();
        let xs = Tensor::from_vec(audio.to_vec(), (1, 1, len), &dev)?;

        tracing::info!(
            "Encoding {} samples ({:.2}s)",
            len,
            len as f64 / SAMPLE_RATE as f64
        );

        let h = self.encoder.forward(&xs)?;
        let h = self.encoder_transformer.forward(&h)?;
        let h = self.downsample.forward(&h)?;

        // Quantize → codes [B, n_q, T]
        let codes = self.quantizer.encode(&h)?;

        // Extract first 16 codebooks, transpose to [T, n_q]
        let n_q = codes.dim(1)?;
        let use_q = n_q.min(NUM_CODEBOOKS);
        let codes = codes.i((0, ..use_q, ..))?.t()?; // [T, use_q]
        let codes = codes.to_dtype(DType::I64)?;
        let (t, q) = codes.dims2()?;

        let codes_vec = codes.to_vec2::<i64>()?;
        tracing::info!("Encoded {} tokens × {} codebooks", t, q);

        Ok(codes_vec)
    }

    /// Encode a WAV file to codec tokens.
    ///
    /// Handles resampling if not 24kHz. Returns `[n_tokens, 16]` codes.
    pub fn encode_wav(&self, wav_path: &str) -> Result<Vec<Vec<i64>>> {
        let reader = hound::WavReader::open(wav_path)
            .with_context(|| format!("Failed to open WAV file: {}", wav_path))?;
        let spec = reader.spec();

        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => {
                let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .into_samples::<i32>()
                    .map(|s| s.unwrap() as f32 / max_val)
                    .collect()
            }
            hound::SampleFormat::Float => {
                reader.into_samples::<f32>().map(|s| s.unwrap()).collect()
            }
        };

        // Convert to mono if stereo
        let mono = if spec.channels > 1 {
            samples
                .chunks(spec.channels as usize)
                .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
                .collect()
        } else {
            samples
        };

        // Resample if needed
        let audio = if spec.sample_rate != SAMPLE_RATE {
            tracing::info!(
                "Resampling from {}Hz to {}Hz",
                spec.sample_rate,
                SAMPLE_RATE
            );
            resample(&mono, spec.sample_rate, SAMPLE_RATE)
        } else {
            mono
        };

        self.encode(&audio)
    }
}

/// Simple linear resampling
fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    let ratio = from_rate as f64 / to_rate as f64;
    let out_len = (samples.len() as f64 / ratio) as usize;
    (0..out_len)
        .map(|i| {
            let src_pos = i as f64 * ratio;
            let idx = src_pos as usize;
            let frac = src_pos - idx as f64;
            if idx + 1 < samples.len() {
                samples[idx] as f64 * (1.0 - frac) + samples[idx + 1] as f64 * frac
            } else {
                samples[idx.min(samples.len() - 1)] as f64
            }
        })
        .map(|v| v as f32)
        .collect()
}
