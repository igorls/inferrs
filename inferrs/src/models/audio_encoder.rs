//! Gemma 4 audio encoder (Conformer / Universal Speech Model).
//!
//! Architecture (per layer):
//!   1. FeedForward1 (macaron, residual_weight=0.5)
//!   2. norm_pre_attn → chunked local self-attention → norm_post_attn + residual
//!   3. LightConv1d (GLU + depthwise conv)
//!   4. FeedForward2 (macaron, residual_weight=0.5)
//!   5. norm_out
//!
//! After N layers: output_proj (hidden_size → output_proj_dims).
//!
//! Weight prefix: `model.audio_tower.*`
//! Projection prefix: `model.embed_audio.*`
//!
//! # Attention
//! Uses chunked local attention with chunk_size=12 and context_left=13.
//! For inference we implement this as full attention with a band mask — each
//! query attends only to itself and the preceding 12 keys.  This is correct
//! (same mathematical result as the blocked implementation) and fast enough
//! given typical audio lengths after 4× subsampling (≤ ~750 frames for 30 s).

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{
    layer_norm_no_bias, linear_no_bias, rms_norm, LayerNorm, Linear, RmsNorm, VarBuilder,
};

use crate::config::AudioConfig;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Clamp a tensor element-wise to `[-clip, clip]`.
fn grad_clip(xs: &Tensor, clip: f32) -> Result<Tensor> {
    Ok(xs.clamp(-clip as f64, clip as f64)?)
}

// ---------------------------------------------------------------------------
// ClippableLinear — weights stored under `.linear.weight`.
// ---------------------------------------------------------------------------

/// A linear layer whose weights are stored under `.linear.weight`.
///
/// Applies QAT clipping bounds (input_min/max, output_min/max) from the checkpoint.
/// The model was trained with quantization-aware training and stores per-layer
/// saturation bounds that must be applied at inference time to match the reference.
struct ClipLinear {
    inner: Linear,
    /// Clamp input to [input_min, input_max] before the linear (if stored).
    input_min: Option<f32>,
    input_max: Option<f32>,
    /// Clamp output to [output_min, output_max] after the linear (if stored).
    output_min: Option<f32>,
    output_max: Option<f32>,
}

impl ClipLinear {
    fn load(vb: VarBuilder, in_dim: usize, out_dim: usize) -> Result<Self> {
        let inner = linear_no_bias(in_dim, out_dim, vb.pp("linear"))?;
        // Load optional QAT clipping bounds.  These are 0-dim scalar tensors (shape []).
        // If absent (e.g. non-QAT checkpoint) the clamping step is skipped.
        let load_scalar = |name: &str| -> Option<f32> {
            // Candle represents a 0-dim tensor as shape `()`.
            vb.get((), name).ok().and_then(|t| {
                t.to_dtype(candle_core::DType::F32)
                    .ok()?
                    .to_scalar::<f32>()
                    .ok()
            })
        };
        Ok(Self {
            inner,
            input_min: load_scalar("input_min"),
            input_max: load_scalar("input_max"),
            output_min: load_scalar("output_min"),
            output_max: load_scalar("output_max"),
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Pre-linear input clamp
        let xs = match (self.input_min, self.input_max) {
            (Some(lo), Some(hi)) => xs.clamp(lo as f64, hi as f64)?,
            _ => xs.clone(),
        };
        let xs = self.inner.forward(&xs)?;
        // Post-linear output clamp
        let xs = match (self.output_min, self.output_max) {
            (Some(lo), Some(hi)) => xs.clamp(lo as f64, hi as f64)?,
            _ => xs,
        };
        Ok(xs)
    }
}

// ---------------------------------------------------------------------------
// SubSampleConvProjection
// ---------------------------------------------------------------------------

struct ConvProjLayer {
    conv: candle_nn::Conv2d,
    norm: LayerNorm,
    out_channels: usize,
}

impl ConvProjLayer {
    fn load(vb: VarBuilder, in_ch: usize, out_ch: usize, eps: f64) -> Result<Self> {
        let conv = candle_nn::conv2d_no_bias(
            in_ch,
            out_ch,
            3,
            candle_nn::Conv2dConfig {
                stride: 2,
                padding: 1,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
            vb.pp("conv"),
        )?;
        let norm = layer_norm_no_bias(out_ch, eps, vb.pp("norm"))?;
        Ok(Self {
            conv,
            norm,
            out_channels: out_ch,
        })
    }

    /// Input: `[batch, in_ch, T, freq]` → Output: `[batch, out_ch, T/2, freq/2]`
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.conv.forward(xs)?;
        // Move channels last for LayerNorm: [batch, T/2, freq/2, out_ch]
        let xs = xs.permute((0, 2, 3, 1))?;
        let xs = self.norm.forward(&xs)?;
        let xs = xs.relu()?;
        // Restore channel-first: [batch, out_ch, T/2, freq/2]
        Ok(xs.permute((0, 3, 1, 2))?)
    }
}

struct SubSampleConvProjection {
    layer0: ConvProjLayer,
    layer1: ConvProjLayer,
    input_proj: Linear,
}

impl SubSampleConvProjection {
    fn load(vb: VarBuilder, cfg: &AudioConfig) -> Result<Self> {
        let layer0 = ConvProjLayer::load(
            vb.pp("layer0"),
            1,
            cfg.subsampling_conv_channels[0],
            cfg.rms_norm_eps,
        )?;
        let layer1 = ConvProjLayer::load(
            vb.pp("layer1"),
            cfg.subsampling_conv_channels[0],
            cfg.subsampling_conv_channels[1],
            cfg.rms_norm_eps,
        )?;
        // proj_input_dim = (N_MEL / 4) * ch1  (128 mel bins halved twice by stride-2 convs)
        let proj_input_dim = (crate::audio::N_MEL / 4) * cfg.subsampling_conv_channels[1];
        let input_proj =
            linear_no_bias(proj_input_dim, cfg.hidden_size, vb.pp("input_proj_linear"))?;
        Ok(Self {
            layer0,
            layer1,
            input_proj,
        })
    }

    /// Input: `[1, T, 128]` log-mel → Output: `[1, T/4, hidden_size]`
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Add channel dim: [1, 1, T, 128]
        let xs = xs.unsqueeze(1)?;
        let xs = self.layer0.forward(&xs)?;
        let xs = self.layer1.forward(&xs)?;

        let (batch, _ch, t4, freq4) = xs.dims4()?;
        // [batch, ch=32, T/4, 32] → [batch, T/4, 32, 32]
        let xs = xs.permute((0, 2, 3, 1))?;
        // → [batch, T/4, 32*32=1024]
        let xs = xs.reshape((batch, t4, freq4 * self.layer1.out_channels))?;
        // Project: [batch, T/4, hidden_size]
        Ok(self.input_proj.forward(&xs)?)
    }
}

// ---------------------------------------------------------------------------
// Relative positional encoding
// ---------------------------------------------------------------------------

/// Relative position embeddings matching the reference Gemma4 audio encoder.
///
/// Shape: `[context_left, hidden_size]` = `[13, 1024]`.
///
/// Reference: `position_ids = torch.arange(12, -1, -1)` → 13 decreasing values [12..0].
/// Standard sinusoidal encoding: `inv_timescales[j] = 1 / 10000^(2j/d)`.
/// `pos_embed[n] = [sin(pos[n] * inv_ts), cos(pos[n] * inv_ts)]`.
///
/// Indexing: for full-sequence attention (query i, key j):
///   embed_idx = max_pos - clamp(i - j, 0, max_pos)
///   = max_pos for j == i  (current position, relative pos = 0)
///   = 0 for i - j >= 12  (furthest past, clamped)
struct RelPositionalEncoding {
    embeddings: Tensor,
}

impl RelPositionalEncoding {
    fn new(cfg: &AudioConfig, device: &Device, dtype: DType) -> Result<Self> {
        // 13 positions: [12, 11, ..., 0] (context_left = 13)
        let n_pos = cfg.attention_context_left; // 13
        let h = cfg.hidden_size;
        let half = h / 2;

        // inv_timescales[j] = 1 / 10000^(2j/h)  (standard sinusoidal PE)
        let mut data = vec![0.0f32; n_pos * h];
        for n in 0..n_pos {
            let pos = (n_pos - 1 - n) as f32; // 12, 11, ..., 0
            for j in 0..half {
                let inv_ts = f32::exp(-2.0 * j as f32 * (10000f32).ln() / h as f32);
                let angle = pos * inv_ts;
                data[n * h + j] = angle.sin();
                data[n * h + half + j] = angle.cos();
            }
        }
        // Shape: [13, hidden_size]
        let embeddings = Tensor::from_vec(data, (n_pos, h), device)?.to_dtype(dtype)?;
        Ok(Self { embeddings })
    }

    fn get(&self) -> &Tensor {
        &self.embeddings
    }
}

// ---------------------------------------------------------------------------
// Audio attention (band-masked full attention with relative position bias)
// ---------------------------------------------------------------------------

struct AudioAttention {
    q_proj: ClipLinear,
    k_proj: ClipLinear,
    v_proj: ClipLinear,
    post: ClipLinear,
    relative_k_proj: Tensor, // [num_heads * head_dim, hidden_size]
    per_dim_scale: Tensor,   // [head_dim]
    num_heads: usize,
    head_dim: usize,
    q_scale: f32,
    k_scale: f32,
    logit_cap: f32,
    invalid_logit: f32,
    #[allow(dead_code)]
    chunk_size: usize, // attention_chunk_size (12) — block boundary step
    context_left: usize, // attention_context_left (13) — lookback includes current
}

impl AudioAttention {
    fn load(vb: VarBuilder, cfg: &AudioConfig) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let q_scale = ((head_dim as f64).powf(-0.5) / 2.0_f64.ln()) as f32;
        let k_scale = ((1.0 + std::f64::consts::E).ln() / 2.0_f64.ln()) as f32;
        let attn = vb.pp("self_attn");
        Ok(Self {
            q_proj: ClipLinear::load(attn.pp("q_proj"), cfg.hidden_size, cfg.hidden_size)?,
            k_proj: ClipLinear::load(attn.pp("k_proj"), cfg.hidden_size, cfg.hidden_size)?,
            v_proj: ClipLinear::load(attn.pp("v_proj"), cfg.hidden_size, cfg.hidden_size)?,
            post: ClipLinear::load(attn.pp("post"), cfg.hidden_size, cfg.hidden_size)?,
            relative_k_proj: attn.get(
                (cfg.num_attention_heads * head_dim, cfg.hidden_size),
                "relative_k_proj.weight",
            )?,
            per_dim_scale: attn.get((head_dim,), "per_dim_scale")?,
            num_heads: cfg.num_attention_heads,
            head_dim,
            q_scale,
            k_scale,
            logit_cap: cfg.attention_logit_cap as f32,
            invalid_logit: cfg.attention_invalid_logits_value as f32,
            chunk_size: cfg.attention_chunk_size,
            context_left: cfg.attention_context_left,
        })
    }

    fn forward(&self, xs: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        let (batch, t, _) = xs.dims3()?;
        let h = self.num_heads;
        let d = self.head_dim;

        // Projections and cast to f32 for attention (matches PyTorch .float()).
        let q = self
            .q_proj
            .forward(xs)?
            .reshape((batch, t, h, d))?
            .to_dtype(DType::F32)?;
        let k = self
            .k_proj
            .forward(xs)?
            .reshape((batch, t, h, d))?
            .to_dtype(DType::F32)?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((batch, t, h, d))?
            .to_dtype(DType::F32)?;

        // Scale Q by q_scale * softplus(per_dim_scale)
        let pds = self.per_dim_scale.to_dtype(DType::F32)?;
        // softplus(x) = log(1 + exp(x))
        let pds_sp = (pds.exp()? + 1.0_f64)?.log()?;
        let q = (q.broadcast_mul(&pds_sp)? * self.q_scale as f64)?;

        // Scale K
        let k = (k * self.k_scale as f64)?;

        // [batch, H, T, D] — make contiguous after permute for Metal
        let q = q.permute((0, 2, 1, 3))?.contiguous()?;
        let k = k.permute((0, 2, 1, 3))?.contiguous()?;
        let v = v.permute((0, 2, 1, 3))?.contiguous()?;

        // Content-content scores: [batch, H, T, T]
        let matrix_ac = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;

        // Relative position bias (matrix_bd).
        //
        // pos_emb: [13, hidden_size]  (context_left = 13, positions [12..0] decreasing)
        // relative_k_proj: [H*D, hidden_size] — standard nn.Linear weight layout
        //
        // Step 1: rel_k = pos_emb @ relative_k_proj.T  →  [13, H*D]
        let rel_kp = self.relative_k_proj.to_dtype(DType::F32)?; // [H*D, hidden_size]
        let pos_emb_f32 = pos_emb.to_dtype(DType::F32)?; // [13, hidden_size]
        let rel_k_flat = pos_emb_f32.matmul(&rel_kp.t()?.contiguous()?)?; // [13, H*D]

        // Step 2: reshape to [13, H, D], then permute to [H, D, 13]
        let n_pos = rel_k_flat.dim(0)?; // 13
        let rel_k = rel_k_flat.reshape((n_pos, h, d))?; // [13, H, D]
        let rel_k = rel_k.permute((1, 2, 0))?.contiguous()?; // [H, D, 13]

        // Step 3: qr = q @ rel_k  →  [batch, H, T, 13]
        let qr = q.matmul(&rel_k.unsqueeze(0)?.broadcast_as((batch, h, d, n_pos))?)?;

        // Step 4: gather indices.
        // For (query i, key j): embed_idx = max_pos - clamp(i - j, 0, max_pos)
        //   where max_pos = 12 (index of embedding for relative position 0).
        // Equivalently: pos_emb[0] = embedding for distance 12 (furthest past),
        //               pos_emb[12] = embedding for distance 0 (current position).
        // Distances beyond 12 are clamped (all share the pos_emb[0] embedding).
        let max_pos = (n_pos - 1) as i64; // 12
        let mut idx_data = vec![0u32; t * t];
        for i in 0..t {
            for j in 0..t {
                let dist = (i as i64 - j as i64).clamp(0, max_pos);
                idx_data[i * t + j] = (max_pos - dist) as u32;
            }
        }
        let idx = Tensor::from_vec(idx_data, (t, t), xs.device())?; // [T, T]

        // Step 5: broadcast idx to [B, H, T, T] and gather from qr [B, H, T, 13]
        let idx = idx
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((batch, h, t, t))?
            .contiguous()?;
        let matrix_bd = qr.contiguous()?.gather(&idx, 3)?; // [B, H, T, T]

        // Step 6: combine content-content and position bias
        let mut attn = (matrix_ac + matrix_bd)?;

        // Softcap: tanh(attn / cap) * cap
        attn = ((attn / self.logit_cap as f64)?.tanh()? * self.logit_cap as f64)?;

        // Causal band mask: apply invalid_logit where mask=0
        let mask = self.build_band_mask(t, xs.device())?; // u8: [1, 1, T, T]
        let mask_f32 = mask.to_dtype(DType::F32)?.broadcast_as(attn.shape())?;
        let fill =
            Tensor::full(self.invalid_logit, attn.shape(), attn.device())?.to_dtype(DType::F32)?;
        // attended positions: attn, masked positions: invalid_logit
        attn = ((&attn * &mask_f32)? + (&fill * (1.0_f64 - &mask_f32)?)?)?;

        // Softmax + weighted sum
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn
            .contiguous()?
            .matmul(&v)?
            .permute((0, 2, 1, 3))?
            .contiguous()?
            .reshape((batch, t, h * d))?;

        // Cast back and output projection
        let out = out.to_dtype(xs.dtype())?;
        self.post.forward(&out)
    }

    /// Causal sliding-window mask: u8 `[1, 1, T, T]`, `1` where query i may attend to key j.
    ///
    /// Each query attends to itself and the `max_past` = `context_left - 2` = 11 preceding
    /// keys: j ∈ [i-11, i], giving a window of 12 positions.
    /// This is mathematically equivalent to the reference blocked chunked attention
    /// (chunk_size=12, context_left=13) when implemented as full T×T attention —
    /// both produce the same attended key sets for every query position.
    fn build_band_mask(&self, t: usize, device: &Device) -> Result<Tensor> {
        // Causal sliding window: query i attends to keys j in [i - max_past, i].
        // Reference: sliding_window_mask_function((chunk_size=12, 0)) → 12 positions.
        // context_left=13 → max_past = context_left - 2 = 11 → window [i-11, i] = 12 keys.
        // (Query 74 → [63..74], query 12 → [1..12], query 0 → [0..0])
        let max_past = self.context_left - 2; // 11
        let mut data = vec![0u8; t * t];
        for i in 0..t {
            let lo = i.saturating_sub(max_past);
            for j in lo..=i {
                data[i * t + j] = 1;
            }
        }
        Ok(Tensor::from_vec(data, (1usize, 1usize, t, t), device)?)
    }
}

// ---------------------------------------------------------------------------
// FeedForward
// ---------------------------------------------------------------------------

struct AudioFeedForward {
    layer1: ClipLinear,
    layer2: ClipLinear,
    pre_norm: RmsNorm,
    post_norm: RmsNorm,
    grad_clip: f32,
    residual_weight: f32,
}

impl AudioFeedForward {
    fn load(vb: VarBuilder, cfg: &AudioConfig) -> Result<Self> {
        Ok(Self {
            layer1: ClipLinear::load(vb.pp("ffw_layer_1"), cfg.hidden_size, cfg.hidden_size * 4)?,
            layer2: ClipLinear::load(vb.pp("ffw_layer_2"), cfg.hidden_size * 4, cfg.hidden_size)?,
            pre_norm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("pre_layer_norm"))?,
            post_norm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_layer_norm"))?,
            grad_clip: cfg.gradient_clipping as f32,
            residual_weight: cfg.residual_weight as f32,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = grad_clip(xs, self.grad_clip)?;
        let xs = self.pre_norm.forward(&xs)?;
        let xs = candle_nn::ops::silu(&self.layer1.forward(&xs)?)?;
        let xs = self.layer2.forward(&xs)?;
        let xs = grad_clip(&xs, self.grad_clip)?;
        let xs = self.post_norm.forward(&xs)?;
        let scaled = (xs * self.residual_weight as f64)?;
        (scaled + residual).context("FFN residual add")
    }
}

// ---------------------------------------------------------------------------
// LightConv1d
// ---------------------------------------------------------------------------

struct LightConv1d {
    linear_start: ClipLinear,
    linear_end: ClipLinear,
    conv_weight: Tensor, // [hidden, 1, kernel_size]
    pre_norm: RmsNorm,
    conv_norm: RmsNorm,
    kernel_size: usize,
    grad_clip: f32,
}

impl LightConv1d {
    fn load(vb: VarBuilder, cfg: &AudioConfig) -> Result<Self> {
        let h = cfg.hidden_size;
        let k = cfg.conv_kernel_size;
        Ok(Self {
            linear_start: ClipLinear::load(vb.pp("linear_start"), h, h * 2)?,
            linear_end: ClipLinear::load(vb.pp("linear_end"), h, h)?,
            conv_weight: vb.get((h, 1, k), "depthwise_conv1d.weight")?,
            pre_norm: rms_norm(h, cfg.rms_norm_eps, vb.pp("pre_layer_norm"))?,
            conv_norm: rms_norm(h, cfg.rms_norm_eps, vb.pp("conv_norm"))?,
            kernel_size: k,
            grad_clip: cfg.gradient_clipping as f32,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs.clone();
        let (batch, _t, h) = xs.dims3()?;

        let xs = self.pre_norm.forward(xs)?;
        // GLU: linear_start → [batch, T, 2H] → split → gate
        let xs2h = self.linear_start.forward(&xs)?; // [batch, T, 2H]
        let xa = xs2h.narrow(D::Minus1, 0, h)?;
        let xb = xs2h.narrow(D::Minus1, h, h)?;
        let xs = (xa * candle_nn::ops::sigmoid(&xb)?)?; // [batch, T, H]

        // Causal depthwise conv1d: left-pad by (kernel_size-1).
        let left_pad = self.kernel_size - 1;
        let pad = Tensor::zeros((batch, left_pad, h), xs.dtype(), xs.device())?;
        let xs_padded = Tensor::cat(&[&pad, &xs], 1)?; // [batch, T+pad, H]

        // Conv1d expects [batch, C, T]: transpose to [batch, H, T+pad]
        // Metal conv1d only supports F32; cast to F32, conv, cast back.
        let orig_dtype = xs.dtype();
        let xs_t = xs_padded.to_dtype(DType::F32)?.permute((0, 2, 1))?;
        let cw = self.conv_weight.to_dtype(DType::F32)?;
        // Depthwise: groups = H, so each channel has its own filter of size k.
        let xs_conv = xs_t.conv1d(&cw, 0, 1, 1, h)?.to_dtype(orig_dtype)?;
        let xs = xs_conv.permute((0, 2, 1))?; // [batch, T, H]

        let xs = grad_clip(&xs, self.grad_clip)?;
        let xs = self.conv_norm.forward(&xs)?;
        let xs = candle_nn::ops::silu(&xs)?;
        let xs = self.linear_end.forward(&xs)?;
        Ok((xs + residual)?)
    }
}

// ---------------------------------------------------------------------------
// Full Conformer layer
// ---------------------------------------------------------------------------

struct AudioLayer {
    ff1: AudioFeedForward,
    ff2: AudioFeedForward,
    attn: AudioAttention,
    lconv: LightConv1d,
    norm_pre_attn: RmsNorm,
    norm_post_attn: RmsNorm,
    norm_out: RmsNorm,
    grad_clip: f32,
}

impl AudioLayer {
    fn load(vb: VarBuilder, cfg: &AudioConfig) -> Result<Self> {
        Ok(Self {
            ff1: AudioFeedForward::load(vb.pp("feed_forward1"), cfg)?,
            ff2: AudioFeedForward::load(vb.pp("feed_forward2"), cfg)?,
            attn: AudioAttention::load(vb.clone(), cfg)?,
            lconv: LightConv1d::load(vb.pp("lconv1d"), cfg)?,
            norm_pre_attn: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm_pre_attn"))?,
            norm_post_attn: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm_post_attn"))?,
            norm_out: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm_out"))?,
            grad_clip: cfg.gradient_clipping as f32,
        })
    }

    fn forward(&self, xs: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        // Macaron FFN1
        let xs = self.ff1.forward(xs)?;

        // Local attention
        let residual = xs.clone();
        let xs_norm = self
            .norm_pre_attn
            .forward(&grad_clip(&xs, self.grad_clip)?)?;
        let attn_out = self.attn.forward(&xs_norm, pos_emb)?;
        let xs = (self
            .norm_post_attn
            .forward(&grad_clip(&attn_out, self.grad_clip)?)?
            + residual)?;

        // LightConv1d
        let xs = self.lconv.forward(&xs)?;

        // Macaron FFN2
        let xs = self.ff2.forward(&xs)?;

        // Final norm
        Ok(self.norm_out.forward(&grad_clip(&xs, self.grad_clip)?)?)
    }
}

// ---------------------------------------------------------------------------
// Public: AudioEncoder
// ---------------------------------------------------------------------------

/// Full Gemma 4 audio encoder.
///
/// Input: `[1, T, 128]` log-mel spectrogram on the model device.
/// Output: `[T/4, lm_hidden_size]` embeddings ready for LM injection.
pub struct AudioEncoder {
    subsample: SubSampleConvProjection,
    rel_pos: RelPositionalEncoding,
    layers: Vec<AudioLayer>,
    output_proj: Linear,      // hidden_size → output_proj_dims (with bias)
    output_proj_bias: Tensor, // [output_proj_dims]
    embed_proj: Linear,       // output_proj_dims → lm_hidden_size
    rms_norm_eps: f64,
    dtype: DType,
}

impl AudioEncoder {
    /// Load from a VarBuilder rooted at `model.`
    pub fn load(
        vb: VarBuilder,
        cfg: &AudioConfig,
        lm_hidden_size: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let at = vb.pp("audio_tower");
        let subsample = SubSampleConvProjection::load(at.pp("subsample_conv_projection"), cfg)?;
        let rel_pos = RelPositionalEncoding::new(cfg, device, dtype)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(AudioLayer::load(at.pp(format!("layers.{i}")), cfg)?);
        }

        let output_proj =
            linear_no_bias(cfg.hidden_size, cfg.output_proj_dims, at.pp("output_proj"))?;
        let output_proj_bias = at.get((cfg.output_proj_dims,), "output_proj.bias")?;

        let embed_proj = linear_no_bias(
            cfg.output_proj_dims,
            lm_hidden_size,
            vb.pp("embed_audio").pp("embedding_projection"),
        )?;

        Ok(Self {
            subsample,
            rel_pos,
            layers,
            output_proj,
            output_proj_bias,
            embed_proj,
            rms_norm_eps: cfg.rms_norm_eps,
            dtype,
        })
    }

    /// Encode a log-mel spectrogram to LM-space embeddings.
    ///
    /// `mel`: f32 Tensor `[1, T, 128]` (on model device, or CPU — will be moved).
    /// Returns: `[T/4, output_proj_dims]`.
    /// Maximum mel frames to process (15s × 100fps = 1500 frames → 375 after 4× subsampling).
    /// Full T×T attention on Metal is stable up to ~375 frames; larger inputs produce degenerate
    /// output. TODO: implement blocked chunked attention (chunk_size=12) to lift this limit.
    pub const MAX_MEL_FRAMES: usize = 1500;

    pub fn encode(&self, mel: &Tensor) -> Result<Tensor> {
        let pos_emb = self.rel_pos.get();

        // Cast mel to the model's dtype (mel is always computed as F32)
        let mel = mel.to_dtype(self.dtype)?;

        // Truncate to MAX_MEL_FRAMES to stay within full-attention memory limits.
        let t = mel.dim(1)?;
        let mel = if t > Self::MAX_MEL_FRAMES {
            mel.narrow(1, 0, Self::MAX_MEL_FRAMES)?
        } else {
            mel
        };

        // Subsampling: [1, T, 128] → [1, T/4, hidden_size]
        let xs = self.subsample.forward(&mel)?;

        // Conformer layers
        let mut xs = xs;
        for layer in &self.layers {
            xs = layer.forward(&xs, pos_emb)?;
        }

        // Output projection: linear + bias
        let xs = self.output_proj.forward(&xs)?;
        let bias = self
            .output_proj_bias
            .to_dtype(xs.dtype())?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let xs = xs.broadcast_add(&bias)?;

        // embed_audio: RMSNorm (no learnable scale) then linear projection
        let xs = rms_norm_no_scale(&xs, self.rms_norm_eps)?;
        let xs = self.embed_proj.forward(&xs)?;

        // Remove batch dim: [T/4, output_proj_dims]
        Ok(xs.squeeze(0)?)
    }
}

/// RMSNorm without a learnable scale: `x / rms(x)`.
fn rms_norm_no_scale(xs: &Tensor, eps: f64) -> Result<Tensor> {
    let x2 = xs.sqr()?;
    let mean = x2.mean_keepdim(D::Minus1)?;
    let rms = (mean + eps)?.sqrt()?;
    Ok(xs.broadcast_div(&rms)?)
}
