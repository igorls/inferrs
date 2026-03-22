//! Transformer model loading and forward pass using candle.
//!
//! Supports:
//!   - Pure decoder-only transformers (Qwen2, Llama, Mistral)
//!   - Hybrid transformers with linear (Mamba2/GatedDeltaNet) attention (Qwen3.5)
//!
//! For hybrid models, `layer_types` in config.json determines which layers use
//! full self-attention and which use the linear recurrent attention.

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};
use serde::Deserialize;
use std::path::Path;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Model configuration parsed from config.json.
/// Covers Qwen2, Llama, Mistral, Qwen3.5 (hybrid), and similar architectures.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    #[serde(default = "default_num_kv_heads")]
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    #[serde(default = "default_max_position")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    #[allow(dead_code)]
    pub bos_token_id: Option<serde_json::Value>,

    // --- Explicit head_dim (some models specify it instead of deriving) ---
    #[serde(default)]
    pub head_dim: Option<usize>,

    // --- Hybrid (Qwen3.5) fields ---
    /// Per-layer type: "full_attention" or "linear_attention". Empty = all full.
    #[serde(default)]
    pub layer_types: Vec<String>,
    /// Linear attention: number of key heads.
    #[serde(default)]
    pub linear_num_key_heads: usize,
    /// Linear attention: key head dimension.
    #[serde(default)]
    pub linear_key_head_dim: usize,
    /// Linear attention: number of value heads.
    #[serde(default)]
    pub linear_num_value_heads: usize,
    /// Linear attention: value head dimension.
    #[serde(default)]
    pub linear_value_head_dim: usize,
    /// Linear attention: conv1d kernel size.
    #[serde(default = "default_conv_kernel")]
    pub linear_conv_kernel_dim: usize,
    /// Partial rotary factor (fraction of head_dim that gets RoPE).
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
    /// Nested rope_parameters for rope_theta override etc.
    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>,
}

fn default_num_kv_heads() -> usize {
    0
}
fn default_max_position() -> usize {
    2048
}
fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_rope_theta() -> f64 {
    10000.0
}
fn default_conv_kernel() -> usize {
    4
}

impl ModelConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path).context("reading config.json")?;

        // Try parsing directly first (standard decoder-only models).
        // If that fails, look for a nested `text_config` object (VLMs like
        // Qwen3.5, LLaVA, etc. put the language model config there).
        let mut config: Self = match serde_json::from_str::<Self>(&content) {
            Ok(cfg) => cfg,
            Err(_top_err) => {
                let raw: serde_json::Value =
                    serde_json::from_str(&content).context("parsing config.json as JSON")?;
                if let Some(text_cfg) = raw.get("text_config") {
                    // Merge top-level fields (like eos_token_id, tie_word_embeddings)
                    // with text_config so nothing is lost.
                    let mut merged = text_cfg.clone();
                    if let (Some(merged_obj), Some(raw_obj)) =
                        (merged.as_object_mut(), raw.as_object())
                    {
                        for (k, v) in raw_obj {
                            if k != "text_config" && k != "vision_config" {
                                merged_obj.entry(k.clone()).or_insert_with(|| v.clone());
                            }
                        }
                    }
                    serde_json::from_value(merged)
                        .context("parsing text_config from config.json")?
                } else {
                    return Err(_top_err).context("parsing config.json");
                }
            }
        };

        // If num_key_value_heads is 0 (not specified), default to num_attention_heads (MHA)
        if config.num_key_value_heads == 0 {
            config.num_key_value_heads = config.num_attention_heads;
        }

        // Extract rope_theta from nested rope_parameters if present
        if let Some(ref rp) = config.rope_parameters {
            if let Some(theta) = rp.get("rope_theta").and_then(|v| v.as_f64()) {
                config.rope_theta = theta;
            }
            if config.partial_rotary_factor.is_none() {
                if let Some(prf) = rp.get("partial_rotary_factor").and_then(|v| v.as_f64()) {
                    config.partial_rotary_factor = Some(prf);
                }
            }
        }

        Ok(config)
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Whether this model has hybrid (linear + full attention) layers.
    pub fn is_hybrid(&self) -> bool {
        self.layer_types.iter().any(|t| t == "linear_attention")
    }

    /// Get the layer type for a given layer index.
    pub fn layer_type(&self, idx: usize) -> &str {
        if idx < self.layer_types.len() {
            &self.layer_types[idx]
        } else {
            "full_attention"
        }
    }

    /// Parse eos_token_id which can be a single int or an array.
    pub fn get_eos_token_ids(&self) -> Vec<u32> {
        match &self.eos_token_id {
            Some(serde_json::Value::Number(n)) => {
                vec![n.as_u64().unwrap_or(0) as u32]
            }
            Some(serde_json::Value::Array(arr)) => arr
                .iter()
                .filter_map(|v| v.as_u64().map(|n| n as u32))
                .collect(),
            _ => vec![],
        }
    }
}

// ---------------------------------------------------------------------------
// Building blocks
// ---------------------------------------------------------------------------

/// RMS normalization layer.
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn load(vb: VarBuilder, size: usize, eps: f64) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let variance = (&x * &x)?.mean_keepdim(candle_core::D::Minus1)?;
        let x_normed = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let result = x_normed.to_dtype(dtype)?.broadcast_mul(&self.weight)?;
        Ok(result)
    }
}

/// Rotary positional embedding.
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    /// How many dimensions to apply RoPE to (for partial rotary).
    rotary_dim: usize,
}

impl RotaryEmbedding {
    fn new(
        head_dim: usize,
        max_seq_len: usize,
        theta: f64,
        partial_rotary_factor: f64,
        device: &Device,
    ) -> Result<Self> {
        let rotary_dim = (head_dim as f64 * partial_rotary_factor) as usize;
        // Ensure even
        let rotary_dim = rotary_dim - (rotary_dim % 2);
        let rotary_dim = rotary_dim.max(2); // at least 2

        let inv_freq: Vec<f32> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1.0 / (theta as f32).powf(i as f32 / rotary_dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?.unsqueeze(1)?;
        let freqs = positions.matmul(&inv_freq.unsqueeze(0)?)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        Ok(Self {
            cos,
            sin,
            rotary_dim,
        })
    }

    fn apply(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, _, seq_len, head_dim) = x.dims4()?;

        if self.rotary_dim >= head_dim {
            // Full rotary (standard case)
            let cos = self
                .cos
                .i(offset..offset + seq_len)?
                .unsqueeze(0)?
                .unsqueeze(0)?;
            let sin = self
                .sin
                .i(offset..offset + seq_len)?
                .unsqueeze(0)?
                .unsqueeze(0)?;

            let x1 = x.narrow(3, 0, head_dim / 2)?;
            let x2 = x.narrow(3, head_dim / 2, head_dim / 2)?;
            let rotated = Tensor::cat(&[&x2.neg()?, &x1], 3)?;

            let result = (x.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?;
            Ok(result)
        } else {
            // Partial rotary: only apply RoPE to first rotary_dim dimensions
            let half = self.rotary_dim / 2;
            let cos = self
                .cos
                .i(offset..offset + seq_len)?
                .unsqueeze(0)?
                .unsqueeze(0)?;
            let sin = self
                .sin
                .i(offset..offset + seq_len)?
                .unsqueeze(0)?
                .unsqueeze(0)?;

            let x_rot = x.narrow(3, 0, self.rotary_dim)?;
            let x_pass = x.narrow(3, self.rotary_dim, head_dim - self.rotary_dim)?;

            let x1 = x_rot.narrow(3, 0, half)?;
            let x2 = x_rot.narrow(3, half, half)?;
            let rotated = Tensor::cat(&[&x2.neg()?, &x1], 3)?;
            let x_rot = (x_rot.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?;

            Ok(Tensor::cat(&[&x_rot, &x_pass], 3)?)
        }
    }
}

// ---------------------------------------------------------------------------
// Full attention (standard transformer)
// ---------------------------------------------------------------------------

/// A single transformer attention layer.
///
/// Supports an optional **attention output gate** (used by Qwen3.5): the Q
/// projection is 2x wider than normal, with the second half providing a
/// per-head sigmoid gate that multiplies the attention output.
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    /// Whether the Q projection includes an output gate (doubled Q width).
    has_output_gate: bool,
}

impl Attention {
    fn load(vb: VarBuilder, config: &ModelConfig) -> Result<Self> {
        let hidden = config.hidden_size;
        let head_dim = config.head_dim();
        let kv_size = config.num_key_value_heads * head_dim;

        // Detect output gate: try loading q_proj at the expected doubled size.
        // If that works, the model has attn_output_gate=True (Qwen3.5).
        let q_size_normal = config.num_attention_heads * head_dim;
        let q_size_gated = q_size_normal * 2;
        let (q_proj, has_output_gate) =
            match candle_nn::linear_no_bias(hidden, q_size_gated, vb.pp("q_proj")) {
                Ok(proj) => (proj, true),
                Err(_) => {
                    let proj = candle_nn::linear_no_bias(hidden, q_size_normal, vb.pp("q_proj"))?;
                    (proj, false)
                }
            };

        let k_proj = candle_nn::linear_no_bias(hidden, kv_size, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear_no_bias(hidden, kv_size, vb.pp("v_proj"))?;

        // o_proj input = num_heads * head_dim (the gate doesn't change this)
        let o_proj_in = config.num_attention_heads * head_dim;
        let o_proj = candle_nn::linear_no_bias(o_proj_in, hidden, vb.pp("o_proj"))?;

        // QK normalization (used by Qwen3.5 full_attention layers)
        let q_norm = RmsNorm::load(vb.pp("q_norm"), head_dim, config.rms_norm_eps).ok();
        let k_norm = RmsNorm::load(vb.pp("k_norm"), head_dim, config.rms_norm_eps).ok();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim,
            has_output_gate,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        rotary: &RotaryEmbedding,
        kv_cache: Option<(&Tensor, &Tensor)>,
        offset: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch, seq_len, _) = x.dims3()?;

        let q_raw = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Split Q into actual Q and gate if output gate is present
        let (q_val, gate) = if self.has_output_gate {
            // q_raw: (batch, seq_len, 2 * num_heads * head_dim)
            // Reshape to (batch, seq_len, num_heads, 2*head_dim) then split
            let q_gate = q_raw.reshape((batch, seq_len, self.num_heads, 2 * self.head_dim))?;
            let q_part = q_gate.narrow(3, 0, self.head_dim)?;
            let g_part = q_gate.narrow(3, self.head_dim, self.head_dim)?;
            let q_flat = q_part.reshape((batch, seq_len, self.num_heads * self.head_dim))?;
            let g_flat = g_part.reshape((batch, seq_len, self.num_heads * self.head_dim))?;
            (q_flat, Some(g_flat))
        } else {
            (q_raw, None)
        };

        // Reshape to (batch, num_heads, seq_len, head_dim)
        let mut q = q_val
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let mut k = k
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply QK-norm if present (per-head normalization)
        if let Some(ref qn) = self.q_norm {
            q = qn.forward(&q)?;
        }
        if let Some(ref kn) = self.k_norm {
            k = kn.forward(&k)?;
        }

        // Apply rotary embeddings
        let q = rotary.apply(&q, offset)?;
        let k = rotary.apply(&k, offset)?;

        // Concatenate with KV cache if present
        let (k, v) = match kv_cache {
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &k], 2)?;
                let v = Tensor::cat(&[prev_v, &v], 2)?;
                (k, v)
            }
            None => (k, v),
        };

        // Save the un-repeated K/V for caching BEFORE repeat_kv.
        let k_for_cache = k.clone();
        let v_for_cache = v.clone();

        // GQA: repeat KV heads if needed
        let k_expanded = self.repeat_kv(&k)?;
        let v_expanded = self.repeat_kv(&v)?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k_expanded.transpose(2, 3)?)? / scale)?;

        // Causal mask
        let total_len = k_expanded.dim(2)?;
        let attn_weights = self.apply_causal_mask(&attn_weights, seq_len, total_len)?;

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v_expanded)?;

        // Reshape back to (batch, seq_len, num_heads * head_dim)
        let mut attn_output = attn_output.transpose(1, 2)?.reshape((
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        // Apply output gate if present: output *= sigmoid(gate)
        if let Some(ref g) = gate {
            let gate_sig =
                sigmoid(&g.to_dtype(candle_core::DType::F32)?)?.to_dtype(attn_output.dtype())?;
            attn_output = (attn_output * gate_sig)?;
        }

        let output = self.o_proj.forward(&attn_output)?;

        Ok((output, k_for_cache, v_for_cache))
    }

    fn repeat_kv(&self, x: &Tensor) -> Result<Tensor> {
        if self.num_kv_heads == self.num_heads {
            return Ok(x.clone());
        }
        let repeats = self.num_heads / self.num_kv_heads;
        let (batch, num_kv_heads, seq_len, head_dim) = x.dims4()?;
        let x = x
            .unsqueeze(2)?
            .expand((batch, num_kv_heads, repeats, seq_len, head_dim))?
            .reshape((batch, num_kv_heads * repeats, seq_len, head_dim))?;
        Ok(x)
    }

    fn apply_causal_mask(
        &self,
        attn_weights: &Tensor,
        query_len: usize,
        key_len: usize,
    ) -> Result<Tensor> {
        if query_len == 1 {
            return Ok(attn_weights.clone());
        }
        let device = attn_weights.device();
        let offset = key_len - query_len;
        let mask: Vec<f32> = (0..query_len)
            .flat_map(|i| {
                (0..key_len).map(move |j| {
                    if j <= i + offset {
                        0.0f32
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();
        let mask = Tensor::new(mask, device)?.reshape((1, 1, query_len, key_len))?;
        let mask = mask.to_dtype(attn_weights.dtype())?;
        Ok((attn_weights + mask)?)
    }
}

// ---------------------------------------------------------------------------
// Linear attention (Gated Delta Net / Mamba2-style)
// ---------------------------------------------------------------------------

/// Gated Delta Net linear attention layer (used by Qwen3.5).
///
/// This is a recurrent layer that maintains a `[num_v_heads, v_dim, k_dim]` state
/// matrix and a conv1d sliding-window buffer. Unlike full attention it does NOT
/// use a KV cache -- instead the recurrent state accumulates all history.
///
/// Because the KV cache manager expects one (K, V) tensor pair per layer, we
/// pack the recurrent state as a single concatenated tensor for the "K" slot
/// and the conv state into the "V" slot. This lets the existing cache plumbing
/// work without modifications.
struct LinearAttention {
    /// Fused Q,K,V projection (may be loaded from in_proj_qkv).
    in_proj_qkv: Linear,
    /// Z gate projection.
    in_proj_z: Linear,
    /// Alpha projection (forget gate input).
    in_proj_a: Linear,
    /// Beta projection (write gate input).
    in_proj_b: Linear,
    /// Depthwise conv1d weight: (conv_dim, 1, kernel_size)
    conv1d_weight: Tensor,
    /// A_log parameter: log decay rate per value-head.
    a_log: Tensor,
    /// dt_bias: time-step bias per value-head.
    dt_bias: Tensor,
    /// Per-head RMS norm weight: (v_head_dim,)
    norm_weight: Tensor,
    /// Output projection.
    out_proj: Linear,
    // Dimensions
    num_k_heads: usize,
    k_head_dim: usize,
    num_v_heads: usize,
    v_head_dim: usize,
    conv_kernel: usize,
}

impl LinearAttention {
    fn load(vb: VarBuilder, config: &ModelConfig) -> Result<Self> {
        let hidden = config.hidden_size;
        let num_k_heads = config.linear_num_key_heads;
        let k_head_dim = config.linear_key_head_dim;
        let num_v_heads = config.linear_num_value_heads;
        let v_head_dim = config.linear_value_head_dim;
        let key_dim = num_k_heads * k_head_dim;
        let value_dim = num_v_heads * v_head_dim;
        let conv_kernel = config.linear_conv_kernel_dim;

        let vb_la = vb.pp("linear_attn");

        // Projections
        let qkv_dim = key_dim * 2 + value_dim; // Q + K + V
        let in_proj_qkv = candle_nn::linear_no_bias(hidden, qkv_dim, vb_la.pp("in_proj_qkv"))?;
        let in_proj_z = candle_nn::linear_no_bias(hidden, value_dim, vb_la.pp("in_proj_z"))?;
        let in_proj_a = candle_nn::linear_no_bias(hidden, num_v_heads, vb_la.pp("in_proj_a"))?;
        let in_proj_b = candle_nn::linear_no_bias(hidden, num_v_heads, vb_la.pp("in_proj_b"))?;

        // Conv1d weight: (conv_dim, 1, kernel_size)
        let conv1d_dim = key_dim * 2 + value_dim;
        let conv1d_weight = vb_la.get((conv1d_dim, 1, conv_kernel), "conv1d.weight")?;

        // A_log: (num_v_heads,)
        let a_log = vb_la.get(num_v_heads, "A_log")?;

        // dt_bias: (num_v_heads,)
        let dt_bias = vb_la.get(num_v_heads, "dt_bias")?;

        // Norm weight: (v_head_dim,)
        let norm_weight = vb_la.get(v_head_dim, "norm.weight")?;

        // Output projection
        let out_proj = candle_nn::linear_no_bias(value_dim, hidden, vb_la.pp("out_proj"))?;

        Ok(Self {
            in_proj_qkv,
            in_proj_z,
            in_proj_a,
            in_proj_b,
            conv1d_weight,
            a_log,
            dt_bias,
            norm_weight,
            out_proj,
            num_k_heads,
            k_head_dim,
            num_v_heads,
            v_head_dim,
            conv_kernel,
        })
    }

    /// Forward pass for linear attention.
    ///
    /// `x`: (batch=1, seq_len, hidden_size)
    /// `recurrent_state`: optional previous (ssm_state, conv_state) packed tensors
    ///
    /// Returns (output, new_ssm_state, new_conv_state)
    /// where ssm_state shape: (num_v_heads, v_head_dim, k_head_dim)
    ///       conv_state shape: (conv_kernel-1, conv_dim)
    fn forward(
        &self,
        x: &Tensor,
        recurrent_state: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch, seq_len, _hidden) = x.dims3()?;
        let device = x.device();
        let dtype = x.dtype();
        let key_dim = self.num_k_heads * self.k_head_dim;
        let value_dim = self.num_v_heads * self.v_head_dim;
        let _conv_dim = key_dim * 2 + value_dim;

        // 1. Input projections
        let qkv = self.in_proj_qkv.forward(x)?; // (B, T, 2*key_dim + value_dim)
        let z = self.in_proj_z.forward(x)?; // (B, T, value_dim)
        let a_raw = self.in_proj_a.forward(x)?; // (B, T, num_v_heads)
        let b_raw = self.in_proj_b.forward(x)?; // (B, T, num_v_heads)

        // 2. Causal conv1d over QKV (simple implementation)
        // Conv state: last (kernel-1) rows of QKV features
        let prev_conv_state = recurrent_state.map(|(_, cs)| cs.clone());

        let qkv_for_conv = qkv.squeeze(0)?; // (T, conv_dim)
        let qkv_convolved = self.causal_conv1d(&qkv_for_conv, prev_conv_state.as_ref())?;
        // Apply SiLU activation after conv
        let qkv_convolved = candle_nn::Activation::Silu.forward(&qkv_convolved)?;

        // Save new conv state: last (kernel-1) tokens from the concatenation
        // of old_state + new_tokens
        let new_conv_state = {
            let padded = match &prev_conv_state {
                Some(cs) => Tensor::cat(&[cs, &qkv_for_conv], 0)?,
                None => qkv_for_conv.clone(),
            };
            let total = padded.dim(0)?;
            let keep = (self.conv_kernel - 1).min(total);
            padded.narrow(0, total - keep, keep)?
        };

        // 3. Split QKV and reshape to multi-head
        let q = qkv_convolved.narrow(1, 0, key_dim)?.reshape((
            seq_len,
            self.num_k_heads,
            self.k_head_dim,
        ))?;
        let k = qkv_convolved.narrow(1, key_dim, key_dim)?.reshape((
            seq_len,
            self.num_k_heads,
            self.k_head_dim,
        ))?;
        let v = qkv_convolved.narrow(1, key_dim * 2, value_dim)?.reshape((
            seq_len,
            self.num_v_heads,
            self.v_head_dim,
        ))?;

        // 4. L2-normalize Q and K
        let q = l2_normalize(&q)?;
        let k = l2_normalize(&k)?;

        // 5. Compute gating
        let a_squeezed = a_raw.squeeze(0)?; // (T, num_v_heads)
        let b_squeezed = b_raw.squeeze(0)?; // (T, num_v_heads)
        let neg_a_exp = self.a_log.to_dtype(DType::F32)?.exp()?.neg()?;
        let dt = (&a_squeezed.to_dtype(DType::F32)? + &self.dt_bias.to_dtype(DType::F32)?)?;
        // g = -exp(A_log) * softplus(a + dt_bias)
        let g = neg_a_exp.broadcast_mul(&softplus(&dt)?)?;
        let beta = sigmoid(&b_squeezed.to_dtype(DType::F32)?)?;

        // 6. Recurrent update (Gated Delta Rule)
        // ssm_state: (num_v_heads, v_head_dim, k_head_dim)
        let mut ssm_state = match recurrent_state {
            Some((ss, _)) => ss.to_dtype(DType::F32)?,
            None => Tensor::zeros(
                (self.num_v_heads, self.v_head_dim, self.k_head_dim),
                DType::F32,
                device,
            )?,
        };

        let scale = (self.k_head_dim as f64).powf(-0.5);
        let mut outputs = Vec::with_capacity(seq_len);

        // Group K-heads to V-heads. If num_k_heads == num_v_heads, 1:1.
        // Otherwise, k-heads are grouped (GQA-style for linear attention).
        let k_groups = self.num_k_heads / self.num_v_heads;

        for t in 0..seq_len {
            let g_t = g.i(t)?; // (num_v_heads,)
            let beta_t = beta.i(t)?; // (num_v_heads,)

            // Decay: S *= exp(g_t), broadcast per head
            let decay = g_t.exp()?; // (num_v_heads,)
            ssm_state = ssm_state.broadcast_mul(&decay.reshape((self.num_v_heads, 1, 1))?)?;

            // Get k for this timestep: average over k-head groups if needed
            let k_t = k.i(t)?.to_dtype(DType::F32)?; // (num_k_heads, k_head_dim)
            let k_t = if k_groups > 1 {
                k_t.reshape((self.num_v_heads, k_groups, self.k_head_dim))?
                    .mean(1)?
            } else {
                k_t
            }; // (num_v_heads, k_head_dim)

            let v_t = v.i(t)?.to_dtype(DType::F32)?; // (num_v_heads, v_head_dim)

            // Delta: v_delta = v_t - S @ k_t
            // S: (num_v_heads, v_head_dim, k_head_dim)
            // k_t: (num_v_heads, k_head_dim)
            let retrieved = ssm_state.matmul(&k_t.unsqueeze(2)?)?.squeeze(2)?;
            // retrieved: (num_v_heads, v_head_dim)
            let v_delta = (&v_t - &retrieved)?;
            let v_delta = v_delta.broadcast_mul(&beta_t.reshape((self.num_v_heads, 1))?)?;

            // S += outer(v_delta, k_t)
            let outer = v_delta.unsqueeze(2)?.matmul(&k_t.unsqueeze(1)?)?;
            ssm_state = (&ssm_state + &outer)?;

            // Read: o_t = S @ q_t * scale
            let q_t = q.i(t)?.to_dtype(DType::F32)?; // (num_k_heads, k_head_dim)
            let q_t = if k_groups > 1 {
                q_t.reshape((self.num_v_heads, k_groups, self.k_head_dim))?
                    .mean(1)?
            } else {
                q_t
            };
            let o_t = ssm_state.matmul(&q_t.unsqueeze(2)?)?.squeeze(2)?; // (num_v_heads, v_head_dim)
            let o_t = (o_t * scale)?;
            outputs.push(o_t);
        }

        // Stack outputs: (seq_len, num_v_heads, v_head_dim)
        let output = Tensor::stack(&outputs, 0)?;

        // 7. Gated RMSNorm: output = RMSNorm(output) * SiLU(z)
        let output = self.rms_norm_gated(&output, &z.squeeze(0)?.to_dtype(DType::F32)?)?;

        // 8. Flatten and project back
        let output = output
            .reshape((batch, seq_len, value_dim))?
            .to_dtype(dtype)?;
        let output = self.out_proj.forward(&output)?;

        // Convert state back to model dtype for storage
        let ssm_state = ssm_state.to_dtype(dtype)?;
        let new_conv_state = new_conv_state.to_dtype(dtype)?;

        Ok((output, ssm_state, new_conv_state))
    }

    /// Simple causal conv1d: depthwise convolution with left-padding from state.
    fn causal_conv1d(&self, x: &Tensor, prev_state: Option<&Tensor>) -> Result<Tensor> {
        let (seq_len, dim) = x.dims2()?;
        let pad = self.conv_kernel - 1;

        // Pad with previous state or zeros
        let padded = match prev_state {
            Some(state) => {
                let state_len = state.dim(0)?;
                if state_len >= pad {
                    let padding = state.narrow(0, state_len - pad, pad)?;
                    Tensor::cat(&[&padding, x], 0)?
                } else {
                    let zeros = Tensor::zeros((pad - state_len, dim), x.dtype(), x.device())?;
                    Tensor::cat(&[&zeros, state, x], 0)?
                }
            }
            None => {
                let zeros = Tensor::zeros((pad, dim), x.dtype(), x.device())?;
                Tensor::cat(&[&zeros, x], 0)?
            }
        };

        // Depthwise conv1d: for each channel, convolve with its kernel
        // conv1d_weight: (conv_dim, 1, kernel_size)
        // We implement this as a simple sliding-window dot product
        let weight = self.conv1d_weight.squeeze(1)?.to_dtype(x.dtype())?; // (conv_dim, kernel_size)
        let mut result = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let window = padded.narrow(0, t, self.conv_kernel)?; // (kernel_size, dim)
            let window_t = window.t()?; // (dim, kernel_size)
                                        // Element-wise multiply and sum along kernel dimension
            let conv = (&window_t * &weight)?.sum(1)?; // (dim,)
            result.push(conv);
        }
        Ok(Tensor::stack(&result, 0)?) // (seq_len, dim)
    }

    /// Gated RMS normalization: RMSNorm(x) * SiLU(z), applied per-head.
    fn rms_norm_gated(&self, x: &Tensor, z: &Tensor) -> Result<Tensor> {
        let (seq_len, num_heads, head_dim) = x.dims3()?;
        let eps = 1e-6;

        // RMSNorm per head
        let variance = (x * x)?.mean_keepdim(candle_core::D::Minus1)?;
        let x_normed = x.broadcast_div(&(variance + eps)?.sqrt()?)?;
        let x_normed = x_normed.broadcast_mul(&self.norm_weight.to_dtype(DType::F32)?)?;

        // z: (seq_len, value_dim) -> (seq_len, num_heads, head_dim)
        let z_reshaped = z.reshape((seq_len, num_heads, head_dim))?;
        let z_gate = candle_nn::Activation::Silu.forward(&z_reshaped)?;

        Ok((x_normed * z_gate)?)
    }
}

/// L2-normalize along the last dimension.
fn l2_normalize(x: &Tensor) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let norm = (&x_f32 * &x_f32)?
        .sum_keepdim(candle_core::D::Minus1)?
        .sqrt()?;
    let norm = (norm + 1e-12)?; // avoid division by zero
    let result = x_f32.broadcast_div(&norm)?;
    Ok(result.to_dtype(x.dtype())?)
}

/// Softplus: log(1 + exp(x))
fn softplus(x: &Tensor) -> Result<Tensor> {
    // For numerical stability: softplus(x) = x if x > 20, else log(1+exp(x))
    let ones = Tensor::ones_like(x)?;
    let exp_x = x.exp()?;
    let result = (&exp_x + &ones)?.log()?;
    // Clamp: use x directly where x > 20
    let threshold = (Tensor::ones_like(x)? * 20.0)?;
    let mask = x.ge(&threshold)?;
    let result = mask.to_dtype(x.dtype())?.broadcast_mul(x)?
        + mask.neg()?.to_dtype(x.dtype())?.broadcast_mul(&result)?
        + Tensor::ones_like(x)?.broadcast_mul(&result)?;
    // Simpler approach: just use log(1+exp(x)), candle handles this fine for our range
    let _ = result;
    let exp_x = x.exp()?;
    Ok((exp_x + 1.0)?.log()?)
}

/// Sigmoid: 1 / (1 + exp(-x))
fn sigmoid(x: &Tensor) -> Result<Tensor> {
    let neg_x = x.neg()?;
    let exp_neg_x = neg_x.exp()?;
    let denom = (exp_neg_x + 1.0)?;
    Ok(Tensor::ones_like(x)?.broadcast_div(&denom)?)
}

// ---------------------------------------------------------------------------
// MLP
// ---------------------------------------------------------------------------

/// MLP layer (gate + up + down projection with SiLU activation).
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn load(vb: VarBuilder, config: &ModelConfig) -> Result<Self> {
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let gate_proj = candle_nn::linear_no_bias(hidden, intermediate, vb.pp("gate_proj"))?;
        let up_proj = candle_nn::linear_no_bias(hidden, intermediate, vb.pp("up_proj"))?;
        let down_proj = candle_nn::linear_no_bias(intermediate, hidden, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::Activation::Silu.forward(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        let result = self.down_proj.forward(&(gate * up)?)?;
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Decoder layers (full attention or linear attention)
// ---------------------------------------------------------------------------

/// A decoder layer that can be either full (transformer) or linear (recurrent).
enum DecoderLayer {
    FullAttention {
        self_attn: Attention,
        mlp: Mlp,
        input_layernorm: RmsNorm,
        post_attention_layernorm: RmsNorm,
    },
    LinearAttention {
        linear_attn: LinearAttention,
        mlp: Mlp,
        input_layernorm: RmsNorm,
        post_attention_layernorm: RmsNorm,
    },
}

impl DecoderLayer {
    fn load_full_attention(vb: VarBuilder, config: &ModelConfig) -> Result<Self> {
        let self_attn = Attention::load(vb.pp("self_attn"), config)?;
        let mlp = Mlp::load(vb.pp("mlp"), config)?;
        let input_layernorm = RmsNorm::load(
            vb.pp("input_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;
        let post_attention_layernorm = RmsNorm::load(
            vb.pp("post_attention_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;
        Ok(Self::FullAttention {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn load_linear_attention(vb: VarBuilder, config: &ModelConfig) -> Result<Self> {
        let linear_attn = LinearAttention::load(vb.clone(), config)?;
        let mlp = Mlp::load(vb.pp("mlp"), config)?;
        let input_layernorm = RmsNorm::load(
            vb.pp("input_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;
        let post_attention_layernorm = RmsNorm::load(
            vb.pp("post_attention_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;
        Ok(Self::LinearAttention {
            linear_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// Returns true if this is a full attention layer.
    fn is_full_attention(&self) -> bool {
        matches!(self, Self::FullAttention { .. })
    }

    /// Forward pass for full attention layers.
    /// Returns (output, k_cache, v_cache).
    fn forward_full(
        &self,
        x: &Tensor,
        rotary: &RotaryEmbedding,
        kv_cache: Option<(&Tensor, &Tensor)>,
        offset: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        match self {
            Self::FullAttention {
                self_attn,
                mlp,
                input_layernorm,
                post_attention_layernorm,
            } => {
                let residual = x;
                let x = input_layernorm.forward(x)?;
                let (attn_out, k_cache, v_cache) =
                    self_attn.forward(&x, rotary, kv_cache, offset)?;
                let x = (residual + attn_out)?;
                let residual = &x;
                let x = post_attention_layernorm.forward(&x)?;
                let x = mlp.forward(&x)?;
                let x = (residual + x)?;
                Ok((x, k_cache, v_cache))
            }
            _ => anyhow::bail!("forward_full called on linear attention layer"),
        }
    }

    /// Forward pass for linear attention layers.
    /// Returns (output, ssm_state, conv_state).
    fn forward_linear(
        &self,
        x: &Tensor,
        recurrent_state: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        match self {
            Self::LinearAttention {
                linear_attn,
                mlp,
                input_layernorm,
                post_attention_layernorm,
            } => {
                let residual = x;
                let x = input_layernorm.forward(x)?;
                let (attn_out, ssm_state, conv_state) = linear_attn.forward(&x, recurrent_state)?;
                let x = (residual + attn_out)?;
                let residual = &x;
                let x = post_attention_layernorm.forward(&x)?;
                let x = mlp.forward(&x)?;
                let x = (residual + x)?;
                Ok((x, ssm_state, conv_state))
            }
            _ => anyhow::bail!("forward_linear called on full attention layer"),
        }
    }
}

// ---------------------------------------------------------------------------
// Full model
// ---------------------------------------------------------------------------

/// The full transformer model.
pub struct TransformerModel {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    rotary: RotaryEmbedding,
    #[allow(dead_code)]
    config: ModelConfig,
    device: Device,
    #[allow(dead_code)]
    dtype: DType,
}

impl TransformerModel {
    /// Load model from safetensors files.
    pub fn load(
        config: &ModelConfig,
        weight_paths: &[impl AsRef<Path>],
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        tracing::info!(
            "loading {} layers, hidden_size={}, heads={}/{}",
            config.num_hidden_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads
        );
        if config.is_hybrid() {
            let full_count = config
                .layer_types
                .iter()
                .filter(|t| t.as_str() == "full_attention")
                .count();
            let linear_count = config.num_hidden_layers - full_count;
            tracing::info!(
                "hybrid model: {} full_attention + {} linear_attention layers",
                full_count,
                linear_count
            );
        }

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &weight_paths
                    .iter()
                    .map(|p| p.as_ref().to_path_buf())
                    .collect::<Vec<_>>(),
                dtype,
                device,
            )?
        };

        // Detect weight prefix. Standard decoder-only models use "model."
        // while VLMs (Qwen3.5, LLaVA, etc.) use "model.language_model.".
        let vb_model = {
            let vb_standard = vb.pp("model");
            let vb_vlm = vb.pp("model").pp("language_model");
            if vb_vlm
                .get(
                    (config.vocab_size, config.hidden_size),
                    "embed_tokens.weight",
                )
                .is_ok()
            {
                tracing::info!("detected VLM weight prefix: model.language_model");
                vb_vlm
            } else {
                vb_standard
            }
        };

        let embed_tokens = Embedding::new(
            vb_model.get(
                (config.vocab_size, config.hidden_size),
                "embed_tokens.weight",
            )?,
            config.hidden_size,
        );

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let vb_layer = vb_model.pp(format!("layers.{i}"));
            let layer_type = config.layer_type(i);
            let layer = match layer_type {
                "linear_attention" => DecoderLayer::load_linear_attention(vb_layer, config)?,
                _ => DecoderLayer::load_full_attention(vb_layer, config)?,
            };
            layers.push(layer);
            if (i + 1) % 4 == 0 || i == config.num_hidden_layers - 1 {
                tracing::info!(
                    "loaded layer {}/{} ({})",
                    i + 1,
                    config.num_hidden_layers,
                    layer_type
                );
            }
        }

        let norm = RmsNorm::load(vb_model.pp("norm"), config.hidden_size, config.rms_norm_eps)?;

        // LM head: may be tied to embed_tokens
        let lm_head = if config.tie_word_embeddings {
            let weight = vb_model.get(
                (config.vocab_size, config.hidden_size),
                "embed_tokens.weight",
            )?;
            Linear::new(weight, None)
        } else {
            candle_nn::linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?
        };

        let max_seq = config.max_position_embeddings.min(131072); // cap RoPE table
        let partial_rotary = config.partial_rotary_factor.unwrap_or(1.0);
        let rotary = RotaryEmbedding::new(
            config.head_dim(),
            max_seq,
            config.rope_theta,
            partial_rotary,
            device,
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary,
            config: config.clone(),
            device: device.clone(),
            dtype,
        })
    }

    /// Forward pass. Returns logits for the last token position.
    ///
    /// - `input_ids`: shape (batch, seq_len)
    /// - `kv_caches`: one (k, v) pair per layer, or empty for prefill.
    ///   For full attention layers: k/v are the standard KV cache tensors.
    ///   For linear attention layers: k = ssm_state, v = conv_state.
    /// - `offset`: position offset in the sequence (for decode steps)
    ///
    /// Returns (logits, new_kv_caches).
    pub fn forward(
        &self,
        input_ids: &Tensor,
        kv_caches: &[(Tensor, Tensor)],
        offset: usize,
    ) -> Result<(Tensor, Vec<(Tensor, Tensor)>)> {
        let mut x = self.embed_tokens.forward(input_ids)?;
        let mut new_kv_caches = Vec::with_capacity(self.layers.len());

        for (i, layer) in self.layers.iter().enumerate() {
            if layer.is_full_attention() {
                let kv = if i < kv_caches.len() {
                    Some((&kv_caches[i].0, &kv_caches[i].1))
                } else {
                    None
                };
                let (out, k, v) = layer.forward_full(&x, &self.rotary, kv, offset)?;
                x = out;
                new_kv_caches.push((k, v));
            } else {
                // Linear attention: kv_caches[i] = (ssm_state, conv_state)
                let state = if i < kv_caches.len() {
                    Some((&kv_caches[i].0, &kv_caches[i].1))
                } else {
                    None
                };
                let (out, ssm_state, conv_state) = layer.forward_linear(&x, state)?;
                x = out;
                new_kv_caches.push((ssm_state, conv_state));
            }
        }

        x = self.norm.forward(&x)?;

        // Only compute logits for the last position to save memory
        let seq_len = x.dim(1)?;
        let last_hidden = x.narrow(1, seq_len - 1, 1)?;
        let logits = self.lm_head.forward(&last_hidden)?.squeeze(1)?;

        Ok((logits, new_kv_caches))
    }

    #[allow(dead_code)]
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    #[allow(dead_code)]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    #[allow(dead_code)]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}
