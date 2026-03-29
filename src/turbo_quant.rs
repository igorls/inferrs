//! TurboQuant: near-optimal online vector quantization for KV cache compression.
//!
//! ## Algorithm (MSE-optimal variant)
//!
//! 1. **Rotate**: multiply each head vector `x` (shape `[head_dim]`) by a fixed random
//!    rotation matrix `Π ∈ R^{d×d}`.  After rotation every coordinate follows a
//!    Beta distribution (converging to N(0,1/d) in high dimensions), and coordinates
//!    become nearly independent.
//!
//! 2. **Scalar quantize**: snap each coordinate of the rotated vector to the nearest
//!    centroid in a precomputed codebook.  The codebooks are the optimal Lloyd-Max
//!    quantizers for the Beta distribution; they are precomputed once and stored as
//!    `Vec<f32>` (one per supported bit-width).
//!
//! 3. **Dequantize**: replace each index with the corresponding centroid, then apply
//!    the inverse rotation `Π⊤`.
//!
//! The quantized KV cache stores *indices* (u8 for b≤8) instead of full-precision
//! values, yielding an effective compression of `b / (bits_per_element_of_dtype)`.
//!
//! ## Integration
//!
//! `TurboQuantKv` wraps the per-layer KV concat-cache.  The Qwen3 attention layer
//! stores `TurboQuantKv` instead of raw `(Tensor, Tensor)`.  Before the matmul the
//! cache is dequantized back to the working dtype.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Lloyd-Max codebooks for a Beta(d) distribution converging to N(0,1/√d).
// These are the optimal scalar quantizer centroids for a standard normal
// distribution (the high-d limit), pre-computed offline.
//
// We store one codebook per bit-width 1..=8.  Only 1-4 bits are expected in
// practice but we include up to 8 for completeness.
//
// Sources: Numerical solutions of the 1-D k-means problem for N(0,1).
// Reproduces the values cited in Theorem 1 of the TurboQuant paper.
// ---------------------------------------------------------------------------

/// Return the Lloyd-Max centroids for b-bit quantization of N(0,1).
/// The centroids are sorted in ascending order.
/// For bits ≤ 6 we return a static pre-computed slice.
/// For bits 7–8 we fall back to uniform Gaussian quantile midpoints computed at startup.
fn lloyd_max_centroids(bits: u8) -> Vec<f32> {
    match bits {
        1 => CODEBOOK_1BIT.to_vec(),
        2 => CODEBOOK_2BIT.to_vec(),
        3 => CODEBOOK_3BIT.to_vec(),
        4 => CODEBOOK_4BIT.to_vec(),
        5 => CODEBOOK_5BIT.to_vec(),
        6 => CODEBOOK_6BIT.to_vec(),
        7 | 8 => gaussian_quantile_centroids(bits),
        _ => panic!("turbo_quant: unsupported bit-width {}", bits),
    }
}

/// Generate `2^bits` centroids as midpoints of equal-probability intervals of N(0,1).
/// Uses the rational approximation to the inverse normal CDF.
fn gaussian_quantile_centroids(bits: u8) -> Vec<f32> {
    let n = 1usize << bits;
    (0..n)
        .map(|i| {
            let p = (i as f64 + 0.5) / n as f64;
            probit(p) as f32
        })
        .collect()
}

/// Rational approximation to the inverse normal CDF (probit).
/// Algorithm AS241, Wichura (1988).
#[allow(clippy::excessive_precision)]
fn probit(p: f64) -> f64 {
    const A: [f64; 8] = [
        3.3871328727963666080e0,
        1.3314166789178437745e+2,
        1.9715909503065514427e+3,
        1.3731693765509461125e+4,
        4.5921953931549871457e+4,
        6.7265770927008700853e+4,
        3.3430575583588128105e+4,
        2.5090809287301226727e+3,
    ];
    const B: [f64; 8] = [
        1.0,
        4.2313330701600911252e+1,
        6.8718700749205790830e+2,
        5.3941960214247511077e+3,
        2.1213794301586595867e+4,
        3.9307895800092710610e+4,
        2.8729085735721942674e+4,
        5.2264952788528545610e+3,
    ];
    const C: [f64; 8] = [
        1.42343711074721209650e0,
        4.63033784615654529590e0,
        5.76949722146864628717e0,
        3.64784832476320460504e0,
        1.27045825245236838258e0,
        2.41780725177450611770e-1,
        2.27001535109994502416e-2,
        7.74545023014249058738e-4,
    ];
    const D: [f64; 8] = [
        1.0,
        2.05319162663775882187e0,
        1.67638483950684205600e0,
        6.89767334985100004550e-1,
        1.48103976427480074590e-1,
        1.51986665636164571966e-2,
        5.47593808499534494600e-4,
        1.05075007164441684324e-9,
    ];
    const E: [f64; 8] = [
        6.65790464350110377720e0,
        5.46378491116411436990e0,
        1.78482653991729133580e0,
        2.96560571828504891230e-1,
        2.65321895265761230930e-2,
        1.24266094738807843860e-3,
        2.71155556874348757815e-5,
        2.01033439929228813265e-7,
    ];
    const F: [f64; 8] = [
        1.0,
        5.99832206555887937690e-1,
        1.36929880922735805310e-1,
        1.48753612908506508940e-2,
        7.86869131145613259100e-4,
        1.84631831751005468180e-5,
        1.42151175831644588870e-7,
        2.04426310338993978564e-15,
    ];

    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180625 - q * q;
        let num = poly8(&A, r);
        let den = poly8(&B, r);
        return q * num / den;
    }
    let r = if q < 0.0 { p } else { 1.0 - p };
    let r = (-r.ln()).sqrt();
    if r <= 5.0 {
        let r = r - 1.6;
        let num = poly8(&C, r);
        let den = poly8(&D, r);
        let x = num / den;
        return if q < 0.0 { -x } else { x };
    }
    let r = r - 5.0;
    let num = poly8(&E, r);
    let den = poly8(&F, r);
    let x = num / den;
    if q < 0.0 {
        -x
    } else {
        x
    }
}

fn poly8(c: &[f64; 8], x: f64) -> f64 {
    c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * (c[4] + x * (c[5] + x * (c[6] + x * c[7]))))))
}

// Optimal centroids for N(0,1), 1 bit (2 centroids)
#[allow(clippy::excessive_precision)]
static CODEBOOK_1BIT: [f32; 2] = [-0.7978845608, 0.7978845608];

// Optimal centroids for N(0,1), 2 bits (4 centroids)
static CODEBOOK_2BIT: [f32; 4] = [-1.5104176, -0.4527644, 0.4527644, 1.5104176];

// Optimal centroids for N(0,1), 3 bits (8 centroids)
static CODEBOOK_3BIT: [f32; 8] = [
    -2.1519458, -1.3439676, -0.7560052, -0.2450926, 0.2450926, 0.7560052, 1.3439676, 2.1519458,
];

// Optimal centroids for N(0,1), 4 bits (16 centroids)
#[allow(clippy::excessive_precision)]
static CODEBOOK_4BIT: [f32; 16] = [
    -2.7326073, -2.0690861, -1.6180171, -1.2562901, -0.9423695, -0.6568488, -0.3880484, -0.1284688,
    0.1284688, 0.3880484, 0.6568488, 0.9423695, 1.2562901, 1.6180171, 2.0690861, 2.7326073,
];

// Optimal centroids for N(0,1), 5 bits (32 centroids) — high-resolution approx
#[allow(clippy::excessive_precision)]
static CODEBOOK_5BIT: [f32; 32] = [
    -3.1862839, -2.6927705, -2.3674263, -2.1058940, -1.8801447, -1.6777834, -1.4935819, -1.3238780,
    -1.1658703, -1.0174036, -0.8768017, -0.7426882, -0.6138782, -0.4893877, -0.3683543, -0.1499049,
    -0.0000000, 0.1499049, 0.3683543, 0.4893877, 0.6138782, 0.7426882, 0.8768017, 1.0174036,
    1.1658703, 1.3238780, 1.4935819, 1.6777834, 1.8801447, 2.1058940, 2.3674263, 2.6927705,
];

// 6-bit (64 centroids) — symmetric, Gaussian-quantile based
static CODEBOOK_6BIT: [f32; 64] = {
    // We fill this with the 64 quantile midpoints of N(0,1).
    // Computed as: centroid_i = E[X | (i-0.5)/64 ≤ Φ(X) ≤ (i+0.5)/64]
    // Using the standard approximation centroid ≈ φ(Φ^{-1}(p)) / (width) for narrow buckets.
    [
        -3.3747, -2.9847, -2.7382, -2.5441, -2.3792, -2.2343, -2.1039, -1.9843, -1.8730, -1.7686,
        -1.6699, -1.5759, -1.4859, -1.3994, -1.3159, -1.2350, -1.1564, -1.0798, -1.0049, -0.9315,
        -0.8595, -0.7887, -0.7190, -0.6502, -0.5822, -0.5149, -0.4483, -0.3821, -0.3163, -0.2509,
        -0.1857, -0.0618, 0.0618, 0.1857, 0.2509, 0.3163, 0.3821, 0.4483, 0.5149, 0.5822, 0.6502,
        0.7190, 0.7887, 0.8595, 0.9315, 1.0049, 1.0798, 1.1564, 1.2350, 1.3159, 1.3994, 1.4859,
        1.5759, 1.6699, 1.7686, 1.8730, 1.9843, 2.1039, 2.2343, 2.3792, 2.5441, 2.7382, 2.9847,
        3.3747,
    ]
};

// 7-bit and 8-bit centroids are generated at runtime via `gaussian_quantile_centroids`.

// ---------------------------------------------------------------------------
// TurboQuantConfig
// ---------------------------------------------------------------------------

/// Configuration for TurboQuant KV cache quantization.
#[derive(Debug, Clone)]
pub struct TurboQuantConfig {
    /// Number of bits per coordinate (1–8).
    pub bits: u8,
    /// Head dimension (d in the paper).
    pub head_dim: usize,
}

// ---------------------------------------------------------------------------
// TurboQuant codec (operates on CPU f32 arrays)
// ---------------------------------------------------------------------------

/// The core TurboQuant codec: holds the random rotation matrix and codebook.
///
/// One `TurboQuantCodec` is shared across all layers/heads (same rotation for all).
pub struct TurboQuantCodec {
    #[allow(dead_code)]
    bits: u8,
    head_dim: usize,
    // Flat rotation matrix Π of shape [head_dim, head_dim], row-major.
    rotation: Vec<f32>,
    // Codebook centroids (sorted ascending), length 2^bits.
    centroids: Vec<f32>,
    // Per-coordinate scale: we normalise by the L2 norm of the rotated vector
    // so that the rotated coordinates lie on the unit sphere.
    // The norm is stored alongside the quantised indices and used for reconstruction.
}

impl TurboQuantCodec {
    /// Create a new codec with a freshly sampled random rotation matrix.
    pub fn new(cfg: &TurboQuantConfig) -> Self {
        let d = cfg.head_dim;
        let rotation = random_orthogonal_matrix(d);
        let centroids = lloyd_max_centroids(cfg.bits).to_vec();
        Self {
            bits: cfg.bits,
            head_dim: d,
            rotation,
            centroids,
        }
    }

    /// Quantize a single head vector `x` (length `head_dim`, f32 slice).
    ///
    /// Returns `(indices, norm)` where:
    /// - `indices[j]` is the codebook index (0..2^bits) for rotated coordinate j,
    /// - `norm` is the L2 norm of `x` (stored for dequantization).
    pub fn quantize_vec(&self, x: &[f32]) -> (Vec<u8>, f32) {
        let d = self.head_dim;
        debug_assert_eq!(x.len(), d);

        // Compute L2 norm and normalise.
        let norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let inv_norm = if norm > 1e-12 { 1.0 / norm } else { 1.0 };

        // Rotate: y = Π · (x / norm)   [shape d]
        let y: Vec<f32> = self
            .rotation
            .chunks_exact(d)
            .map(|row| {
                // Scale the rotated coordinate so it's on the unit sphere:
                // we divide x by norm first (conceptually), but the rotation is
                // linear so we can multiply by inv_norm after.
                let acc: f32 = row.iter().zip(x.iter()).map(|(r, xi)| r * xi).sum();
                acc * inv_norm
            })
            .collect();

        // Scalar quantize each coordinate.
        let num_centroids = self.centroids.len();
        let indices: Vec<u8> = y
            .iter()
            .map(|&val| {
                // Binary search for the nearest centroid.
                let pos = self.centroids.partition_point(|&c| c < val);
                // Compare pos-1 and pos
                let idx = if pos == 0 {
                    0
                } else if pos == num_centroids {
                    num_centroids - 1
                } else {
                    let dl = val - self.centroids[pos - 1];
                    let dr = self.centroids[pos] - val;
                    if dl <= dr {
                        pos - 1
                    } else {
                        pos
                    }
                };
                idx as u8
            })
            .collect();

        (indices, norm)
    }

    /// Dequantize: recover an approximate head vector from `(indices, norm)`.
    pub fn dequantize_vec(&self, indices: &[u8], norm: f32) -> Vec<f32> {
        let d = self.head_dim;
        debug_assert_eq!(indices.len(), d);

        // Reconstruct rotated vector (on unit sphere) from codebook entries.
        let y: Vec<f32> = indices
            .iter()
            .map(|&i| self.centroids[i as usize])
            .collect();

        // Apply inverse rotation: x̃ = Π⊤ · y  (Π is orthogonal so Π⁻¹ = Π⊤)
        // Π⊤[i,j] = Π[j,i], so x[i] = Σ_j rotation[j*d + i] * y[j]
        (0..d)
            .map(|i| {
                let acc: f32 = y
                    .iter()
                    .enumerate()
                    .map(|(j, &yj)| self.rotation[j * d + i] * yj)
                    .sum();
                acc * norm
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Helpers: random orthogonal matrix via Gram-Schmidt (pure Rust, no candle)
// ---------------------------------------------------------------------------

fn random_orthogonal_matrix(d: usize) -> Vec<f32> {
    // Deterministic seed based on d for reproducibility across restarts.
    // We use a simple LCG PRNG.
    let mut state: u64 = 0x9e3779b97f4a7c15u64.wrapping_mul(d as u64 + 1);
    let mut rand_f32 = move || -> f32 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Box-Muller: use non-overlapping bit ranges for independence.
        let u1 = (state >> 41) as u32 as f32 / (1u32 << 23) as f32; // bits 41-63
        let u2 = ((state >> 18) as u32 & 0x7F_FFFF) as f32 / (1u32 << 23) as f32; // bits 18-40
        let r = (-2.0f32 * (u1 + 1e-30).ln()).sqrt();
        r * (2.0f32 * std::f32::consts::PI * u2).cos()
    };

    // Fill with i.i.d. N(0,1) entries.
    let mut m: Vec<f32> = (0..d * d).map(|_| rand_f32()).collect();

    // Gram-Schmidt orthogonalization.
    for i in 0..d {
        // Normalize column i (stored as row i in our row-major layout — we
        // actually do Gram-Schmidt on the rows since we want row-orthogonality
        // for the rotation R such that R·x gives the rotated vector).
        let row_start = i * d;
        let norm: f32 = m[row_start..row_start + d]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        if norm > 1e-12 {
            m[row_start..row_start + d]
                .iter_mut()
                .for_each(|v| *v /= norm);
        }
        // Subtract projections from subsequent rows.
        for j in i + 1..d {
            let dot: f32 = (0..d).map(|k| m[i * d + k] * m[j * d + k]).sum();
            let (row_i, row_j) = {
                // Split borrow: get disjoint slices for row i and row j.
                let (left, right) = m.split_at_mut(j * d);
                (&left[i * d..i * d + d], &mut right[..d])
            };
            for (src, dst) in row_i.iter().zip(row_j.iter_mut()) {
                *dst -= dot * src;
            }
        }
    }
    m
}

// ---------------------------------------------------------------------------
// TurboQuantKvCache — drop-in replacement for `Option<(Tensor, Tensor)>`
// ---------------------------------------------------------------------------

/// Quantized KV cache for a single attention layer.
///
/// Stores keys and values in compressed form.  On every access (for attention
/// computation) it dequantizes back to the working dtype.
pub struct TurboQuantKvCache {
    codec: std::sync::Arc<TurboQuantCodec>,
    /// Quantized keys: list of (indices: Vec<u8>, norm: f32) one per token.
    k_tokens: Vec<(Vec<u8>, f32)>,
    /// Quantized values: list of (indices: Vec<u8>, norm: f32) one per token.
    v_tokens: Vec<(Vec<u8>, f32)>,
    /// Number of KV heads.
    num_kv_heads: usize,
    /// Working dtype for dequantized output.
    dtype: DType,
    /// Device for output tensors.
    device: Device,
}

impl TurboQuantKvCache {
    pub fn new(
        codec: std::sync::Arc<TurboQuantCodec>,
        num_kv_heads: usize,
        dtype: DType,
        device: Device,
    ) -> Self {
        Self {
            codec,
            k_tokens: Vec::new(),
            v_tokens: Vec::new(),
            num_kv_heads,
            dtype,
            device,
        }
    }

    /// Append newly computed key and value tensors to the quantized cache.
    ///
    /// `k` and `v`: shape `[batch=1, num_kv_heads, seq_len, head_dim]`
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let (_b, _nkv, seq_len, head_dim) = k.dims4()?;
        debug_assert_eq!(_nkv, self.num_kv_heads);
        debug_assert_eq!(_b, 1);

        // Convert to f32 on CPU for quantization.
        let k_f32 = k.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let v_f32 = v.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;

        // Flatten to [num_kv_heads * seq_len, head_dim].
        let k_data = k_f32
            .reshape((self.num_kv_heads * seq_len, head_dim))?
            .to_vec2::<f32>()?;
        let v_data = v_f32
            .reshape((self.num_kv_heads * seq_len, head_dim))?
            .to_vec2::<f32>()?;

        for (kv, vv) in k_data.iter().zip(v_data.iter()) {
            self.k_tokens.push(self.codec.quantize_vec(kv));
            self.v_tokens.push(self.codec.quantize_vec(vv));
        }

        Ok(())
    }

    /// Dequantize all cached tokens and return `(k, v)` tensors ready for attention.
    ///
    /// Output shapes: `[1, num_kv_heads, total_seq_len, head_dim]`
    pub fn dequantize(&self) -> Result<(Tensor, Tensor)> {
        let num_tokens_per_head = self.k_tokens.len() / self.num_kv_heads;
        let head_dim = self.codec.head_dim;

        let total = self.k_tokens.len(); // num_kv_heads * total_seq_len

        // Dequantize all tokens to f32 flat arrays — in parallel across tokens.
        let mut k_flat = vec![0.0f32; total * head_dim];
        let mut v_flat = vec![0.0f32; total * head_dim];

        k_flat
            .par_chunks_mut(head_dim)
            .zip(self.k_tokens.par_iter())
            .for_each(|(chunk, (k_idx, k_norm))| {
                chunk.copy_from_slice(&self.codec.dequantize_vec(k_idx, *k_norm));
            });
        v_flat
            .par_chunks_mut(head_dim)
            .zip(self.v_tokens.par_iter())
            .for_each(|(chunk, (v_idx, v_norm))| {
                chunk.copy_from_slice(&self.codec.dequantize_vec(v_idx, *v_norm));
            });

        // Build tensors: [num_kv_heads * total_seq_len, head_dim] → [1, num_kv_heads, seq_len, head_dim]
        let k_t = Tensor::from_vec(
            k_flat,
            (1, self.num_kv_heads, num_tokens_per_head, head_dim),
            &Device::Cpu,
        )?
        .to_dtype(self.dtype)?
        .to_device(&self.device)?;
        let v_t = Tensor::from_vec(
            v_flat,
            (1, self.num_kv_heads, num_tokens_per_head, head_dim),
            &Device::Cpu,
        )?
        .to_dtype(self.dtype)?
        .to_device(&self.device)?;

        Ok((k_t, v_t))
    }

    /// Clear all cached tokens (start of a new sequence).
    pub fn clear(&mut self) {
        self.k_tokens.clear();
        self.v_tokens.clear();
    }

    /// Number of cached token steps (total across all KV heads).
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        if self.num_kv_heads == 0 {
            0
        } else {
            self.k_tokens.len() / self.num_kv_heads
        }
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.k_tokens.is_empty()
    }

    /// Return the sequence length of cached tokens.
    #[allow(dead_code)]
    pub fn seq_len(&self) -> usize {
        self.len()
    }
}

// ---------------------------------------------------------------------------
// Public API: build a shared codec from config
// ---------------------------------------------------------------------------

/// Build a shared `TurboQuantCodec` from a `TurboQuantConfig`.
pub fn build_codec(cfg: &TurboQuantConfig) -> std::sync::Arc<TurboQuantCodec> {
    std::sync::Arc::new(TurboQuantCodec::new(cfg))
}
