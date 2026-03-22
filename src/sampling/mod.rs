//! Token sampling with temperature, top-p, top-k, and repetition penalty.

use anyhow::Result;
use candle_core::Tensor;
use serde::{Deserialize, Serialize};

/// Per-request sampling parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub max_tokens: usize,
    pub repetition_penalty: f64,
    /// Stop token IDs.
    pub stop_token_ids: Vec<u32>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            max_tokens: 2048,
            repetition_penalty: 1.0,
            stop_token_ids: vec![],
        }
    }
}

/// Sample a token from logits using the given parameters.
///
/// `logits` shape: (vocab_size,)
/// `past_tokens`: tokens generated so far (for repetition penalty)
pub fn sample_token(logits: &Tensor, params: &SamplingParams, past_tokens: &[u32]) -> Result<u32> {
    let logits = logits.to_dtype(candle_core::DType::F32)?.to_vec1::<f32>()?;
    let mut logits = logits;

    // Apply repetition penalty
    if params.repetition_penalty != 1.0 {
        apply_repetition_penalty(&mut logits, past_tokens, params.repetition_penalty);
    }

    // Temperature
    if params.temperature <= 0.0 || params.temperature < 1e-6 {
        // Greedy
        let token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);
        return Ok(token);
    }

    // Scale by temperature
    for l in logits.iter_mut() {
        *l /= params.temperature as f32;
    }

    // Top-k filtering
    if params.top_k > 0 && params.top_k < logits.len() {
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let threshold = indexed[params.top_k].1;
        for l in logits.iter_mut() {
            if *l < threshold {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    // Softmax
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|l| (l - max_logit).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }

    // Top-p (nucleus) filtering
    if params.top_p < 1.0 {
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut cumulative = 0.0f32;
        let mut cutoff_idx = indexed.len();
        for (i, (_, p)) in indexed.iter().enumerate() {
            cumulative += p;
            if cumulative > params.top_p as f32 {
                cutoff_idx = i + 1;
                break;
            }
        }
        // Zero out everything below cutoff
        let kept: std::collections::HashSet<usize> =
            indexed[..cutoff_idx].iter().map(|(idx, _)| *idx).collect();
        for (i, p) in probs.iter_mut().enumerate() {
            if !kept.contains(&i) {
                *p = 0.0;
            }
        }
        // Re-normalize
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in probs.iter_mut() {
                *p /= sum;
            }
        }
    }

    // Sample from the distribution
    let token = weighted_sample(&probs);
    Ok(token)
}

/// Apply repetition penalty to logits.
fn apply_repetition_penalty(logits: &mut [f32], past_tokens: &[u32], penalty: f64) {
    let penalty = penalty as f32;
    for &token in past_tokens {
        let idx = token as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

/// Weighted random sampling from a probability distribution.
fn weighted_sample(probs: &[f32]) -> u32 {
    // Simple PRNG using thread-local state for speed
    let r: f32 = fast_random();
    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i as u32;
        }
    }
    // Fallback to last token
    (probs.len() - 1) as u32
}

/// Fast random float in [0, 1) using a simple xorshift.
fn fast_random() -> f32 {
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u64> = Cell::new(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64
        );
    }
    STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        (x as f32) / (u64::MAX as f32)
    })
}
