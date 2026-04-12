//! Chunk-based GatedDeltaNet parallel scan for prefill.
//!
//! Replaces the O(T) sequential recurrence with the HuggingFace-proven
//! chunked WY (Woodbury) approach: O(T/chunk_size) sequential steps.
//!
//! Reference: transformers/models/qwen3_5/modeling_qwen3_5.py
//!            `torch_chunk_gated_delta_rule`

use anyhow::Result;
use candle_core::{DType, Tensor};

const CHUNK_SIZE: usize = 64;

/// Pure-Candle chunked GatedDeltaNet for prefill (t > 1).
///
/// Inputs are all F32, shapes:
///   q:    [b, t, n_heads, head_k_dim]
///   k:    [b, t, n_heads, head_k_dim]
///   v:    [b, t, n_heads, head_v_dim]
///   g:    [b, t, n_heads]           (decay, already exp'd)
///   beta: [b, t, n_heads]
///   state: [b, n_heads, head_k_dim, head_v_dim]  (mutable, updated in-place)
///
/// Returns: [b, t, n_heads, head_v_dim]
pub fn gated_delta_rule_chunked(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    let device = q.device().clone();
    let (b, t, n_heads, head_k_dim) = q.dims4()?;
    let head_v_dim = v.dim(3)?;

    let chunk = CHUNK_SIZE;
    let num_chunks = t.div_ceil(chunk);
    let pad_t = num_chunks * chunk;
    let needs_pad = pad_t != t;
    // Flat batch size used to collapse [b, n_h, C] into one dim for 3D matmul.
    // CUDA gemm_config only handles ≤2 batch-prefix dims; 5D tensors with 3 batch
    // dims fall through to MatMulNonContiguous. Reshaping to [bhnc, S, d] avoids this.
    let bhnc = b * n_heads * num_chunks;

    tracing::trace!(
        "gated_delta_rule_chunked: t={} chunk={} num_chunks={}",
        t,
        chunk,
        num_chunks
    );

    let log_g = g.log()?;

    // Reshape [b, t, n_h, d] -> [b, n_h, num_chunks, chunk, d] with padding
    let reshape_4d = |tensor: &Tensor, d: usize| -> Result<Tensor> {
        let padded = if needs_pad {
            let zeros = Tensor::zeros((b, pad_t - t, n_heads, d), DType::F32, &device)?;
            Tensor::cat(&[tensor, &zeros], 1)?
        } else {
            tensor.clone()
        };
        Ok(padded
            .permute((0, 2, 1, 3))?
            .contiguous()?
            .reshape((b, n_heads, num_chunks, chunk, d))?)
    };

    let reshape_3d = |tensor: &Tensor| -> Result<Tensor> {
        let padded = if needs_pad {
            let zeros = Tensor::zeros((b, pad_t - t, n_heads), DType::F32, &device)?;
            Tensor::cat(&[tensor, &zeros], 1)?
        } else {
            tensor.clone()
        };
        Ok(padded
            .permute((0, 2, 1))?
            .contiguous()?
            .reshape((b, n_heads, num_chunks, chunk))?)
    };

    let q_c = reshape_4d(q, head_k_dim)?; // [b, n_h, C, S, hk]
    let k_c = reshape_4d(k, head_k_dim)?; // [b, n_h, C, S, hk]
    let v_c = reshape_4d(v, head_v_dim)?; // [b, n_h, C, S, hv]
    let log_g_c = reshape_3d(&log_g)?; // [b, n_h, C, S]
    let beta_c = reshape_3d(beta)?; // [b, n_h, C, S]

    // ── Step 3a: Log-decay cumsum + decay mask ────────────────────────────
    // g_cumsum[i] = sum(log_g[0..i+1]) within each chunk
    let g_cumsum = log_g_c.cumsum(candle_core::D::Minus1)?; // [b, n_h, C, S]

    // decay_mask[i,j] = exp(g_cumsum[i] - g_cumsum[j]) for i > j, else 0
    // tril_strict: strictly lower triangular (diagonal = 0) — position i must not
    // read its own write in the same step, so A[i,i] = 0 by definition.
    let tril = Tensor::tril2(chunk, DType::F32, &device)?; // [S, S]
    let tril_strict = (&tril - &Tensor::eye(chunk, DType::F32, &device)?)?; // diagonal zeroed
                                                                            // Materialize both expansion directions before sub: broadcast_sub on CUDA
                                                                            // doesn't work when both sides need different-dimension expansion simultaneously.
    let outer = (b, n_heads, num_chunks, chunk, chunk);
    let gc_i = g_cumsum
        .unsqueeze(candle_core::D::Minus1)?
        .broadcast_as(outer)?
        .contiguous()?; // [b, n_h, C, S, S] — each row i repeated S times
    let gc_j = g_cumsum
        .unsqueeze(candle_core::D::Minus2)?
        .broadcast_as(outer)?
        .contiguous()?; // [b, n_h, C, S, S] — each col j repeated S times
    let decay_diff = (&gc_i - &gc_j)?.exp()?;
    // decay_mask_strict: used for (I+A) solve — diagonal must be zero
    let decay_mask = decay_diff.broadcast_mul(&tril_strict)?;
    // decay_mask_full: used for output a_intra — diagonal = 1 (position reads own write)
    let decay_mask_full = decay_diff.broadcast_mul(&tril)?;

    // ── Step 3b: Weighted keys/values ─────────────────────────────────────
    let beta_unsq = beta_c.unsqueeze(candle_core::D::Minus1)?; // [b, n_h, C, S, 1]
    let k_beta = k_c.broadcast_mul(&beta_unsq)?; // [b, n_h, C, S, hk]
    let v_beta = v_c.broadcast_mul(&beta_unsq)?; // [b, n_h, C, S, hv]

    // ── Step 3c: Lower-triangular A = -k_beta @ k^T * decay_mask ──────────
    // k_beta: [b, n_h, C, S, hk], k_c: [b, n_h, C, S, hk]
    // Reshape to 3D [bhnc, S, hk] so CUDA gemm_config sees exactly 1 batch dim.
    let kk = k_beta
        .reshape((bhnc, chunk, head_k_dim))?
        .broadcast_matmul(
            &k_c.reshape((bhnc, chunk, head_k_dim))?
                .transpose(candle_core::D::Minus1, candle_core::D::Minus2)?,
        )?
        .reshape((b, n_heads, num_chunks, chunk, chunk))?;
    // kk: [b, n_h, C, S, S] — this is k_beta @ k^T
    let a_mat: Tensor = kk.broadcast_mul(&decay_mask)?.neg()?;

    // ── Step 3d: Solve (I + A) via forward substitution ───────────────────
    // attn starts as identity + lower-triangular part solved row by row.
    // Row 0: attn[0] = e_0 (identity, since A[0,:] = 0 for strictly lower tri)
    // Row i: attn[i,j] = A[i,j] + sum_k(attn[k,j] * A[i,k]) for k < i, j < i
    //
    // We build attn as [b, n_h, C, S, S], init with identity.
    let identity = Tensor::eye(chunk, DType::F32, &device)?.reshape((1, 1, 1, chunk, chunk))?;
    let identity = identity.broadcast_as((b, n_heads, num_chunks, chunk, chunk))?;
    let mut attn = identity;

    // Forward substitution: solve (I − a_mat) * attn = I  →  attn = (I − a_mat)^{-1}
    //
    // Derivation: unrolling the recurrence gives u_j = rhs_j + Σ_{k<j} a_mat[j,k] u_k
    // i.e. (I − a_mat) u = rhs, so attn = (I − a_mat)^{-1}.
    //
    // Row i recurrence: attn[i, :] = e_i + sum_{k<i} a_mat[i, k] * attn[k, :]
    // (a_mat entries are negative, so this slightly shrinks each row away from the identity)
    //
    // Row 0 is already e_0 from the identity initialisation.
    for i in 1..chunk {
        // attn[0..i, :] — already-solved rows
        let attn_sub = attn.narrow(candle_core::D::Minus2, 0, i)?; // [b, n_h, C, i, S]

        // a_mat[i, 0:i] — row i of A, first i sub-diagonal entries
        let a_sub = a_mat
            .narrow(candle_core::D::Minus2, i, 1)? // row i:    [b, n_h, C, 1, S]
            .narrow(candle_core::D::Minus1, 0, i)?; // cols 0..i: [b, n_h, C, 1, i]

        // contrib = a_mat[i, 0:i] @ attn[0:i, :] — [b,n_h,C,1,i] @ [b,n_h,C,i,S] → [b,n_h,C,1,S]
        // Reshape to 3D: CUDA gemm_config only handles ≤2 batch-prefix dims.
        let contrib = a_sub
            .reshape((bhnc, 1, i))?
            .broadcast_matmul(&attn_sub.reshape((bhnc, i, chunk))?)?
            .reshape((b, n_heads, num_chunks, 1, chunk))?;

        // new_row = e_i + contrib  (attn[i, :] starts as e_i from the identity)
        // Solving (I − a_mat) X = I: X[i,:] = e_i + Σ_{k<i} a_mat[i,k] X[k,:]
        // (a_mat has negative entries, so this slightly shrinks each row)
        let cur_row = attn.narrow(candle_core::D::Minus2, i, 1)?; // [b, n_h, C, 1, S] = e_i
        let new_row = (&cur_row + &contrib)?; // [b, n_h, C, 1, S]

        // Splice new_row into attn at position i
        let top = attn.narrow(candle_core::D::Minus2, 0, i)?;
        let bottom = if i + 1 < chunk {
            attn.narrow(candle_core::D::Minus2, i + 1, chunk - i - 1)?
        } else {
            Tensor::zeros((b, n_heads, num_chunks, 0, chunk), DType::F32, &device)?
        };
        attn = Tensor::cat(&[top, new_row, bottom], candle_core::D::Minus2)?;
    }

    // ── Step 3e: Apply WY representation ──────────────────────────────────
    // value_new = attn @ v_beta — reshape to 3D for CUDA gemm_config
    let value_new = attn
        .reshape((bhnc, chunk, chunk))?
        .broadcast_matmul(&v_beta.reshape((bhnc, chunk, head_v_dim))?)?
        .reshape((b, n_heads, num_chunks, chunk, head_v_dim))?; // [b, n_h, C, S, hv]

    // w = attn @ (k_beta * exp(g_cumsum))
    let g_exp = g_cumsum.exp()?.unsqueeze(candle_core::D::Minus1)?; // [b, n_h, C, S, 1]
    let k_beta_scaled = k_beta.broadcast_mul(&g_exp)?; // [b, n_h, C, S, hk]
                                                       // w_ci @ state gives the intra-chunk state correction.
    let w = attn
        .reshape((bhnc, chunk, chunk))?
        .broadcast_matmul(&k_beta_scaled.reshape((bhnc, chunk, head_k_dim))?)?
        .reshape((b, n_heads, num_chunks, chunk, head_k_dim))?; // [b, n_h, C, S, hk]

    // ── Step 4: Inter-chunk state propagation ─────────────────────────────
    let mut outputs = Vec::with_capacity(num_chunks);
    let mut s = state.clone(); // [b, n_heads, hk, hv]

    for ci in 0..num_chunks {
        let q_ci = q_c.narrow(2, ci, 1)?.squeeze(2)?; // [b, n_h, S, hk]
        let k_ci = k_c.narrow(2, ci, 1)?.squeeze(2)?; // [b, n_h, S, hk]
        let v_new_ci = value_new.narrow(2, ci, 1)?.squeeze(2)?; // [b, n_h, S, hv]
        let w_ci = w.narrow(2, ci, 1)?.squeeze(2)?; // [b, n_h, S, hk]
                                                    // decay_ci_full: uses tril (diagonal=1) so position i reads its own write
        let decay_ci_full = decay_mask_full.narrow(2, ci, 1)?.squeeze(2)?; // [b, n_h, S, S]
        let gc_ci = g_cumsum.narrow(2, ci, 1)?.squeeze(2)?; // [b, n_h, S]

        // v_prime = w_ci @ state: [b, n_h, S, hk] @ [b, n_h, hk, hv] -> [b, n_h, S, hv]
        let v_prime = w_ci.broadcast_matmul(&s)?;

        // v_new_corrected = value_new - v_prime
        let v_corrected = (&v_new_ci - &v_prime)?;

        // Intra-chunk attention scores: q @ k^T * decay_mask_full (tril, not tril_strict)
        // Diagonal = 1: position i attends to its own write (consistent with sequential step).
        let a_intra = q_ci
            .broadcast_matmul(&k_ci.transpose(candle_core::D::Minus1, candle_core::D::Minus2)?)?
            .broadcast_mul(&decay_ci_full)?; // [b, n_h, S, S]

        // Inter-chunk: (q * exp(g_cumsum)) @ state
        let gc_exp = gc_ci.exp()?.unsqueeze(candle_core::D::Minus1)?; // [b, n_h, S, 1]
        let q_scaled = q_ci.broadcast_mul(&gc_exp)?; // [b, n_h, S, hk]
        let attn_inter = q_scaled.broadcast_matmul(&s)?; // [b, n_h, S, hv]

        // output = attn_inter + A_intra @ v_corrected
        let out_ci = (attn_inter + a_intra.broadcast_matmul(&v_corrected)?)?; // [b, n_h, S, hv]
        outputs.push(out_ci.unsqueeze(2)?); // [b, n_h, 1, S, hv]

        // State update: compute decay from chunk start to chunk end
        let g_end = gc_ci
            .narrow(candle_core::D::Minus1, chunk - 1, 1)?
            .squeeze(candle_core::D::Minus1)?; // [b, n_h]
        let g_end_exp = g_end
            .exp()?
            .unsqueeze(candle_core::D::Minus1)?
            .unsqueeze(candle_core::D::Minus1)?; // [b, n_h, 1, 1]

        // decay_to_end[j] = exp(g_cumsum[S-1] - g_cumsum[j])
        let gc_last = gc_ci.narrow(candle_core::D::Minus1, chunk - 1, 1)?; // [b, n_h, 1]
        let decay_to_end = gc_last
            .broadcast_sub(&gc_ci)?
            .exp()?
            .unsqueeze(candle_core::D::Minus1)?; // [b, n_h, S, 1]

        // state = state * exp(g_cumsum[-1]) + (k * decay_to_end)^T @ v_corrected
        let k_weighted = k_ci.broadcast_mul(&decay_to_end)?; // [b, n_h, S, hk]
        let state_update = k_weighted
            .transpose(candle_core::D::Minus1, candle_core::D::Minus2)?
            .broadcast_matmul(&v_corrected)?; // [b, n_h, hk, hv]
        s = (s.broadcast_mul(&g_end_exp)? + state_update)?;
    }

    // Save state
    *state = s.detach();

    // ── Step 5: Truncate padding + reshape output ─────────────────────────
    let out_all = Tensor::cat(&outputs, 2)?; // [b, n_h, C, S, hv]
                                             // Permute back to [b, C*S, n_h, hv] = [b, pad_t, n_h, hv]
    let out_perm = out_all
        .permute((0, 2, 3, 1, 4))?
        .contiguous()?
        .reshape((b, pad_t, n_heads, head_v_dim))?;
    // Truncate padding
    let out = out_perm.narrow(1, 0, t)?;
    Ok(out)
}

/// Reference sequential loop for t>1 (used in tests to verify chunked output).
/// Inputs: q/k/v [b, t, n_h, d], g/beta [b, t, n_h]. State: [b, n_h, hk, hv].
/// Returns [b, t, n_h, hv].
#[cfg(test)]
fn sequential_loop(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    let (b, t, n_heads, _) = q.dims4()?;
    let head_v_dim = v.dim(3)?;
    // Permute to [b, n_h, t, d] so we can iterate over the time axis
    let q_p = q.permute((0, 2, 1, 3))?.contiguous()?; // [b, n_h, t, hk]
    let k_p = k.permute((0, 2, 1, 3))?.contiguous()?;
    let v_p = v.permute((0, 2, 1, 3))?.contiguous()?;
    let g_p = g.permute((0, 2, 1))?.contiguous()?; // [b, n_h, t]
    let b_p = beta.permute((0, 2, 1))?.contiguous()?;
    let mut outputs = Vec::with_capacity(t);
    for i in 0..t {
        let q_t = q_p.narrow(2, i, 1)?.squeeze(2)?; // [b, n_h, hk]
        let k_t = k_p.narrow(2, i, 1)?.squeeze(2)?;
        let v_t = v_p.narrow(2, i, 1)?.squeeze(2)?;
        let g_t = g_p.narrow(2, i, 1)?.squeeze(2)?; // [b, n_h]
        let beta_t = b_p.narrow(2, i, 1)?.squeeze(2)?;
        let out = sequential_step(&q_t, &k_t, &v_t, &g_t, &beta_t, state)?; // [b, n_h, hv]
        outputs.push(out.unsqueeze(2)?); // [b, n_h, 1, hv]
    }
    // cat along time → [b, n_h, t, hv], then permute to [b, t, n_h, hv]
    let out_all = Tensor::cat(&outputs, 2)?;
    out_all
        .permute((0, 2, 1, 3))?
        .contiguous()
        .map_err(Into::into)
}

/// Sequential single-step decode (t=1). Extracted from the original loop.
///
/// Inputs are all F32:
///   q_t:  [b, n_heads, head_k_dim]
///   k_t:  [b, n_heads, head_k_dim]
///   v_t:  [b, n_heads, head_v_dim]
///   g_t:  [b, n_heads]
///   beta_t: [b, n_heads]
///   state: [b, n_heads, head_k_dim, head_v_dim]
///
/// Returns: [b, n_heads, head_v_dim]
pub fn sequential_step(
    q_t: &Tensor,
    k_t: &Tensor,
    v_t: &Tensor,
    g_t: &Tensor,
    beta_t: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    // Decay
    *state = state.broadcast_mul(&g_t.unsqueeze(2)?.unsqueeze(3)?)?;

    // Read: kv_mem = (state * k_t[:,:,None,:]).sum(-2)
    let kv_mem = (state.broadcast_mul(&k_t.unsqueeze(3)?)?).sum(candle_core::D::Minus2)?;

    // Delta
    let diff = (v_t - &kv_mem)?;
    let delta = diff.broadcast_mul(&beta_t.unsqueeze(2)?)?;

    // Write: state += k_t[:,:,:,None] * delta[:,:,None,:]
    *state = (&*state + k_t.unsqueeze(3)?.broadcast_mul(&delta.unsqueeze(2)?)?)?;

    // Read output
    let out = (state.broadcast_mul(&q_t.unsqueeze(3)?)?).sum(candle_core::D::Minus2)?;

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let diff = (a - b).unwrap().abs().unwrap();
        diff.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .cloned()
            .fold(0.0f32, f32::max)
    }

    fn run_chunked_vs_sequential(device: Device) {
        let b = 1usize;
        let t = 128usize; // two full chunks
        let n_heads = 2usize;
        let hk = 4usize;
        let hv = 4usize;

        // Use fixed seeds by constructing simple tensors
        let q = Tensor::randn(0f32, 0.1f32, (b, t, n_heads, hk), &device).unwrap();
        let k = Tensor::randn(0f32, 0.1f32, (b, t, n_heads, hk), &device).unwrap();
        let v = Tensor::randn(0f32, 1.0f32, (b, t, n_heads, hv), &device).unwrap();
        // g must be in (0,1): use sigmoid of randn * 0.5 offset to keep away from 0/1
        let g_raw = Tensor::randn(-2.0f32, 1.0f32, (b, t, n_heads), &device).unwrap();
        let g = candle_nn::ops::sigmoid(&g_raw).unwrap();
        let beta_raw = Tensor::randn(0f32, 1.0f32, (b, t, n_heads), &device).unwrap();
        let beta = candle_nn::ops::sigmoid(&beta_raw).unwrap();

        let mut state_seq = Tensor::zeros((b, n_heads, hk, hv), DType::F32, &device).unwrap();
        let mut state_chk = Tensor::zeros((b, n_heads, hk, hv), DType::F32, &device).unwrap();

        // Permute q,k,v from [b,t,n_h,d] to sequential_loop's expected format
        let out_seq = sequential_loop(&q, &k, &v, &g, &beta, &mut state_seq).unwrap();
        let out_chk = gated_delta_rule_chunked(&q, &k, &v, &g, &beta, &mut state_chk).unwrap();

        let out_diff = max_abs_diff(&out_seq, &out_chk);
        let state_diff = max_abs_diff(&state_seq, &state_chk);

        println!("output max abs diff = {out_diff:.6}");
        println!("state  max abs diff = {state_diff:.6}");

        assert!(out_diff < 1e-4, "output mismatch: max diff = {out_diff}");
        assert!(state_diff < 1e-4, "state mismatch: max diff = {state_diff}");
    }

    #[test]
    fn chunked_matches_sequential_cpu() {
        run_chunked_vs_sequential(Device::Cpu);
    }

    #[test]
    fn chunked_matches_sequential_cuda() {
        let device = Device::cuda_if_available(0).expect("cuda device");
        if matches!(device, Device::Cpu) {
            eprintln!("CUDA not available, skipping GPU test");
            return;
        }
        run_chunked_vs_sequential(device);
    }
}
