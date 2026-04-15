// Monolithic CUDA kernel for the GatedDeltaNet chunked scan (prefill).
//
// Replaces ~15 per-chunk candle kernel launches with a single kernel that
// processes all chunks sequentially, keeping state in global memory and
// intra-chunk buffers in shared memory.
//
// Grid:  (B * N_HEADS, 1, 1)  — one block per (batch, head) pair
// Block: (256, 1, 1)
//
// Shared memory layout (floats, static):
//   s_attn  [S*S]    16 KB   — (I − a_mat)^{-1} after forward substitution
//   s_a_row [S]     256 B   — scratch for one a_mat row during fwd subst
//   s_log_g [S]     256 B
//   s_gcsum [S]     256 B
//   s_kbeta [S*HK]  up to 32 KB  — k*beta, read-only after load (always F32)
//   s_vcorr [S*HV]  up to 32 KB  — v_delta (step 4), then output (step 6)
//
// Total for HK=HV=64:  ~49 KB   (needs cudaFuncAttributeMaxDynamicSharedMemorySize=96KB)
// Total for HK=HV=128: ~81 KB   (same opt-in)
//
// Template parameter T: dtype of q, k, v inputs (float or __nv_bfloat16).
// log_g, beta, state, out are always float.
// All internal accumulators are float (F32) regardless of T.
//
// Algorithm order per chunk:
//   1. Load log_g → compute g_cumsum (inclusive prefix sum)
//   2. Load k_beta into s_kbeta (cast to F32)
//   3. Forward substitution → s_attn = (I − a_mat)^{-1}
//   4. Compute v_delta[s2,hv] = beta[s2]*(v[s2,hv] − exp(gc[s2])*Σ_hk kbeta[s2,hk]*state[hk,hv])
//      into s_vcorr; s_kbeta becomes free.
//   5. v_corrected[s1,hv] = Σ_{s2} attn[s1,s2]*v_delta[s2,hv] → write to out_c (global)
//      s_attn and s_vcorr become free.
//   6. Output[s1,hv] = inter[s1,hv] + intra[s1,hv] → s_vcorr
//      inter: q_exp[s1] @ state;  intra: Σ_{s2≤s1} decay*dot(q[s1],k[s2])*v_corr[s2]
//   7. State update: state *= g_end + k_weighted^T @ v_corrected  (reads out_c)
//   8. Copy s_vcorr → out_c  (final chunk output)
//
// Numerical invariants (agentic_doc/qwen35-cuda-correctness.md):
//   • j < i in forward subst: exp(gc[i]-gc[j]) where gc non-decreasing (log_g ≤ 0 typically)
//   • decay_full[i,j] = exp(gc[i]-gc[j]) for j ≤ i: exponent ≤ 0, no overflow
//   • log_g pre-computed by caller; no log() call inside this kernel

#include <stdint.h>
#include <float.h>
#include <cuda_bf16.h>

// ── Type helpers ─────────────────────────────────────────────────────────────

// Scalar load of T → float.
template<typename T>
__device__ __forceinline__ float load_as_f32(const T* ptr, int i);

template<>
__device__ __forceinline__ float load_as_f32<float>(const float* ptr, int i) {
    return ptr[i];
}

template<>
__device__ __forceinline__ float load_as_f32<__nv_bfloat16>(const __nv_bfloat16* ptr, int i) {
    return __bfloat162float(ptr[i]);
}

// Vectorised 2-element load of T → float2.
// Caller must ensure `i` is even so that &ptr[i] is 4-byte aligned.
// (Safe for all our uses: HK ∈ {64,128}, loops stride by 2.)
template<typename T>
__device__ __forceinline__ float2 load2_as_f32(const T* ptr, int i);

template<>
__device__ __forceinline__ float2 load2_as_f32<float>(const float* ptr, int i) {
    return make_float2(ptr[i], ptr[i + 1]);
}

template<>
__device__ __forceinline__ float2 load2_as_f32<__nv_bfloat16>(const __nv_bfloat16* ptr, int i) {
    // &ptr[i] is 4-byte aligned when i is even — holds for HK ∈ {64,128}, hk += 2.
    return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ptr[i]));
}

// ── Helpers ───────────────────────────────────────────────────────────────────

// Block-wide inclusive prefix sum of smem[0..S).
// ALL threads in the block must call this — no `if (tid < S)` guard at the call site.
// Threads with tid >= S participate in barriers but do not modify smem.
__device__ __forceinline__ void prefix_sum_inplace(float* smem, int tid, int S) {
    float v = (tid < S) ? smem[tid] : 0.0f;
    __syncthreads();
    for (int step = 1; step < S; step <<= 1) {
        float prev = (tid >= step && tid < S) ? smem[tid - step] : 0.0f;
        __syncthreads();
        v += prev;
        if (tid < S) smem[tid] = v;
        __syncthreads();
    }
}

// ── Main kernel ───────────────────────────────────────────────────────────────
//
// q, k, v have shape [B*NH, C, S, dim], contiguous, dtype T.
// log_g, beta have shape [B*NH, C, S], contiguous, float.
// state has shape [B*NH, HK, HV], float.
// out   has shape [B*NH, C, S, HV], float.

template<int HK, int HV, int S = 64, typename T = float>
static __device__ void gated_delta_net_scan_impl(
    const T*     __restrict__ q,
    const T*     __restrict__ k,
    const T*     __restrict__ v,
    const float* __restrict__ log_g,
    const float* __restrict__ beta,
    float*       __restrict__ state,
    float*       __restrict__ out,
    int C
) {
    const int bh   = blockIdx.x;
    const int tid  = threadIdx.x;
    const int NTHR = blockDim.x; // 256

    extern __shared__ float smem[];
    float* const s_attn  = smem;                            // [S, S]
    float* const s_a_row = smem + S * S;                   // [S]
    float* const s_log_g = s_a_row + S;                    // [S]
    float* const s_gcsum = s_log_g + S;                    // [S]
    float* const s_kbeta = s_gcsum + S;                    // [S, HK]  (F32, cast on load)
    float* const s_vcorr = s_kbeta + S * HK;              // [S, HV]

    float*    my_state  = state   + (long)bh * HK * HV;
    const T*  q_bh      = q       + (long)bh * C * S * HK;
    const T*  k_bh      = k       + (long)bh * C * S * HK;
    const T*  v_bh      = v       + (long)bh * C * S * HV;
    const float* logg_bh   = log_g   + (long)bh * C * S;
    const float* beta_bh   = beta    + (long)bh * C * S;
    float*       out_bh    = out     + (long)bh * C * S * HV;

    for (int ci = 0; ci < C; ci++) {
        const T*     q_c    = q_bh    + ci * S * HK;
        const T*     k_c    = k_bh    + ci * S * HK;
        const T*     v_c    = v_bh    + ci * S * HV;
        const float* logg_c = logg_bh + ci * S;
        const float* beta_c = beta_bh + ci * S;
        float*       out_c  = out_bh  + ci * S * HV;

        // ── Step 1: g_cumsum ──────────────────────────────────────────────────
        if (tid < S) {
            s_log_g[tid] = logg_c[tid];
            s_gcsum[tid] = logg_c[tid];
        }
        __syncthreads();
        prefix_sum_inplace(s_gcsum, tid, S); // all threads must call — no if(tid<S) guard

        // ── Step 2: Load k_beta (cast T → F32, store in s_kbeta) ─────────────
        for (int idx = tid; idx < S * HK; idx += NTHR) {
            int s  = idx / HK;
            int hk = idx % HK;
            s_kbeta[idx] = load_as_f32(k_c, s * HK + hk) * beta_c[s];
        }
        __syncthreads();

        // ── Step 3: Forward substitution → s_attn = (I − a_mat)^{-1} ─────────
        //
        // a_mat[i,j] = −dot(k_beta[i], k[j]) * exp(gc[i]−gc[j])  for j < i
        //
        // Row 0 of attn is e_0 (a_mat[0,:]=0 trivially).
        // Row i: attn[i,col] = e_i[col] + Σ_{j<i} a[i,j]*attn[j,col]
        //
        // Parallelism: tid ∈ [0, S) owns one column during the update phase.
        //              Threads [S, NTHR) are idle but must reach __syncthreads.

        // Init s_attn = I
        for (int idx = tid; idx < S * S; idx += NTHR) {
            int r = idx / S, c = idx % S;
            s_attn[idx] = (r == c) ? 1.0f : 0.0f;
        }
        __syncthreads();

        for (int i = 1; i < S; i++) {
            // Phase A: thread j (= tid, for tid < i) computes s_a_row[j].
            // Dot: Σ_hk s_kbeta[i*HK+hk] * k_c[tid*HK+hk]
            // s_kbeta is F32; k_c is T — use load2 on k_c (HK always even).
            if (tid < i) {
                float dot_val = 0.0f;
                for (int hk = 0; hk < HK; hk += 2) {
                    float2 kb = load2_as_f32<float>(s_kbeta + i * HK, hk);
                    float2 kc = load2_as_f32<T>(k_c + tid * HK, hk);
                    dot_val += kb.x * kc.x + kb.y * kc.y;
                }
                float decay   = __expf(s_gcsum[i] - s_gcsum[tid]);
                s_a_row[tid] = -dot_val * decay;
            }
            __syncthreads();

            // Phase B: thread col (= tid, for tid < S) updates attn[i, col].
            // Reads: s_a_row[0..i), s_attn[j*S+col] for j < i (rows 0..i-1).
            // Writes: s_attn[i*S+col] (row i, column col).
            // No aliasing: reads are from rows 0..i-1, write is to row i.
            if (tid < S) {
                float acc = 0.0f;
                for (int j = 0; j < i; j++) {
                    acc += s_a_row[j] * s_attn[j * S + tid];
                }
                s_attn[i * S + tid] += acc;
            }
            __syncthreads();
        }
        // s_attn now holds (I − a_mat)^{-1}.

        // ── Step 4: Compute v_delta into s_vcorr ──────────────────────────────
        //
        // v_delta[s2, hv] = beta[s2] * (v[s2,hv] − exp(gc[s2]) * Σ_hk kbeta[s2,hk]*state[hk,hv])
        //
        // This fuses value_new and the w@state correction into a single [S,HV] buffer.
        // v_c is T; scalar load per element (no vectorisation opportunity: different hv per thread).
        for (int idx = tid; idx < S * HV; idx += NTHR) {
            int s2 = idx / HV;
            int hv = idx % HV;
            float ws = 0.0f;
            for (int hk = 0; hk < HK; hk++) {
                ws += s_kbeta[s2 * HK + hk] * my_state[hk * HV + hv];
            }
            float g_exp_s2  = __expf(s_gcsum[s2]);
            float v_beta_s2 = load_as_f32(v_c + s2 * HV, hv) * beta_c[s2];
            s_vcorr[idx] = v_beta_s2 - g_exp_s2 * ws;
        }
        __syncthreads();
        // s_kbeta is free from this point.

        // ── Step 5: v_corrected = attn @ v_delta → out_c (global) ─────────────
        //
        // Reads: s_attn [S,S], s_vcorr [S,HV] (v_delta)
        // Writes: out_c [S,HV] (global)
        // No shared memory aliasing.
        for (int idx = tid; idx < S * HV; idx += NTHR) {
            int s1 = idx / HV;
            int hv = idx % HV;
            float acc = 0.0f;
            for (int s2 = 0; s2 < S; s2++) {
                acc += s_attn[s1 * S + s2] * s_vcorr[s2 * HV + hv];
            }
            out_c[s1 * HV + hv] = acc; // v_corrected stored in out_c
        }
        __syncthreads();
        // s_attn and s_vcorr (v_delta) are free from this point.

        // ── Step 6: Output = inter-chunk + intra-chunk → s_vcorr ─────────────
        //
        // inter[s1,hv] = Σ_hk (q[s1,hk]*exp(gc[s1])) * state[hk,hv]
        // intra[s1,hv] = Σ_{s2≤s1} exp(gc[s1]−gc[s2]) * dot(q[s1],k[s2]) * v_corr[s2,hv]
        //   where v_corr is in out_c.
        //
        // q_c and k_c are T.
        // inter: q_c pair-loaded, state stride-HV so scalar (non-contiguous hk axis).
        // intra dot: both q_c and k_c pair-loaded (contiguous within a row).
        for (int idx = tid; idx < S * HV; idx += NTHR) {
            int s1 = idx / HV;
            int hv = idx % HV;

            float gc_s1     = s_gcsum[s1];
            float exp_gc_s1 = __expf(gc_s1);

            // Inter-chunk: q_exp[s1] @ state[:, hv]
            // state[hk*HV+hv]: stride HV between consecutive hk → non-contiguous, scalar.
            float inter = 0.0f;
            for (int hk = 0; hk < HK; hk += 2) {
                float2 qv = load2_as_f32<T>(q_c + s1 * HK, hk);
                inter += qv.x * exp_gc_s1 * my_state[(hk    ) * HV + hv];
                inter += qv.y * exp_gc_s1 * my_state[(hk + 1) * HV + hv];
            }

            // Intra-chunk: decay_full[s1,s2] * dot(q[s1],k[s2]) * v_corr[s2,hv]
            float intra = 0.0f;
            for (int s2 = 0; s2 <= s1; s2++) {
                float gc_s2 = s_gcsum[s2];
                float decay = __expf(gc_s1 - gc_s2); // s2 <= s1 → exponent ≤ 0, safe
                float dot_qk = 0.0f;
                for (int hk = 0; hk < HK; hk += 2) {
                    float2 qv = load2_as_f32<T>(q_c + s1 * HK, hk);
                    float2 kv = load2_as_f32<T>(k_c + s2 * HK, hk);
                    dot_qk += qv.x * kv.x + qv.y * kv.y;
                }
                intra += dot_qk * decay * out_c[s2 * HV + hv];
            }

            s_vcorr[idx] = inter + intra;
        }
        __syncthreads();

        // ── Step 7: State update ──────────────────────────────────────────────
        //
        // state[hk,hv] = g_end*state[hk,hv] + Σ_{s2} k[s2,hk]*decay_to_end[s2]*v_corr[s2,hv]
        //   (beta already absorbed in v_corrected via v_beta in step 4)
        //   g_end = exp(gc[S-1]);  decay_to_end[s2] = exp(gc[S-1]−gc[s2])
        //
        // k_c is T; scalar load per (s2, hk) element (each thread owns one hk, loops S).
        {
            float gc_last = s_gcsum[S - 1];
            float g_end   = __expf(gc_last);

            for (int idx = tid; idx < HK * HV; idx += NTHR) {
                int hk = idx / HV;
                int hv = idx % HV;
                float acc = my_state[idx] * g_end;
                for (int s2 = 0; s2 < S; s2++) {
                    float decay_to_end = __expf(gc_last - s_gcsum[s2]);
                    float k_w = load_as_f32(k_c + s2 * HK, hk) * decay_to_end;
                    acc += k_w * out_c[s2 * HV + hv];
                }
                my_state[idx] = acc;
            }
        }
        __syncthreads();

        // ── Step 8: Write final output to out_c ───────────────────────────────
        for (int idx = tid; idx < S * HV; idx += NTHR) {
            out_c[idx] = s_vcorr[idx];
        }
        __syncthreads();
    } // end chunk loop
}

// ── Kernel entry points (extern "C") ─────────────────────────────────────────

#define DEF_SCAN_KERNEL(DTYPE_NAME, HK_VAL, HV_VAL, T_TYPE)                      \
extern "C" __global__                                                             \
__launch_bounds__(256, 2)                                                         \
void gated_delta_net_scan_##DTYPE_NAME##_hk##HK_VAL##_hv##HV_VAL(               \
    const T_TYPE* q,                                                              \
    const T_TYPE* k,                                                              \
    const T_TYPE* v,                                                              \
    const float*  log_g,                                                          \
    const float*  beta,                                                           \
    float*        state,                                                          \
    float*        out,                                                            \
    int C                                                                         \
) {                                                                               \
    gated_delta_net_scan_impl<HK_VAL, HV_VAL, 64, T_TYPE>(                       \
        q, k, v, log_g, beta, state, out, C);                                    \
}

DEF_SCAN_KERNEL(f32,  64,  64,  float)
DEF_SCAN_KERNEL(f32,  128, 128, float)
DEF_SCAN_KERNEL(bf16, 64,  64,  __nv_bfloat16)
DEF_SCAN_KERNEL(bf16, 128, 128, __nv_bfloat16)
