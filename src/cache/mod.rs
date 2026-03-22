//! KV Cache with grow-on-demand memory management.
//!
//! Unlike vLLM which pre-allocates gpu_memory_utilization (90%) of device
//! memory at startup, our KV cache grows like a C++ vector:
//!   - Start with a small initial allocation
//!   - Double capacity when more blocks are needed
//!   - Never shrink or deallocate blocks
//!   - Reuse freed blocks via a free list
//!
//! For a desktop: minimal footprint, leaving memory for other apps.
//! For a server: warms up to steady-state and stops allocating.

use crate::buffer::GrowVec;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

/// A block of KV cache for a fixed number of tokens.
///
/// Each block stores key and value tensors for all layers at `block_size`
/// token positions. This is the unit of allocation.
#[derive(Debug)]
pub struct CacheBlock {
    /// Key tensors per layer: shape (num_kv_heads, block_size, head_dim)
    pub keys: Vec<Tensor>,
    /// Value tensors per layer: shape (num_kv_heads, block_size, head_dim)
    pub values: Vec<Tensor>,
    /// How many token positions are filled in this block.
    pub used: usize,
}

impl CacheBlock {
    fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let mut keys = Vec::with_capacity(num_layers);
        let mut values = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            keys.push(Tensor::zeros(
                (num_kv_heads, block_size, head_dim),
                dtype,
                device,
            )?);
            values.push(Tensor::zeros(
                (num_kv_heads, block_size, head_dim),
                dtype,
                device,
            )?);
        }
        Ok(Self {
            keys,
            values,
            used: 0,
        })
    }

    fn reset(&mut self) {
        self.used = 0;
        // Don't deallocate tensors -- keep the memory. On next use,
        // we overwrite the tensor data positions as they're filled.
    }
}

/// Block pool that grows on demand and never shrinks.
///
/// This is the central allocator. It maintains a free list of reusable blocks.
/// When the free list is empty, it allocates new blocks in a doubling fashion.
pub struct BlockPool {
    /// All blocks ever allocated (never removed).
    blocks: GrowVec<CacheBlock>,
    /// Indices of free (reusable) blocks.
    free_list: GrowVec<usize>,
    /// Configuration
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    dtype: DType,
    device: Device,
    max_blocks: usize,
}

impl BlockPool {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        initial_blocks: usize,
        max_blocks: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let mut pool = Self {
            blocks: GrowVec::new(),
            free_list: GrowVec::new(),
            num_layers,
            num_kv_heads,
            head_dim,
            block_size,
            dtype,
            device: device.clone(),
            max_blocks,
        };

        // Allocate initial blocks
        if initial_blocks > 0 {
            pool.grow(initial_blocks)?;
        }

        tracing::info!(
            "block pool: initial={}, block_size={}, max={}",
            initial_blocks,
            block_size,
            if max_blocks == 0 {
                "unlimited".to_string()
            } else {
                max_blocks.to_string()
            }
        );

        Ok(pool)
    }

    /// Allocate a block index. Grows the pool if no free blocks.
    pub fn allocate(&mut self) -> Result<usize> {
        if let Some(idx) = self.free_list.pop() {
            self.blocks[idx].reset();
            return Ok(idx);
        }

        // No free blocks -- grow the pool
        let current = self.blocks.len();
        let grow_by = current.max(1); // Double, minimum 1
        if self.max_blocks > 0 && current + grow_by > self.max_blocks {
            let allowed = self.max_blocks.saturating_sub(current);
            if allowed == 0 {
                anyhow::bail!(
                    "KV cache exhausted: {} blocks allocated (max {})",
                    current,
                    self.max_blocks
                );
            }
            self.grow(allowed)?;
        } else {
            self.grow(grow_by)?;
        }

        // Now we have free blocks
        let idx = self.free_list.pop().unwrap();
        self.blocks[idx].reset();
        Ok(idx)
    }

    /// Return a block to the free list for reuse.
    pub fn free(&mut self, idx: usize) {
        debug_assert!(idx < self.blocks.len());
        self.free_list.push(idx);
    }

    /// Get a reference to a block.
    pub fn get(&self, idx: usize) -> &CacheBlock {
        &self.blocks[idx]
    }

    /// Get a mutable reference to a block.
    pub fn get_mut(&mut self, idx: usize) -> &mut CacheBlock {
        &mut self.blocks[idx]
    }

    /// Total blocks allocated (including free).
    #[allow(dead_code)]
    pub fn total_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Number of free (reusable) blocks.
    #[allow(dead_code)]
    pub fn free_blocks(&self) -> usize {
        self.free_list.len()
    }

    /// Number of blocks currently in use.
    #[allow(dead_code)]
    pub fn used_blocks(&self) -> usize {
        self.blocks.len() - self.free_list.len()
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Grow the pool by `count` new blocks.
    fn grow(&mut self, count: usize) -> Result<()> {
        let start = self.blocks.len();
        tracing::debug!("growing block pool: {} -> {} blocks", start, start + count);
        for i in 0..count {
            let block = CacheBlock::new(
                self.num_layers,
                self.num_kv_heads,
                self.head_dim,
                self.block_size,
                self.dtype,
                &self.device,
            )?;
            self.blocks.push(block);
            self.free_list.push(start + i);
        }
        Ok(())
    }
}

/// Per-sequence KV cache, tracking which blocks are used by a sequence.
pub struct SequenceCache {
    /// Block indices owned by this sequence, in order.
    pub block_indices: GrowVec<usize>,
    /// Total number of tokens cached.
    pub num_tokens: usize,
}

impl SequenceCache {
    pub fn new() -> Self {
        Self {
            block_indices: GrowVec::new(),
            num_tokens: 0,
        }
    }

    /// Number of blocks used by this sequence.
    #[allow(dead_code)]
    pub fn num_blocks(&self) -> usize {
        self.block_indices.len()
    }
}

/// Manages KV cache allocation across sequences.
///
/// This is the analogue of vLLM's KVCacheManager but with grow-on-demand
/// semantics instead of pre-allocation.
pub struct KvCacheManager {
    pool: BlockPool,
    block_size: usize,
}

impl KvCacheManager {
    pub fn new(pool: BlockPool) -> Self {
        let block_size = pool.block_size();
        Self { pool, block_size }
    }

    /// Allocate enough blocks for `num_new_tokens` more tokens in a sequence.
    pub fn allocate_slots(
        &mut self,
        seq_cache: &mut SequenceCache,
        num_new_tokens: usize,
    ) -> Result<()> {
        let total_needed = seq_cache.num_tokens + num_new_tokens;
        let blocks_needed = total_needed.div_ceil(self.block_size);

        while seq_cache.block_indices.len() < blocks_needed {
            let idx = self.pool.allocate()?;
            seq_cache.block_indices.push(idx);
        }

        seq_cache.num_tokens = total_needed;
        Ok(())
    }

    /// Free all blocks owned by a sequence.
    pub fn free_sequence(&mut self, seq_cache: &mut SequenceCache) {
        for &idx in seq_cache.block_indices.iter() {
            self.pool.free(idx);
        }
        seq_cache.block_indices.clear();
        seq_cache.num_tokens = 0;
    }

    /// Build KV cache tensors from a sequence's blocks for the model forward pass.
    ///
    /// Returns one (k, v) pair per layer, concatenated across all blocks.
    /// Shape per layer: (1, num_kv_heads, total_tokens, head_dim)
    pub fn build_kv_tensors(&self, seq_cache: &SequenceCache) -> Result<Vec<(Tensor, Tensor)>> {
        if seq_cache.block_indices.is_empty() {
            return Ok(vec![]);
        }

        let num_layers = self.pool.blocks[0].keys.len();
        let total_tokens = seq_cache.num_tokens;
        let mut result = Vec::with_capacity(num_layers);

        for layer_idx in 0..num_layers {
            let mut k_parts = Vec::new();
            let mut v_parts = Vec::new();

            let mut tokens_remaining = total_tokens;
            for &block_idx in seq_cache.block_indices.iter() {
                let block = self.pool.get(block_idx);
                let tokens_in_block = tokens_remaining.min(self.block_size);

                let k = block.keys[layer_idx].narrow(1, 0, tokens_in_block)?;
                let v = block.values[layer_idx].narrow(1, 0, tokens_in_block)?;
                k_parts.push(k);
                v_parts.push(v);
                tokens_remaining -= tokens_in_block;
            }

            let k = Tensor::cat(&k_parts, 1)?.unsqueeze(0)?;
            let v = Tensor::cat(&v_parts, 1)?.unsqueeze(0)?;
            result.push((k, v));
        }

        Ok(result)
    }

    /// Store new KV cache entries from the model's full accumulated KV tensors.
    ///
    /// The model forward pass returns the full KV (prev_cache + new) per layer.
    /// new_kv[layer] shape: (1, num_kv_heads, total_seq_len, head_dim)
    /// We extract tokens at positions [offset..offset+num_new_tokens] and store
    /// them into the appropriate cache blocks.
    pub fn store_kv_from_full(
        &mut self,
        seq_cache: &SequenceCache,
        new_kv: &[(Tensor, Tensor)],
        offset: usize,
        num_new_tokens: usize,
    ) -> Result<()> {
        for (layer_idx, (k, v)) in new_kv.iter().enumerate() {
            let k = k.squeeze(0)?; // (num_kv_heads, total_seq_len, head_dim)
            let v = v.squeeze(0)?;

            let mut token_pos = offset;
            for t in 0..num_new_tokens {
                let block_idx_in_seq = token_pos / self.block_size;
                let pos_in_block = token_pos % self.block_size;

                if block_idx_in_seq < seq_cache.block_indices.len() {
                    let block_idx = seq_cache.block_indices[block_idx_in_seq];
                    let block = self.pool.get_mut(block_idx);

                    // The full tensor has positions [0..total_seq_len].
                    // The token we want is at position (offset + t) in the full tensor.
                    let src_pos = offset + t;
                    let k_token = k.narrow(1, src_pos, 1)?; // (num_kv_heads, 1, head_dim)
                    let v_token = v.narrow(1, src_pos, 1)?;

                    block.keys[layer_idx] = block.keys[layer_idx].slice_assign(
                        &[
                            0..k_token.dim(0)?,
                            pos_in_block..pos_in_block + 1,
                            0..k_token.dim(2)?,
                        ],
                        &k_token,
                    )?;
                    block.values[layer_idx] = block.values[layer_idx].slice_assign(
                        &[
                            0..v_token.dim(0)?,
                            pos_in_block..pos_in_block + 1,
                            0..v_token.dim(2)?,
                        ],
                        &v_token,
                    )?;

                    block.used = block.used.max(pos_in_block + 1);
                }
                token_pos += 1;
                let _ = t; // suppress unused warning
            }
        }

        Ok(())
    }

    #[allow(dead_code)]
    pub fn pool(&self) -> &BlockPool {
        &self.pool
    }

    #[allow(dead_code)]
    pub fn block_size(&self) -> usize {
        self.block_size
    }
}
