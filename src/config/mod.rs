//! Configuration for the inference engine.

use serde::{Deserialize, Serialize};

/// Top-level configuration for the engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferrsConfig {
    pub model: ModelConfig,
    pub cache: CacheConfig,
    pub scheduler: SchedulerConfig,
    pub server: ServerConfig,
    pub sampling: SamplingConfig,
}

/// Model-related configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// HuggingFace model ID, e.g. "Qwen/Qwen3.5-0.8B"
    pub model_id: String,
    /// Optional revision / branch
    pub revision: Option<String>,
    /// Data type for weights: "f32", "f16", "bf16"
    pub dtype: String,
    /// Maximum sequence length (0 = use model default)
    pub max_seq_len: usize,
    /// Device to run on: "cpu", "cuda", "metal"
    pub device: String,
}

/// KV cache configuration.
///
/// Unlike vLLM which pre-allocates gpu_memory_utilization (default 90%) of GPU
/// memory, we start with a small number of cache blocks and grow on demand.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Number of tokens per cache block (like vLLM's block_size).
    pub block_size: usize,
    /// Initial number of blocks to allocate.
    /// We start small and grow. This is the opposite of vLLM's approach.
    pub initial_blocks: usize,
    /// Maximum number of blocks (0 = no limit, grow until OOM).
    pub max_blocks: usize,
}

/// Scheduler configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Maximum number of sequences that can be batched together.
    pub max_batch_size: usize,
    /// Maximum total tokens per scheduler step.
    pub max_tokens_per_step: usize,
}

/// HTTP server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

/// Default sampling parameters (can be overridden per-request).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub max_tokens: usize,
    pub repetition_penalty: f64,
}

impl Default for InferrsConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig {
                model_id: String::new(),
                revision: None,
                dtype: "f32".to_string(),
                max_seq_len: 0,
                device: detect_device(),
            },
            cache: CacheConfig {
                block_size: 16,
                initial_blocks: 16,
                max_blocks: 0,
            },
            scheduler: SchedulerConfig {
                max_batch_size: 32,
                max_tokens_per_step: 2048,
            },
            server: ServerConfig {
                host: "0.0.0.0".to_string(),
                port: 8080,
            },
            sampling: SamplingConfig {
                temperature: 0.7,
                top_p: 0.9,
                top_k: 50,
                max_tokens: 2048,
                repetition_penalty: 1.0,
            },
        }
    }
}

/// Auto-detect the best available device.
fn detect_device() -> String {
    if cfg!(feature = "cuda") {
        "cuda".to_string()
    } else if cfg!(feature = "metal") {
        "metal".to_string()
    } else {
        "cpu".to_string()
    }
}

impl ModelConfig {
    pub fn candle_dtype(&self) -> candle_core::DType {
        match self.dtype.as_str() {
            "f16" => candle_core::DType::F16,
            "bf16" => candle_core::DType::BF16,
            _ => candle_core::DType::F32,
        }
    }

    pub fn candle_device(&self) -> anyhow::Result<candle_core::Device> {
        match self.device.as_str() {
            "cpu" => Ok(candle_core::Device::Cpu),
            #[cfg(feature = "cuda")]
            "cuda" => Ok(candle_core::Device::new_cuda(0)?),
            #[cfg(feature = "metal")]
            "metal" => Ok(candle_core::Device::new_metal(0)?),
            other => anyhow::bail!(
                "unsupported device: {other} (compile with --features {other} to enable)"
            ),
        }
    }
}
