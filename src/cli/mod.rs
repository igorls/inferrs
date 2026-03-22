//! CLI entry point.
//!
//! Usage:
//!   inferrs serve Qwen/Qwen3.5-0.8B
//!   inferrs serve Qwen/Qwen3.5-0.8B --port 8080 --dtype f16

use crate::config::{InferrsConfig, SamplingConfig};
use crate::server;
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "inferrs",
    version,
    about = "Conservative-memory LLM inference engine"
)]
pub struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start the inference server.
    ///
    /// Example: inferrs serve Qwen/Qwen3.5-0.8B
    Serve(ServeArgs),
}

#[derive(Parser, Debug)]
pub struct ServeArgs {
    /// HuggingFace model ID (e.g. Qwen/Qwen3.5-0.8B)
    pub model: String,

    /// Model revision or branch
    #[arg(long)]
    pub revision: Option<String>,

    /// Weight data type: f32, f16, bf16
    #[arg(long, default_value = "f32")]
    pub dtype: String,

    /// Maximum sequence length (0 = model default)
    #[arg(long, default_value_t = 0)]
    pub max_seq_len: usize,

    /// Device: cpu, cuda, metal
    #[arg(long)]
    pub device: Option<String>,

    /// Host to bind to
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,

    /// Port to listen on
    #[arg(long, default_value_t = 8080)]
    pub port: u16,

    /// KV cache block size in tokens
    #[arg(long, default_value_t = 16)]
    pub block_size: usize,

    /// Initial KV cache blocks to allocate
    #[arg(long, default_value_t = 16)]
    pub initial_blocks: usize,

    /// Maximum KV cache blocks (0 = no limit)
    #[arg(long, default_value_t = 0)]
    pub max_blocks: usize,

    /// Maximum batch size
    #[arg(long, default_value_t = 32)]
    pub max_batch_size: usize,

    /// Maximum tokens per scheduler step
    #[arg(long, default_value_t = 2048)]
    pub max_tokens_per_step: usize,

    /// Default sampling temperature
    #[arg(long, default_value_t = 0.7)]
    pub temperature: f64,

    /// Default top-p
    #[arg(long, default_value_t = 0.9)]
    pub top_p: f64,

    /// Default top-k
    #[arg(long, default_value_t = 50)]
    pub top_k: usize,

    /// Default max tokens to generate
    #[arg(long, default_value_t = 2048)]
    pub max_tokens: usize,
}

impl Cli {
    pub fn run(self) -> anyhow::Result<()> {
        match self.command {
            Commands::Serve(args) => {
                let device = args.device.clone().unwrap_or_else(|| {
                    let cfg = InferrsConfig::default();
                    cfg.model.device.clone()
                });
                let config = InferrsConfig {
                    model: crate::config::ModelConfig {
                        model_id: args.model.clone(),
                        revision: args.revision.clone(),
                        dtype: args.dtype.clone(),
                        max_seq_len: args.max_seq_len,
                        device,
                    },
                    cache: crate::config::CacheConfig {
                        block_size: args.block_size,
                        initial_blocks: args.initial_blocks,
                        max_blocks: args.max_blocks,
                    },
                    scheduler: crate::config::SchedulerConfig {
                        max_batch_size: args.max_batch_size,
                        max_tokens_per_step: args.max_tokens_per_step,
                    },
                    server: crate::config::ServerConfig {
                        host: args.host.clone(),
                        port: args.port,
                    },
                    sampling: SamplingConfig {
                        temperature: args.temperature,
                        top_p: args.top_p,
                        top_k: args.top_k,
                        max_tokens: args.max_tokens,
                        repetition_penalty: 1.0,
                    },
                };

                tracing::info!("inferrs v{}", env!("CARGO_PKG_VERSION"));
                tracing::info!("model: {}", config.model.model_id);
                tracing::info!("device: {}", config.model.device);
                tracing::info!("dtype: {}", config.model.dtype);
                tracing::info!(
                    "cache: block_size={}, initial_blocks={}, max_blocks={}",
                    config.cache.block_size,
                    config.cache.initial_blocks,
                    config.cache.max_blocks
                );

                let rt = tokio::runtime::Runtime::new()?;
                rt.block_on(server::run(config))?;
                Ok(())
            }
        }
    }
}
