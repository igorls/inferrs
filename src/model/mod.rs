//! Model loading and management.
//!
//! Downloads model files from HuggingFace Hub, parses config.json,
//! and loads weights from safetensors.

mod loader;
mod transformer;

pub use loader::ModelLoader;
pub use transformer::{ModelConfig as TransformerConfig, TransformerModel};
