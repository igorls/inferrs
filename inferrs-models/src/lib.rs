//! Shared model implementations and supporting types for `inferrs`.
//!
//! This crate is a workspace-level extraction from the main `inferrs` binary
//! that allows backend plugins (e.g. `inferrs-backend-cuda`) to share the same
//! model code as the main binary while compiling against different
//! `candle-core` feature sets.
//!
//! The extraction is being performed incrementally — this initial revision
//! only hosts the block-based [`kv_cache`] module.  Additional modules
//! (`config`, `turbo_quant`, `nvfp4`, and the model implementations
//! themselves) will move here in subsequent PRs.

pub mod kv_cache;
