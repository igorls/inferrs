//! `inferrs rm` — remove a cached HuggingFace model from local disk.

use anyhow::Result;
use clap::Parser;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::util::format_bytes;

#[derive(Parser, Clone)]
pub struct RmArgs {
    /// HuggingFace model ID(s) to remove (e.g. google/gemma-3-1b-it)
    pub models: Vec<String>,

    /// Skip confirmation prompt
    #[arg(short, long)]
    pub force: bool,
}

pub fn run(args: RmArgs) -> Result<()> {
    if args.models.is_empty() {
        anyhow::bail!("No model IDs specified. Usage: inferrs rm <model-id> [<model-id>...]");
    }

    let cache_dir = cache_root();

    for model_id in &args.models {
        let folder_name = model_folder_name(model_id);
        let model_path = cache_dir.join(&folder_name);

        if !model_path.exists() {
            eprintln!("Model not cached: {model_id}");
            continue;
        }

        let size = dir_size(&model_path).unwrap_or(0);

        if !args.force {
            eprint!("Remove {} ({})? [y/N] ", model_id, format_bytes(size));
            std::io::stderr().flush()?;
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            if !input.trim().eq_ignore_ascii_case("y") {
                println!("Skipped {model_id}");
                continue;
            }
        }

        std::fs::remove_dir_all(&model_path)?;
        println!("Removed {model_id} (freed {})", format_bytes(size));
    }

    Ok(())
}

/// Resolve the hf-hub cache root: `$HF_HOME/hub` or `~/.cache/huggingface/hub`.
fn cache_root() -> PathBuf {
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        PathBuf::from(hf_home).join("hub")
    } else if let Ok(xdg_cache) = std::env::var("XDG_CACHE_HOME") {
        PathBuf::from(xdg_cache).join("huggingface/hub")
    } else {
        dirs_home().join(".cache/huggingface/hub")
    }
}

/// Portable home directory without pulling in the `dirs` crate.
fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/"))
}

/// Convert "Org/Name" → "models--Org--Name" (mirrors hf-hub's `Repo::folder_name`).
fn model_folder_name(model_id: &str) -> String {
    format!("models--{model_id}").replace('/', "--")
}

/// Recursively sum the size of all files under `path`.
fn dir_size(path: &Path) -> Result<u64> {
    let mut total = 0u64;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            total += dir_size(&entry.path()).unwrap_or(0);
        } else {
            total += metadata.len();
        }
    }
    Ok(total)
}
