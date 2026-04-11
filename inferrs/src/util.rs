//! Shared utility helpers.
//!
//! Includes model cache discovery shared by the `inferrs list` CLI command
//! and the HTTP server's `/v1/models` and `/api/tags` endpoints.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::Result;

/// Format a byte count as a human-readable string (GiB / MiB / KiB / B).
pub fn format_bytes(bytes: u64) -> String {
    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;
    const KIB: u64 = 1 << 10;
    if bytes >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.1} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.1} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Resolve the hf-hub cache root: `$HF_HOME/hub` or `~/.cache/huggingface/hub`.
pub fn cache_root() -> PathBuf {
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        PathBuf::from(hf_home).join("hub")
    } else if let Ok(xdg_cache) = std::env::var("XDG_CACHE_HOME") {
        PathBuf::from(xdg_cache).join("huggingface/hub")
    } else {
        home_dir().join(".cache/huggingface/hub")
    }
}

/// Portable home directory without pulling in the `dirs` crate.
///
/// Checks `HOME` (Unix) then `USERPROFILE` (Windows), falling back to `/`.
pub fn home_dir() -> PathBuf {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/"))
}

// ── Cached model discovery ────────────────────────────────────────────────────

/// Metadata for a single model found in the HuggingFace hub cache.
#[derive(Debug, Clone)]
pub struct CachedModel {
    /// HuggingFace model ID, e.g. `"google/gemma-4-E2B-it"`.
    pub model_id: String,
    /// Total size of all files in the model's cache directory (bytes).
    pub size_bytes: u64,
    /// Last modification time of the snapshot directory, if available.
    pub modified: Option<SystemTime>,
}

/// Scan the HuggingFace hub cache and return all models found there.
///
/// The cache layout is:
/// ```text
/// $HF_HOME/hub/
///   models--Org--Name/
///     snapshots/
///       <sha>/
///         config.json   ← presence indicates a usable model snapshot
/// ```
///
/// Each `models--*` subdirectory maps to one model ID.  The returned list is
/// sorted alphabetically by model ID.
pub fn list_cached_models() -> Vec<CachedModel> {
    let cache_dir = cache_root();
    if !cache_dir.exists() {
        return vec![];
    }

    let mut models: Vec<CachedModel> = std::fs::read_dir(&cache_dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_type().map(|t| t.is_dir()).unwrap_or(false)
                && e.file_name().to_string_lossy().starts_with("models--")
        })
        .map(|e| {
            let folder = e.file_name().to_string_lossy().into_owned();
            let model_id = folder_to_model_id(&folder);
            let size_bytes = dir_size(&e.path()).unwrap_or(0);
            // Try to find the most recent snapshot modification time.
            let modified = snapshot_mtime(&e.path());
            CachedModel {
                model_id,
                size_bytes,
                modified,
            }
        })
        .collect();

    models.sort_by(|a, b| a.model_id.cmp(&b.model_id));
    models
}

/// Convert the HF hub folder name `"models--Org--Name"` → `"Org/Name"`.
pub fn folder_to_model_id(folder: &str) -> String {
    folder
        .strip_prefix("models--")
        .unwrap_or(folder)
        .replace("--", "/")
}

/// Return the most recent mtime among all snapshot directories for a model.
fn snapshot_mtime(model_dir: &Path) -> Option<SystemTime> {
    let snapshots_dir = model_dir.join("snapshots");
    std::fs::read_dir(&snapshots_dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .filter_map(|e| e.metadata().ok()?.modified().ok())
        .max()
}

/// Recursively sum the size of all files under `path`.
///
/// Uses `symlink_metadata` so that symbolic links (e.g. the HuggingFace
/// `snapshots/` → `blobs/` indirection) are not followed and their targets
/// are not double-counted.
pub fn dir_size(path: &Path) -> Result<u64> {
    let mut total = 0u64;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let metadata = entry.path().symlink_metadata()?;
        if metadata.is_dir() {
            total += dir_size(&entry.path()).unwrap_or(0);
        } else if metadata.is_file() {
            total += metadata.len();
        }
    }
    Ok(total)
}
