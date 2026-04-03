//! HuggingFace Hub model downloading.

use anyhow::{Context, Result};
use candle_core::quantized::GgmlDType;
use hf_hub::api::sync::Api;
use std::path::PathBuf;

/// Files needed to load a model.
pub struct ModelFiles {
    pub config_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub tokenizer_config_path: Option<PathBuf>,
    /// Original safetensors shards (always present).
    pub weight_paths: Vec<PathBuf>,
    /// Path to the quantized GGUF file, populated when `--quantize` was given.
    /// When `Some`, callers should load weights from this GGUF instead of
    /// `weight_paths`.
    pub gguf_path: Option<PathBuf>,
}

/// Load model files from a local directory (no network required).
pub fn load_local_model(path: &std::path::Path) -> Result<ModelFiles> {
    tracing::info!("Loading model from local path: {}", path.display());

    let config_path = path.join("config.json");
    anyhow::ensure!(
        config_path.exists(),
        "config.json not found in {}",
        path.display()
    );

    let tokenizer_path = path.join("tokenizer.json");
    anyhow::ensure!(
        tokenizer_path.exists(),
        "tokenizer.json not found in {}",
        path.display()
    );

    let tokenizer_config_path = {
        let p = path.join("tokenizer_config.json");
        if p.exists() {
            Some(p)
        } else {
            None
        }
    };

    // Prefer model.safetensors, then scan for shards
    let weight_paths = if path.join("model.safetensors").exists() {
        vec![path.join("model.safetensors")]
    } else {
        let mut shards: Vec<PathBuf> = std::fs::read_dir(path)
            .with_context(|| format!("Cannot read {}", path.display()))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|e| e == "safetensors").unwrap_or(false))
            .collect();
        shards.sort();
        anyhow::ensure!(
            !shards.is_empty(),
            "No safetensors files found in {}",
            path.display()
        );
        shards
    };

    Ok(ModelFiles {
        config_path,
        tokenizer_path,
        tokenizer_config_path,
        weight_paths,
        gguf_path: None,
    })
}

/// Download model files from HuggingFace Hub.
pub fn download_model(model_id: &str, revision: &str) -> Result<ModelFiles> {
    // If the model_id looks like a local path, load directly without network.
    let as_path = std::path::Path::new(model_id);
    if as_path.is_absolute()
        || model_id.starts_with("./")
        || model_id.starts_with("../")
        || as_path.exists()
    {
        return load_local_model(as_path);
    }

    tracing::info!("Downloading model {} (revision: {})", model_id, revision);

    let api = Api::new().context("Failed to create HuggingFace API client")?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        model_id.to_string(),
        hf_hub::RepoType::Model,
        revision.to_string(),
    ));

    // Download config.json
    let config_path = repo
        .get("config.json")
        .context("Failed to download config.json")?;
    tracing::info!("Downloaded config.json");

    // Download tokenizer.json
    let tokenizer_path = repo
        .get("tokenizer.json")
        .context("Failed to download tokenizer.json")?;
    tracing::info!("Downloaded tokenizer.json");

    // Try to download tokenizer_config.json (optional)
    let tokenizer_config_path = repo.get("tokenizer_config.json").ok();
    if tokenizer_config_path.is_some() {
        tracing::info!("Downloaded tokenizer_config.json");
    }

    // Download safetensors weight files
    let weight_paths = download_safetensors(&repo)?;

    Ok(ModelFiles {
        config_path,
        tokenizer_path,
        tokenizer_config_path,
        weight_paths,
        gguf_path: None,
    })
}

/// Download the model (same as [`download_model`]) and, when `quant_dtype` is
/// `Some`, ensure a quantized GGUF is present on disk.
///
/// The GGUF is written next to the safetensors shards in the HF hub cache.
/// If the file already exists it is reused without re-running the conversion.
/// Quantization happens on the CPU and can take up to a few minutes for large
/// models; progress is logged at INFO level.
pub fn download_and_maybe_quantize(
    model_id: &str,
    revision: &str,
    quant_dtype: Option<GgmlDType>,
) -> Result<ModelFiles> {
    let mut files = download_model(model_id, revision)?;

    let Some(dtype) = quant_dtype else {
        return Ok(files);
    };

    let gguf = crate::quantize::gguf_path(&files.weight_paths, dtype);

    if gguf.exists() {
        tracing::info!("Reusing cached GGUF at {} ({:?})", gguf.display(), dtype);
    } else {
        tracing::info!(
            "Quantizing model to {:?} — this runs once and is then cached…",
            dtype
        );
        // Write to a temp path then atomically rename so that an interrupted
        // conversion (OOM, Ctrl-C, disk-full) never leaves a truncated file
        // that would be silently reused on the next run.
        let tmp = gguf.with_extension("gguf.tmp");
        crate::quantize::convert_to_gguf(&files.weight_paths, &tmp, dtype)?;
        std::fs::rename(&tmp, &gguf)
            .with_context(|| format!("Failed to rename {} → {}", tmp.display(), gguf.display()))?;
    }

    files.gguf_path = Some(gguf);
    Ok(files)
}

fn download_safetensors(repo: &hf_hub::api::sync::ApiRepo) -> Result<Vec<PathBuf>> {
    // Try model.safetensors first (single file models)
    if let Ok(path) = repo.get("model.safetensors") {
        tracing::info!("Downloaded model.safetensors");
        return Ok(vec![path]);
    }

    // Try model.safetensors.index.json for sharded models
    let index_path = repo
        .get("model.safetensors.index.json")
        .context("No model.safetensors or model.safetensors.index.json found")?;

    let index_content =
        std::fs::read_to_string(&index_path).context("Failed to read safetensors index")?;
    let index: serde_json::Value =
        serde_json::from_str(&index_content).context("Failed to parse safetensors index")?;

    let weight_map = index
        .get("weight_map")
        .and_then(|v| v.as_object())
        .context("No weight_map in safetensors index")?;

    // Collect unique filenames
    let mut filenames: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    filenames.sort();
    filenames.dedup();

    let mut paths = Vec::new();
    for filename in &filenames {
        let path = repo
            .get(filename)
            .with_context(|| format!("Failed to download {filename}"))?;
        tracing::info!("Downloaded {}", filename);
        paths.push(path);
    }

    Ok(paths)
}
