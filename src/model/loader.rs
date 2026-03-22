//! Model downloading and weight loading from HuggingFace Hub.

use anyhow::{Context, Result};
use hf_hub::api::sync::Api;
use std::path::PathBuf;

/// Handles downloading and locating model files.
pub struct ModelLoader {
    model_id: String,
    revision: Option<String>,
}

impl ModelLoader {
    pub fn new(model_id: &str, revision: Option<&str>) -> Self {
        Self {
            model_id: model_id.to_string(),
            revision: revision.map(|s| s.to_string()),
        }
    }

    /// Download (or locate cached) model files and return their paths.
    pub fn fetch(&self) -> Result<ModelFiles> {
        tracing::info!("fetching model: {}", self.model_id);

        let api = Api::new().context("failed to create HuggingFace Hub API client")?;
        let repo = match &self.revision {
            Some(rev) => api.repo(hf_hub::Repo::with_revision(
                self.model_id.clone(),
                hf_hub::RepoType::Model,
                rev.clone(),
            )),
            None => api.model(self.model_id.clone()),
        };

        // Download config.json
        let config_path = repo
            .get("config.json")
            .context("failed to download config.json")?;
        tracing::info!("config: {}", config_path.display());

        // Download tokenizer
        let tokenizer_path = repo
            .get("tokenizer.json")
            .context("failed to download tokenizer.json")?;
        tracing::info!("tokenizer: {}", tokenizer_path.display());

        // Download tokenizer_config.json (optional, for chat template)
        let tokenizer_config_path = repo.get("tokenizer_config.json").ok();
        if tokenizer_config_path.is_some() {
            tracing::debug!("tokenizer_config.json downloaded");
        }

        // Download safetensors weight files
        let weight_paths = self.fetch_weights(&repo)?;
        tracing::info!("weights: {} file(s)", weight_paths.len());

        // Try to download generation_config.json (optional)
        let generation_config_path = repo.get("generation_config.json").ok();

        Ok(ModelFiles {
            config: config_path,
            tokenizer: tokenizer_path,
            tokenizer_config: tokenizer_config_path,
            weights: weight_paths,
            generation_config: generation_config_path,
        })
    }

    /// Fetch safetensors weight files.
    fn fetch_weights(&self, repo: &hf_hub::api::sync::ApiRepo) -> Result<Vec<PathBuf>> {
        // Try single-file model first
        if let Ok(path) = repo.get("model.safetensors") {
            return Ok(vec![path]);
        }

        // Try sharded model: model.safetensors.index.json
        let index_path = repo
            .get("model.safetensors.index.json")
            .context("no model.safetensors or model.safetensors.index.json found")?;

        let index_content =
            std::fs::read_to_string(&index_path).context("failed to read safetensors index")?;
        let index: serde_json::Value =
            serde_json::from_str(&index_content).context("failed to parse safetensors index")?;

        let weight_map = index["weight_map"]
            .as_object()
            .context("missing weight_map in safetensors index")?;

        // Collect unique filenames
        let mut filenames: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect();
        filenames.sort();
        filenames.dedup();

        let mut paths = Vec::new();
        for filename in &filenames {
            let path = repo
                .get(filename)
                .with_context(|| format!("failed to download weight file: {filename}"))?;
            paths.push(path);
        }

        Ok(paths)
    }
}

/// Paths to all downloaded model files.
pub struct ModelFiles {
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub tokenizer_config: Option<PathBuf>,
    pub weights: Vec<PathBuf>,
    #[allow(dead_code)]
    pub generation_config: Option<PathBuf>,
}
