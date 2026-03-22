//! Tokenizer wrapper using HuggingFace tokenizers.

use anyhow::Result;
use tokenizers::Tokenizer;

/// Wraps a HuggingFace tokenizer loaded from the Hub.
pub struct TokenizerWrapper {
    inner: Tokenizer,
    eos_token_id: Option<u32>,
    /// Raw chat_template string from tokenizer_config.json (Jinja2 format).
    /// We don't render Jinja2 in Rust; instead we use this as a signal that
    /// a chat template exists and pick a known format.
    chat_template_hint: Option<String>,
}

impl TokenizerWrapper {
    /// Load tokenizer from a local directory (already downloaded via hf-hub).
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let inner = Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

        // Try to find EOS token ID
        let eos_token_id = inner.get_vocab(true).get("</s>").copied().or_else(|| {
            inner
                .get_vocab(true)
                .get("<|endoftext|>")
                .copied()
                .or_else(|| inner.get_vocab(true).get("<|end|>").copied())
        });

        // Try to detect chat template from the tokenizer config
        let chat_template_hint = Self::detect_chat_template_from_vocab(&inner);

        Ok(Self {
            inner,
            eos_token_id,
            chat_template_hint,
        })
    }

    /// Load the chat template hint from tokenizer_config.json if it exists
    /// alongside the tokenizer.json file.
    pub fn load_chat_template_from_config(&mut self, tokenizer_config_path: &std::path::Path) {
        if let Ok(content) = std::fs::read_to_string(tokenizer_config_path) {
            if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(template) = config.get("chat_template").and_then(|v| v.as_str()) {
                    self.chat_template_hint = Some(template.to_string());
                }
            }
        }
    }

    /// Detect which chat template format to use based on special tokens in
    /// the vocabulary.
    fn detect_chat_template_from_vocab(tokenizer: &Tokenizer) -> Option<String> {
        let vocab = tokenizer.get_vocab(true);
        if vocab.contains_key("<|im_start|>") && vocab.contains_key("<|im_end|>") {
            Some("chatml".to_string())
        } else if vocab.contains_key("[INST]") {
            Some("llama".to_string())
        } else {
            None
        }
    }

    /// Apply a chat template to a list of (role, content) pairs.
    ///
    /// We support two common formats:
    /// - ChatML: `<|im_start|>role\ncontent<|im_end|>\n` (Qwen, Yi, etc.)
    /// - Llama/Mistral: `[INST] content [/INST]`
    ///
    /// Returns an error if no template can be determined.
    pub fn apply_chat_template(&self, messages: &[(&str, &str)]) -> Result<String> {
        let template = self.chat_template_hint.as_deref().unwrap_or("chatml");

        match template {
            t if t.contains("im_start") || t == "chatml" => {
                let mut prompt = String::new();
                for (role, content) in messages {
                    prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, content));
                }
                prompt.push_str("<|im_start|>assistant\n");
                Ok(prompt)
            }
            t if t.contains("[INST]") || t == "llama" => {
                let mut prompt = String::new();
                let mut i = 0;
                // Handle system message
                if !messages.is_empty() && messages[0].0 == "system" {
                    prompt.push_str(&format!("[INST] <<SYS>>\n{}\n<</SYS>>\n\n", messages[0].1));
                    i = 1;
                }
                while i < messages.len() {
                    let (role, content) = messages[i];
                    if role == "user" {
                        if i == 0 || (i == 1 && messages[0].0 == "system") {
                            if !prompt.contains("[INST]") {
                                prompt.push_str(&format!("[INST] {} [/INST]", content));
                            } else {
                                prompt.push_str(&format!(" [INST] {} [/INST]", content));
                            }
                        } else {
                            prompt.push_str(&format!(" [INST] {} [/INST]", content));
                        }
                    } else if role == "assistant" {
                        prompt.push_str(&format!(" {} </s>", content));
                    }
                    i += 1;
                }
                // If the last message was from a user, the model should generate
                if !prompt.ends_with("[/INST]") {
                    prompt.push_str(" [INST] [/INST]");
                }
                Ok(prompt)
            }
            // Default: ChatML
            _ => {
                let mut prompt = String::new();
                for (role, content) in messages {
                    prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, content));
                }
                prompt.push_str("<|im_start|>assistant\n");
                Ok(prompt)
            }
        }
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenization error: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("detokenization error: {e}"))
    }

    /// Decode a single token to text.
    pub fn decode_token(&self, id: u32) -> Result<String> {
        self.decode(&[id])
    }

    /// Get EOS token ID.
    #[allow(dead_code)]
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    /// Set EOS token ID (used when loading from model config).
    pub fn set_eos_token_id(&mut self, id: u32) {
        self.eos_token_id = Some(id);
    }

    /// Get vocabulary size.
    #[allow(dead_code)]
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}
