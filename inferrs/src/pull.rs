//! `inferrs pull` — pre-download a model to the local cache.
//!
//! Reference resolution:
//!   - `oneword`                → OCI pull via `POST /api/pull` on the server
//!   - `wordone/wordtwo`        → HuggingFace pull (default for org/model)
//!   - `hf.co/org/model`        → HuggingFace pull
//!   - `huggingface.co/org/model` → HuggingFace pull
//!   - `docker.io/org/model`    → OCI pull via `POST /api/pull` on the server
//!   - `registry.io/org/model`  → OCI pull via `POST /api/pull` on the server
//!
//! OCI pulls are delegated to the `inferrs serve` daemon through its
//! Ollama-compatible `/api/pull` endpoint.  If the server is not running,
//! it is auto-started (same pattern as `inferrs run`).

use anyhow::{Context, Result};
use clap::Parser;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::io::Write;
use std::path::PathBuf;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// On-demand loading of the Go OCI shared library (libocimodels) via dlopen.
//
// The library is NOT linked at build time — it is loaded lazily the first
// time an OCI operation is requested.  This keeps the inferrs binary small
// and allows it to start even when the library is absent (HuggingFace models
// still work; only OCI pulls require the library).
// ---------------------------------------------------------------------------

/// Function-pointer signatures matching the Go C exports in lib.go.
type FnOciPull = unsafe extern "C" fn(*const c_char) -> *mut c_char;
type FnOciBundle = unsafe extern "C" fn(*const c_char) -> *mut c_char;
type FnOciPullStream = unsafe extern "C" fn(
    *const c_char,
    Option<unsafe extern "C" fn(*const c_char, *mut c_void)>,
    *mut c_void,
) -> c_int;
type FnOciList = unsafe extern "C" fn() -> *mut c_char;
type FnOciLastError = unsafe extern "C" fn() -> *mut c_char;
type FnOciFreeString = unsafe extern "C" fn(*mut c_char);

/// Holds the dlopen'd library handle and resolved function pointers.
struct OciLib {
    // The Library owns the dlopen handle; dropping it would dlclose.
    // We keep it alive for the lifetime of the process.
    _lib: libloading::Library,
    pull: FnOciPull,
    bundle: FnOciBundle,
    pull_stream: FnOciPullStream,
    list: FnOciList,
    last_error: FnOciLastError,
    free_string: FnOciFreeString,
}

// SAFETY: The Go shared library is thread-safe (uses a mutex internally).
unsafe impl Send for OciLib {}
unsafe impl Sync for OciLib {}

/// Lazily loaded library — initialised on first OCI operation.
static OCI_LIB: OnceLock<Result<OciLib, String>> = OnceLock::new();

/// Platform-specific library filename.
#[cfg(target_os = "macos")]
const LIB_NAME: &str = "libocimodels.dylib";
#[cfg(target_os = "linux")]
const LIB_NAME: &str = "libocimodels.so";
#[cfg(target_os = "windows")]
const LIB_NAME: &str = "ocimodels.dll";
#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
const LIB_NAME: &str = "libocimodels.so";

/// Try to load the OCI shared library, returning a reference to the resolved
/// function pointers.  The library is loaded at most once; subsequent calls
/// return the cached result.
fn load_oci_lib() -> Result<&'static OciLib, String> {
    OCI_LIB
        .get_or_init(try_load_oci_lib)
        .as_ref()
        .map_err(|e| e.clone())
}

fn try_load_oci_lib() -> Result<OciLib, String> {
    // Search order:
    //   1. Same directory as the running executable (typical deployment layout)
    //   2. System library search path (LD_LIBRARY_PATH, DYLD_LIBRARY_PATH, PATH)
    let exe_dir_lib = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join(LIB_NAME)));

    let lib = if let Some(ref path) = exe_dir_lib {
        // Try next to the executable first.
        unsafe { libloading::Library::new(path) }
            .or_else(|_| unsafe { libloading::Library::new(LIB_NAME) })
    } else {
        unsafe { libloading::Library::new(LIB_NAME) }
    };

    let lib = lib.map_err(|e| {
        format!(
            "OCI model operations require {LIB_NAME} but it was not found. \
             Build it with `make oci-lib` and place it next to the inferrs binary. \
             (dlopen error: {e})"
        )
    })?;

    // Resolve the exported symbols.
    unsafe {
        let pull: FnOciPull = *lib
            .get::<FnOciPull>(b"oci_pull\0")
            .map_err(|e| format!("failed to resolve oci_pull: {e}"))?;
        let bundle: FnOciBundle = *lib
            .get::<FnOciBundle>(b"oci_bundle\0")
            .map_err(|e| format!("failed to resolve oci_bundle: {e}"))?;
        let pull_stream: FnOciPullStream = *lib
            .get::<FnOciPullStream>(b"oci_pull_stream\0")
            .map_err(|e| format!("failed to resolve oci_pull_stream: {e}"))?;
        let list: FnOciList = *lib
            .get::<FnOciList>(b"oci_list\0")
            .map_err(|e| format!("failed to resolve oci_list: {e}"))?;
        let last_error: FnOciLastError = *lib
            .get::<FnOciLastError>(b"oci_last_error\0")
            .map_err(|e| format!("failed to resolve oci_last_error: {e}"))?;
        let free_string: FnOciFreeString = *lib
            .get::<FnOciFreeString>(b"oci_free_string\0")
            .map_err(|e| format!("failed to resolve oci_free_string: {e}"))?;

        Ok(OciLib {
            _lib: lib,
            pull,
            bundle,
            pull_stream,
            list,
            last_error,
            free_string,
        })
    }
}

/// Read and free the last error from the Go library.
fn get_last_oci_error(lib: &OciLib) -> String {
    unsafe {
        let err_ptr = (lib.last_error)();
        if err_ptr.is_null() {
            return "unknown error".to_string();
        }
        let msg = CStr::from_ptr(err_ptr).to_string_lossy().into_owned();
        (lib.free_string)(err_ptr);
        msg
    }
}

/// Normalize an OCI reference the same way Docker Model Runner does.
///
/// This is used only for `/api/pull` bookkeeping so concurrent pull requests
/// such as `gemma3` and `ai/gemma3:latest` share the same in-flight job.
pub fn normalize_oci_reference(reference: &str) -> String {
    const DEFAULT_ORG: &str = "ai";
    const DEFAULT_TAG: &str = "latest";

    let mut reference = reference.trim().to_string();
    if reference.is_empty() {
        return reference;
    }

    if let Some(rest) = reference.strip_prefix("hf.co/") {
        reference = format!("huggingface.co/{rest}");
    }

    let last_slash = reference.rfind('/').unwrap_or(0);
    let last_colon = reference.rfind(':');
    let has_tag = matches!(last_colon, Some(idx) if idx > last_slash);

    let (mut name, tag) = if has_tag {
        let colon = last_colon.expect("checked above");
        let name = reference[..colon].to_string();
        let tag = reference[colon + 1..].trim();
        let tag = if tag.is_empty() { DEFAULT_TAG } else { tag };
        (name, tag.to_string())
    } else {
        (reference, DEFAULT_TAG.to_string())
    };

    let first_slash = name.find('/');
    let has_registry = matches!(first_slash, Some(idx) if name[..idx].contains('.'));

    if !has_registry && !name.contains('/') {
        name = format!("{DEFAULT_ORG}/{name}");
    }

    format!("{}:{tag}", name.to_lowercase())
}

// ---------------------------------------------------------------------------
// OCI pull streaming (via FFI callback) — used by the HTTP server's
// `POST /api/pull` endpoint to stream Ollama-compatible NDJSON progress
// without spawning a separate helper process.
// ---------------------------------------------------------------------------

/// Handle returned by [`oci_pull_stream_start`].  The receiver yields one
/// NDJSON line per message; when the channel closes the pull is complete.
pub struct OciStreamHandle {
    /// Receiver for NDJSON progress lines from the Go shared library.
    pub lines: tokio::sync::mpsc::UnboundedReceiver<String>,
}

/// C callback trampoline invoked by the Go shared library for each NDJSON
/// progress line.  `ctx` points to a boxed `UnboundedSender<String>`.
///
/// # Safety
/// Called from Go goroutines via the C trampoline.  The pointer must remain
/// valid until `oci_pull_stream` returns (guaranteed by the caller).
unsafe extern "C" fn pull_stream_trampoline(line: *const c_char, ctx: *mut c_void) {
    if line.is_null() || ctx.is_null() {
        return;
    }
    let tx = unsafe { &*(ctx as *const tokio::sync::mpsc::UnboundedSender<String>) };
    if let Ok(s) = unsafe { CStr::from_ptr(line) }.to_str() {
        let _ = tx.send(s.to_string());
    }
}

/// Start an OCI pull with streaming Ollama-compatible NDJSON progress.
///
/// The pull runs on a dedicated OS thread (the Go shared library blocks).
/// Each NDJSON line is delivered through the returned channel.  When the
/// channel closes, the pull is complete.  Errors are communicated as NDJSON
/// objects with an `"error"` field, matching the Ollama wire format.
pub fn oci_pull_stream_start(reference: &str) -> Result<OciStreamHandle> {
    let lib = load_oci_lib().map_err(|e| anyhow::anyhow!("{e}"))?;
    let c_ref = CString::new(reference).context("OCI reference contains interior NUL byte")?;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();

    std::thread::spawn(move || {
        // Box the sender and convert to a raw pointer inside the thread so
        // the raw pointer never crosses a thread boundary from Rust's
        // perspective.  The Go callback may invoke it from any goroutine
        // thread, which is safe because UnboundedSender::send is threadsafe.
        let tx_ptr = Box::into_raw(Box::new(tx));
        unsafe {
            (lib.pull_stream)(
                c_ref.as_ptr(),
                Some(pull_stream_trampoline),
                tx_ptr as *mut c_void,
            );
            // Reclaim and drop the sender, closing the channel.
            drop(Box::from_raw(tx_ptr));
        }
    });

    Ok(OciStreamHandle { lines: rx })
}

/// List all OCI models in the local store.
///
/// Returns a list of `(tag, digest)` pairs.
#[allow(dead_code)]
pub fn oci_list_models() -> Result<Vec<(String, String)>> {
    let lib = load_oci_lib().map_err(|e| anyhow::anyhow!("{e}"))?;

    let result = unsafe { (lib.list)() };
    if result.is_null() {
        let err = get_last_oci_error(lib);
        anyhow::bail!("OCI list failed: {err}");
    }

    let raw = unsafe {
        let s = CStr::from_ptr(result).to_string_lossy().into_owned();
        (lib.free_string)(result);
        s
    };

    let entries = raw
        .lines()
        .filter_map(|line| {
            let (tag, id) = line.split_once('\t')?;
            Some((tag.to_string(), id.to_string()))
        })
        .collect();

    Ok(entries)
}

// ---------------------------------------------------------------------------
// CLI args
// ---------------------------------------------------------------------------

/// Default port for the Ollama-compatible API.
const DEFAULT_PORT: u16 = 17434;

#[derive(Parser, Clone)]
pub struct PullArgs {
    /// Model reference.
    ///
    /// Examples:
    ///   inferrs pull gemma3                     (docker.io/ai, OCI)
    ///   inferrs pull Qwen/Qwen3.5-0.8B          (HuggingFace)
    ///   inferrs pull docker.io/myorg/model:v1    (OCI registry)
    ///   inferrs pull hf.co/org/model:Q4_K_M      (HuggingFace) or a GGUF-only repo
    /// (e.g. ggml-org/gemma-4-E2B-it-GGUF).
    pub model: String,

    /// Git branch or tag on HuggingFace Hub (only for HF pulls)
    #[arg(long, default_value = "main")]
    pub revision: String,

    /// Specific GGUF filename to download from a GGUF-only repo.
    ///
    /// Only used when the repo contains GGUF files but no safetensors weights
    /// (e.g. ggml-org/gemma-4-E2B-it-GGUF).  When omitted, inferrs picks the
    /// best available quantization automatically (preferring Q4K, then Q8_0,
    /// then the first .gguf file found).
    #[arg(long, value_name = "FILENAME")]
    pub gguf_file: Option<String>,

    /// Optional HuggingFace repository to download tokenizer.json and config.json from
    /// (e.g. microsoft/Phi-4-reasoning-plus). Useful for GGUF-only repos that lack source metadata.
    #[arg(long, value_name = "REPO")]
    pub tokenizer_source: Option<String>,

    /// Quantize weights and cache the result as a GGUF file.
    ///
    /// Accepted formats (case-insensitive): Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
    /// Q2K, Q3K, Q4K (Q4_K_M), Q5K, Q6K.
    ///
    /// When used as a plain flag (`--quantize`) the default Q4_K_M (= Q4K) is used.
    #[arg(long, num_args(0..=1), default_missing_value("Q4K"), require_equals(true),
          value_name = "FORMAT")]
    pub quantize: Option<String>,

    // ── Server connection (OCI pulls only) ────────────────────────────────────
    /// Address of the `inferrs serve` daemon.
    /// Overrides `INFERRS_HOST`.  Defaults to `127.0.0.1`.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Port of the `inferrs serve` daemon.
    /// Overrides the port part of `INFERRS_HOST`.  Defaults to 17434.
    #[arg(long, default_value_t = DEFAULT_PORT)]
    pub port: u16,
}

impl PullArgs {
    /// Resolve the base URL for the server, matching the pattern used by
    /// `inferrs run` and `inferrs stop`.
    fn server_url(&self) -> String {
        let from_flags = self.host != "127.0.0.1" || self.port != DEFAULT_PORT;
        if from_flags {
            return format!("http://{}:{}", self.host, self.port);
        }

        if let Ok(env) = std::env::var("INFERRS_HOST") {
            let env = env.trim().to_string();
            if !env.is_empty() {
                return if env.starts_with("http://") || env.starts_with("https://") {
                    env
                } else {
                    format!("http://{env}")
                };
            }
        }

        format!("http://{}:{}", self.host, self.port)
    }
}

// ---------------------------------------------------------------------------
// Reference classification
// ---------------------------------------------------------------------------

/// Classify a model reference into OCI or HuggingFace.
#[derive(Debug, PartialEq)]
pub enum RefKind {
    /// Pull from an OCI registry (docker.io, custom registries).
    Oci,
    /// Pull from HuggingFace Hub.
    HuggingFace,
}

/// Determine whether a reference should go to an OCI registry or HuggingFace.
///
/// Rules (matching Docker Model Runner conventions):
///   - `hf.co/...` or `huggingface.co/...` → HuggingFace
///   - Single word (no `/`) → OCI.  The Go library is responsible for
///     expanding this to `docker.io/ai/<name>` when calling the registry.
///   - Has explicit registry (dot or colon before first `/`) → OCI
///   - `localhost/...` (no port) → OCI  (local registry)
///   - `org/model` (no dots before first `/`) → HuggingFace
pub fn classify_reference(reference: &str) -> RefKind {
    let reference = reference.trim();

    // Explicit HuggingFace prefixes
    if reference.starts_with("hf.co/") || reference.starts_with("huggingface.co/") {
        return RefKind::HuggingFace;
    }

    // Find the first slash
    if let Some(slash_pos) = reference.find('/') {
        let before_slash = &reference[..slash_pos];
        // If the part before the first slash contains a dot or colon,
        // it's an explicit registry → OCI
        if before_slash.contains('.') || before_slash.contains(':') {
            return RefKind::Oci;
        }
        // `localhost` without a port is also an OCI local registry
        if before_slash == "localhost" {
            return RefKind::Oci;
        }
        // Otherwise it's org/model → HuggingFace
        return RefKind::HuggingFace;
    }

    // No slash at all → single word → OCI (docker.io/ai/<name>)
    RefKind::Oci
}

// ---------------------------------------------------------------------------
// OCI operations (via FFI into the Go shared library)
// ---------------------------------------------------------------------------

/// Pull an OCI model and return its bundle path.
pub fn oci_pull_model(reference: &str) -> Result<PathBuf> {
    let lib = load_oci_lib().map_err(|e| anyhow::anyhow!("{e}"))?;
    let c_ref = CString::new(reference).context("OCI reference contains interior NUL byte")?;

    tracing::info!("Pulling OCI model: {}", reference);

    let result = unsafe { (lib.pull)(c_ref.as_ptr()) };

    if result.is_null() {
        let err = get_last_oci_error(lib);
        anyhow::bail!("OCI pull failed for '{}': {}", reference, err);
    }

    let path_str = unsafe {
        let s = CStr::from_ptr(result).to_string_lossy().into_owned();
        (lib.free_string)(result);
        s
    };

    if path_str.is_empty() {
        anyhow::bail!("OCI pull returned an empty bundle path for '{}'", reference);
    }

    Ok(PathBuf::from(path_str))
}

/// Look up an already-pulled OCI model's bundle path without pulling.
///
/// Returns `None` if the model is not in the local store.
pub fn oci_bundle_path(reference: &str) -> Option<PathBuf> {
    let lib = load_oci_lib().ok()?;
    let c_ref = CString::new(reference).ok()?;

    let result = unsafe { (lib.bundle)(c_ref.as_ptr()) };

    if result.is_null() {
        // Log a warning when the lookup fails so silent failures are visible.
        let err = get_last_oci_error(lib);
        tracing::warn!(
            "OCI bundle lookup failed for '{}': {}; will attempt pull",
            reference,
            err,
        );
        return None;
    }

    let path_str = unsafe {
        let s = CStr::from_ptr(result).to_string_lossy().into_owned();
        (lib.free_string)(result);
        s
    };

    if path_str.is_empty() {
        return None;
    }

    let p = PathBuf::from(&path_str);
    if p.exists() {
        Some(p)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// OCI pull via HTTP — NDJSON progress rendering
// ---------------------------------------------------------------------------

/// One NDJSON status object from `POST /api/pull` (Ollama-compatible).
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[allow(dead_code)]
struct PullStatus {
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    digest: Option<String>,
    total: Option<u64>,
    #[serde(default)]
    completed: Option<u64>,
    #[serde(default)]
    error: Option<String>,
}

/// Format a byte count as a human-readable string (e.g. "1.2 GB").
fn human_bytes(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;

    let b = bytes as f64;
    if b >= GB {
        format!("{:.1} GB", b / GB)
    } else if b >= MB {
        format!("{:.0} MB", b / MB)
    } else if b >= KB {
        format!("{:.0} KB", b / KB)
    } else {
        format!("{bytes} B")
    }
}

/// State for multi-line progress rendering (one line per layer digest).
struct PullProgress {
    /// Ordered list of digest strings we've seen so far.
    digests: Vec<String>,
    /// Current total number of output lines (status + progress lines).
    total_lines: usize,
}

impl PullProgress {
    fn new() -> Self {
        Self {
            digests: Vec::new(),
            total_lines: 0,
        }
    }

    /// Return the line index (0-based from top) for a digest, adding a new
    /// line if this digest hasn't been seen before.
    fn line_for_digest(&mut self, digest: &str, out: &mut impl Write) -> std::io::Result<usize> {
        if let Some(pos) = self.digests.iter().position(|d| d == digest) {
            return Ok(pos);
        }
        // New digest — allocate a new line at the bottom.
        let idx = self.digests.len();
        self.digests.push(digest.to_string());
        // Print a blank line for this new entry.
        writeln!(out)?;
        self.total_lines += 1;
        Ok(idx)
    }

    /// Move the cursor to a specific progress line, update it, then move back
    /// to the bottom.
    fn update_line(
        &self,
        line_idx: usize,
        content: &str,
        out: &mut impl Write,
    ) -> std::io::Result<()> {
        let bottom = self.total_lines - 1;
        let lines_up = bottom - line_idx;

        if lines_up > 0 {
            // Move cursor up N lines.
            write!(out, "\x1b[{lines_up}A")?;
        }
        // Overwrite the line.
        write!(out, "\r\x1b[2K{content}")?;
        if lines_up > 0 {
            // Move cursor back down.
            write!(out, "\x1b[{lines_up}B")?;
        }
        out.flush()
    }
}

/// Pull an OCI model via the server's `/api/pull` endpoint with streaming
/// progress, rendering Ollama-style multi-line output to the terminal.
async fn oci_pull_via_server(model: &str, base_url: &str) -> Result<()> {
    use futures::StreamExt;

    let client = reqwest::Client::new();

    // Ensure the daemon is up, auto-starting it when needed.
    crate::run::ensure_server_running(&client, base_url).await?;

    let url = format!("{base_url}/api/pull");
    let response = client
        .post(&url)
        .json(&serde_json::json!({
            "model": model,
            "stream": true,
        }))
        .send()
        .await
        .with_context(|| format!("POST {url} failed"))?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("Server returned {status}: {text}");
    }

    // Drain the NDJSON stream and render multi-line progress.
    // Only use ANSI escape codes when stderr is an interactive terminal;
    // fall back to plain line-based output otherwise (e.g. log files).
    use std::io::IsTerminal;
    let is_tty = std::io::stderr().is_terminal();
    let mut out = std::io::stderr();
    let mut byte_stream = response.bytes_stream();
    let mut line_buf = String::new();
    let mut progress = PullProgress::new();

    while let Some(chunk) = byte_stream.next().await {
        let chunk = chunk.context("Error reading pull response stream")?;
        let text = std::str::from_utf8(&chunk).context("Non-UTF-8 bytes in pull response")?;
        line_buf.push_str(text);

        while let Some(newline_pos) = line_buf.find('\n') {
            let line = line_buf[..newline_pos].trim().to_string();
            line_buf.drain(..=newline_pos);

            if line.is_empty() {
                continue;
            }

            let status: PullStatus = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("Warning: failed to parse pull progress: {e}");
                    continue;
                }
            };

            // Check for errors.
            if let Some(ref err) = status.error {
                writeln!(out)?;
                anyhow::bail!("Pull failed: {err}");
            }

            // Progress update for a specific layer (has digest + total > 0).
            if let (Some(ref digest), Some(total), Some(completed)) =
                (&status.digest, status.total, status.completed)
            {
                if total > 0 && !digest.is_empty() {
                    if is_tty {
                        let line_idx = progress.line_for_digest(digest, &mut out)?;

                        let pct = (completed as f64 / total as f64 * 100.0).min(100.0);
                        let bar_width = 20;
                        let filled = (pct / 100.0 * bar_width as f64) as usize;
                        let empty = bar_width - filled;

                        let short = short_digest(digest);
                        let bar = format!(
                            "pulling {short}  {pct:5.1}% ▕{}{}▏ {}/{}",
                            "█".repeat(filled),
                            "░".repeat(empty),
                            human_bytes(completed),
                            human_bytes(total),
                        );
                        progress.update_line(line_idx, &bar, &mut out)?;
                    }
                    // Non-TTY: skip per-chunk progress to avoid flooding logs;
                    // status-only lines below still print completion milestones.
                    continue;
                }
            }

            // Status-only line (no progress bar) — print at the bottom.
            if let Some(ref msg) = status.status {
                if is_tty {
                    writeln!(out, "\r\x1b[2K{msg}")?;
                    progress.total_lines += 1;
                } else {
                    writeln!(out, "{msg}")?;
                }
                out.flush()?;
            }
        }
    }

    Ok(())
}

/// Shorten a digest for display (e.g. "sha256:abcdef123456..." → "abcdef123456").
fn short_digest(digest: &str) -> &str {
    const PREFIX: &str = "sha256:";
    const SHORT_LEN: usize = 12;

    let s = digest.strip_prefix(PREFIX).unwrap_or(digest);
    if s.len() > SHORT_LEN {
        &s[..SHORT_LEN]
    } else {
        s
    }
}

// ---------------------------------------------------------------------------
// `inferrs pull` entry point
// ---------------------------------------------------------------------------

pub async fn run(args: PullArgs) -> Result<()> {
    match classify_reference(&args.model) {
        RefKind::Oci => {
            // --revision is only meaningful for HuggingFace references.
            if args.revision != "main" {
                anyhow::bail!(
                    "--revision is not supported for OCI references \
                     (got --revision '{}' for OCI model '{}'). \
                     Use an OCI tag instead, e.g. docker.io/org/model:v2",
                    args.revision,
                    args.model,
                );
            }

            let base_url = args.server_url();
            oci_pull_via_server(&args.model, &base_url).await?;
        }
        RefKind::HuggingFace => {
            // Strip explicit HF prefixes for the HF Hub API
            let hf_model = args
                .model
                .strip_prefix("hf.co/")
                .or_else(|| args.model.strip_prefix("huggingface.co/"))
                .unwrap_or(&args.model);

            let quant_dtype = args
                .quantize
                .as_deref()
                .map(crate::quantize::parse_format)
                .transpose()?;

            let files = crate::hub::download_and_maybe_quantize(
                hf_model,
                &args.revision,
                args.gguf_file.as_deref(),
                args.tokenizer_source.as_deref(),
                quant_dtype,
            )?;

            println!("Pulled {} (HuggingFace)", args.model);
            println!("  config:    {}", files.config_path.display());
            println!("  tokenizer: {}", files.tokenizer_path.display());
            for w in &files.weight_paths {
                println!("  weights:   {}", w.display());
            }
            if let Some(gguf) = &files.gguf_path {
                println!("  gguf:      {}", gguf.display());
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_reference() {
        // Single word → OCI (docker.io/ai)
        assert_eq!(classify_reference("gemma3"), RefKind::Oci);
        assert_eq!(classify_reference("llama"), RefKind::Oci);
        assert_eq!(classify_reference("gemma3:latest"), RefKind::Oci);

        // org/model → HuggingFace
        assert_eq!(
            classify_reference("Qwen/Qwen3.5-0.8B"),
            RefKind::HuggingFace
        );
        assert_eq!(classify_reference("myorg/mymodel"), RefKind::HuggingFace);

        // Explicit HF prefixes → HuggingFace
        assert_eq!(classify_reference("hf.co/org/model"), RefKind::HuggingFace);
        assert_eq!(
            classify_reference("huggingface.co/org/model:Q4_K_M"),
            RefKind::HuggingFace
        );

        // Explicit registry → OCI
        assert_eq!(
            classify_reference("docker.io/ai/gemma3:latest"),
            RefKind::Oci
        );
        assert_eq!(
            classify_reference("registry.example.com/org/model:v1"),
            RefKind::Oci
        );
        assert_eq!(classify_reference("docker.io/myorg/mymodel"), RefKind::Oci);
        assert_eq!(classify_reference("localhost:5000/model"), RefKind::Oci);

        // localhost without a port → OCI (local registry)
        assert_eq!(classify_reference("localhost/model"), RefKind::Oci);
    }

    #[test]
    fn test_normalize_oci_reference_matches_dmr_defaults() {
        assert_eq!(normalize_oci_reference("gemma3"), "ai/gemma3:latest");
        assert_eq!(normalize_oci_reference("AI/Gemma3"), "ai/gemma3:latest");
        assert_eq!(
            normalize_oci_reference("registry.example.com/MyOrg/Model:Q4_K_M"),
            "registry.example.com/myorg/model:Q4_K_M"
        );
    }
}
