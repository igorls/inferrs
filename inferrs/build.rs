//! Build script for inferrs.
//!
//! **Web UI bundling** — read `ui/index.html`, inline every local
//! `<link rel="stylesheet" href="…">` and `<script src="…"></script>` reference
//! by reading the target file from disk and replacing the tag with an inline
//! `<style>` or `<script>` block.  External URLs (anything not a local relative
//! path under `ui/`) are left untouched.  The resulting single-file HTML is
//! gzipped with best-level compression and embedded into the binary via
//! `include_bytes!(concat!(env!("OUT_DIR"), "/ui.html.gz"))`.
//!
//! This split (HTML / CSS / JS / vendor libraries) gives much better DX while
//! keeping the runtime delivery path identical: one gzipped blob, served from
//! `GET /` in daemon mode with `Content-Encoding: gzip`.
//!
//! The build fails loudly if the compressed bundle exceeds 1 MiB so the size
//! budget is enforced at compile time rather than discovered at runtime.
//!
//! Note: the Go OCI shared library (`libocimodels`) is loaded on demand via
//! `dlopen` at runtime (see `pull.rs`), so no link-time configuration is
//! needed here.

use flate2::{write::GzEncoder, Compression};
use std::{env, fs, io::Write, path::PathBuf};

const SIZE_LIMIT_BYTES: u64 = 1024 * 1024; // 1 MiB
const UI_DIR: &str = "ui";
const ENTRY_HTML: &str = "ui/index.html";

fn main() {
    let out_dir: PathBuf = env::var("OUT_DIR").expect("OUT_DIR not set").into();
    let gz_path = out_dir.join("ui.html.gz");

    // Re-run on any UI source change.  The walk is shallow enough to enumerate
    // up front; cargo deduplicates.
    rerun_on_dir(UI_DIR.as_ref());

    let html = fs::read_to_string(ENTRY_HTML)
        .unwrap_or_else(|e| panic!("failed to read {ENTRY_HTML}: {e}"));

    let bundled = inline_assets(&html);

    let file = fs::File::create(&gz_path).expect("failed to create ui.html.gz in OUT_DIR");
    let mut encoder = GzEncoder::new(file, Compression::best());
    encoder
        .write_all(bundled.as_bytes())
        .expect("failed to compress UI bundle");
    encoder.finish().expect("failed to finalise gzip stream");

    let compressed_size = fs::metadata(&gz_path)
        .expect("failed to stat ui.html.gz")
        .len();

    assert!(
        compressed_size <= SIZE_LIMIT_BYTES,
        "ui.html.gz is {compressed_size} bytes ({:.1} KiB), which exceeds the 1 MiB limit. \
         Reduce the UI size before building.",
        compressed_size as f64 / 1024.0
    );

    println!(
        "cargo:warning=inferrs web UI bundled to {:.1} KiB (raw: {:.1} KiB, budget: 1024 KiB)",
        compressed_size as f64 / 1024.0,
        bundled.len() as f64 / 1024.0
    );
}

/// Walk `dir` recursively and emit `cargo:rerun-if-changed` for every entry.
fn rerun_on_dir(dir: &std::path::Path) {
    println!("cargo:rerun-if-changed={}", dir.display());
    let Ok(entries) = fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rerun_on_dir(&path);
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

/// Replace every `<link rel="stylesheet" href="…">` and `<script src="…"></script>`
/// pointing to a local file under `ui/` with the inlined contents of that file.
/// External URLs (`http://`, `https://`, `//…`, `data:`) are left alone.
fn inline_assets(html: &str) -> String {
    // Two passes: stylesheet links, then script tags.
    let html = inline_tag(
        html,
        // <link rel="stylesheet" href="..."> — order-tolerant: rel before href
        r#"<link rel="stylesheet" href=""#,
        r#"">"#,
        |body| format!("<style>\n{body}\n</style>"),
    );
    inline_tag(
        &html,
        r#"<script src=""#,
        r#"></script>"#,
        |body| format!("<script>\n{body}\n</script>"),
    )
}

/// Find every `prefix + path + suffix` occurrence and substitute the file at
/// `path` (relative to `ui/`) using `wrap` to format the final block.
fn inline_tag<F: Fn(&str) -> String>(
    html: &str,
    prefix: &str,
    suffix: &str,
    wrap: F,
) -> String {
    let mut out = String::with_capacity(html.len());
    let mut cursor = 0;

    while let Some(start) = html[cursor..].find(prefix) {
        let abs_start = cursor + start;
        out.push_str(&html[cursor..abs_start]);

        let after_prefix = abs_start + prefix.len();
        // Locate the closing `"` that ends the URL value.
        let Some(quote_end_rel) = html[after_prefix..].find('"') else {
            // Malformed tag — leave the rest alone and bail.
            out.push_str(&html[abs_start..]);
            return out;
        };
        let quote_end = after_prefix + quote_end_rel;
        let url = &html[after_prefix..quote_end];

        // Find the suffix that closes the tag (`">` or `></script>`).
        let Some(suffix_rel) = html[quote_end..].find(suffix) else {
            out.push_str(&html[abs_start..]);
            return out;
        };
        let after_tag = quote_end + suffix_rel + suffix.len();

        if is_external(url) {
            // Leave external references untouched.
            out.push_str(&html[abs_start..after_tag]);
        } else {
            let path = PathBuf::from(UI_DIR).join(url);
            let body = fs::read_to_string(&path).unwrap_or_else(|e| {
                panic!("failed to inline UI asset {}: {e}", path.display())
            });
            out.push_str(&wrap(body.trim_end_matches('\n')));
        }

        cursor = after_tag;
    }

    out.push_str(&html[cursor..]);
    out
}

fn is_external(url: &str) -> bool {
    url.starts_with("http://")
        || url.starts_with("https://")
        || url.starts_with("//")
        || url.starts_with("data:")
}
