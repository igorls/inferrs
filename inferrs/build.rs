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
use std::{
    env, fs,
    io::Write,
    path::{Component, PathBuf},
};

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

    // Belt-and-suspenders: after bundling, no local `<link>` or `<script src=>`
    // referencing files under `ui/` should remain.  If one slips through (e.g.
    // an attribute order this script's parser missed), the runtime UI breaks
    // silently because the daemon doesn't serve those files.  Fail the build
    // instead of shipping a broken bundle.
    assert_no_unbundled_local_assets(&bundled);

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
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rerun_on_dir(&path);
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

/// Replace every local stylesheet `<link>` and `<script src>` reference with
/// the inlined contents of the target file.  External URLs (`http://`,
/// `https://`, `//…`, `data:`, absolute paths) are left untouched.
///
/// The matcher is tolerant of attribute order: `<link href="x" rel="stylesheet">`
/// works just as well as `<link rel="stylesheet" href="x">`, and arbitrary
/// whitespace / extra attributes between them are fine.
fn inline_assets(html: &str) -> String {
    let html = inline_tag(html, "link", |attrs| {
        let rel = get_attr(attrs, "rel")?;
        if !rel.split_ascii_whitespace().any(|t| t == "stylesheet") {
            return None;
        }
        let href = get_attr(attrs, "href")?;
        Some(("style", href))
    });
    inline_tag(&html, "script", |attrs| {
        let src = get_attr(attrs, "src")?;
        Some(("script", src))
    })
}

/// Walk `html` looking for `<TAG ...>` (tag matched case-insensitively).  For
/// each match, hand the attribute string to `decide` which returns either
/// `None` (leave the tag alone) or `Some((wrap_tag, url))` to replace it with
/// `<wrap_tag>{file_contents}</wrap_tag>`.
///
/// For `<script>` tags we additionally consume the matching `</script>`.
fn inline_tag<F>(html: &str, tag: &str, decide: F) -> String
where
    F: Fn(&str) -> Option<(&'static str, &str)>,
{
    let needle_lower = format!("<{}", tag.to_ascii_lowercase());
    let mut out = String::with_capacity(html.len());
    let mut cursor = 0;

    while let Some(rel_idx) = find_ignore_ascii_case(&html[cursor..], &needle_lower) {
        let abs_start = cursor + rel_idx;
        // Make sure the match is followed by whitespace or `>` (so we don't
        // catch e.g. `<scripted>` while looking for `<script`).
        let after_name = abs_start + needle_lower.len();
        let next = html[after_name..].chars().next();
        if !matches!(next, Some(c) if c.is_ascii_whitespace() || c == '>' || c == '/') {
            out.push_str(&html[cursor..after_name]);
            cursor = after_name;
            continue;
        }
        // Locate the end of the opening tag.  Naive search for `>` is fine for
        // our hand-authored HTML — no `<` inside attribute values.
        let Some(gt_rel) = html[after_name..].find('>') else {
            out.push_str(&html[abs_start..]);
            return out;
        };
        let tag_open_end = after_name + gt_rel; // index of '>'
        let attrs = &html[after_name..tag_open_end];

        // For `<script>`, also consume the matching closing tag.
        let tag_full_end = if tag.eq_ignore_ascii_case("script") {
            let after_open = tag_open_end + 1;
            match find_ignore_ascii_case(&html[after_open..], "</script>") {
                Some(close_rel) => after_open + close_rel + "</script>".len(),
                None => {
                    out.push_str(&html[abs_start..]);
                    return out;
                }
            }
        } else {
            tag_open_end + 1
        };

        out.push_str(&html[cursor..abs_start]);

        match decide(attrs) {
            Some((wrap_tag, url)) if !is_external(url) => {
                let safe_url = sanitize_local_url(url);
                let path = PathBuf::from(UI_DIR).join(&safe_url);
                let body = fs::read_to_string(&path).unwrap_or_else(|e| {
                    panic!("failed to inline UI asset {}: {e}", path.display())
                });
                out.push_str(&format!(
                    "<{wrap_tag}>\n{}\n</{wrap_tag}>",
                    body.trim_end_matches('\n')
                ));
            }
            // External URL or tag we don't transform — leave verbatim.
            _ => out.push_str(&html[abs_start..tag_full_end]),
        }

        cursor = tag_full_end;
    }

    out.push_str(&html[cursor..]);
    out
}

/// Reject anything that could escape the `ui/` directory at build time.  An
/// absolute path or a `..` component would, after `PathBuf::join`, point
/// outside the source tree — letting a tampered `index.html` silently embed
/// arbitrary host files into the binary in CI.  Treat such inputs as a hard
/// build failure.
fn sanitize_local_url(url: &str) -> PathBuf {
    let trimmed = url.split(['?', '#']).next().unwrap_or(url);
    let path = PathBuf::from(trimmed);
    for c in path.components() {
        match c {
            Component::Normal(_) => {}
            Component::CurDir => {}
            _ => panic!(
                "refusing to inline UI asset with non-normal path component: {url} \
                 (only relative paths under {UI_DIR}/ are allowed)"
            ),
        }
    }
    path
}

fn is_external(url: &str) -> bool {
    url.starts_with("http://")
        || url.starts_with("https://")
        || url.starts_with("//")
        || url.starts_with("data:")
        // Absolute paths would be served by the daemon at the URL given, not
        // inlined — treat as external so we leave them in place.
        || url.starts_with('/')
}

/// Read the value of `attr_name` from a tag's attribute string, accepting
/// double- or single-quoted values and arbitrary attribute order.  Case-
/// insensitive on the attribute name.
fn get_attr<'a>(attrs: &'a str, attr_name: &str) -> Option<&'a str> {
    let lower = attrs.to_ascii_lowercase();
    let needle = format!("{}=", attr_name.to_ascii_lowercase());
    let mut search_from = 0;
    loop {
        let rel = lower[search_from..].find(&needle)?;
        let pos = search_from + rel;
        // Make sure this is actually an attribute boundary, not the tail of
        // another attribute name (e.g. `data-href=` matching `href=`).
        let prev = if pos == 0 {
            None
        } else {
            attrs[..pos].chars().next_back()
        };
        if matches!(prev, Some(c) if c.is_ascii_alphanumeric() || c == '-' || c == '_') {
            search_from = pos + needle.len();
            continue;
        }
        let after_eq = pos + needle.len();
        let bytes = attrs.as_bytes();
        if after_eq >= bytes.len() {
            return None;
        }
        let quote = bytes[after_eq];
        if quote != b'"' && quote != b'\'' {
            return None;
        }
        let value_start = after_eq + 1;
        let close_rel = attrs[value_start..].find(quote as char)?;
        return Some(&attrs[value_start..value_start + close_rel]);
    }
}

fn find_ignore_ascii_case(haystack: &str, needle: &str) -> Option<usize> {
    let lower = haystack.to_ascii_lowercase();
    lower.find(&needle.to_ascii_lowercase())
}

/// Belt-and-suspenders: ensure no `<link rel="stylesheet" href="x">` or
/// `<script src="x"></script>` for a local URL slips through the bundler.
/// If one does, the daemon doesn't serve those paths and the UI ends up
/// unstyled / non-interactive — a silent failure that's much easier to catch
/// at build time.
fn assert_no_unbundled_local_assets(bundled: &str) {
    let lower = bundled.to_ascii_lowercase();
    for (tag, attr) in [("link", "href"), ("script", "src")] {
        let mut cursor = 0;
        let needle = format!("<{tag}");
        while let Some(rel) = lower[cursor..].find(&needle) {
            let pos = cursor + rel;
            let after = pos + needle.len();
            let next = bundled[after..].chars().next();
            if !matches!(next, Some(c) if c.is_ascii_whitespace() || c == '>' || c == '/') {
                cursor = after;
                continue;
            }
            let Some(gt_rel) = bundled[after..].find('>') else {
                break;
            };
            let attrs = &bundled[after..after + gt_rel];
            if let Some(value) = get_attr(attrs, attr) {
                if !is_external(value) {
                    panic!(
                        "build.rs bundler missed a local <{tag} {attr}=\"{value}\">; \
                         the runtime daemon doesn't serve files at this path. \
                         Check the tag in ui/index.html for an unusual attribute layout."
                    );
                }
            }
            cursor = after + gt_rel + 1;
        }
    }
}
