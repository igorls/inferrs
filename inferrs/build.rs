//! Build script for inferrs.
//!
//! 1. **Web UI compression** — compress `ui/index.html` into `$OUT_DIR/ui.html.gz`
//!    using best-level gzip compression.  The server embeds the result via
//!    `include_bytes!(concat!(env!("OUT_DIR"), "/ui.html.gz"))` and serves it
//!    with `Content-Encoding: gzip` in daemon mode (no model argument).
//!    The build fails loudly if the compressed output exceeds 1 MiB so that the
//!    size budget is enforced at compile time rather than discovered at runtime.
//!
//! 2. **OCI shared library linking** — links the Go OCI shared library
//!    (`libocipull`) built by `make oci-lib`.  This script tells rustc where to
//!    find it and sets rpath so the binary can locate the library at runtime.

use flate2::{write::GzEncoder, Compression};
use std::{env, fs, io::Write, path::PathBuf};

const SIZE_LIMIT_BYTES: u64 = 1024 * 1024; // 1 MiB

fn main() {
    // -----------------------------------------------------------------------
    // 1. Web UI compression
    // -----------------------------------------------------------------------

    // Re-run this script if the UI source changes.
    println!("cargo:rerun-if-changed=ui/index.html");

    let out_dir: PathBuf = env::var("OUT_DIR").expect("OUT_DIR not set").into();
    let gz_path = out_dir.join("ui.html.gz");

    let html = fs::read("ui/index.html")
        .expect("failed to read inferrs/ui/index.html – run from the crate root");

    let file = fs::File::create(&gz_path).expect("failed to create ui.html.gz in OUT_DIR");
    let mut encoder = GzEncoder::new(file, Compression::best());
    encoder
        .write_all(&html)
        .expect("failed to compress ui/index.html");
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
        "cargo:warning=inferrs web UI compressed to {:.1} KiB (budget: 1024 KiB)",
        compressed_size as f64 / 1024.0
    );

    // -----------------------------------------------------------------------
    // 2. OCI shared library (libocipull) linking
    // -----------------------------------------------------------------------

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let workspace_root = PathBuf::from(&manifest_dir)
        .parent()
        .expect("inferrs crate must be inside a workspace")
        .to_path_buf();

    let profile = env::var("PROFILE").unwrap(); // "debug" or "release"
    let target_dir = workspace_root.join("target").join(&profile);

    // Tell rustc where to find libocipull.{dylib,so}.
    println!("cargo:rustc-link-search=native={}", target_dir.display());
    println!("cargo:rustc-link-lib=dylib=ocipull");

    // Set rpath so the binary can find the library next to the executable
    // at runtime without needing LD_LIBRARY_PATH / DYLD_LIBRARY_PATH.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "macos" => {
            println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
        }
        "linux" => {
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
        }
        _ => {}
    }

    // Re-run if the library is rebuilt.
    println!(
        "cargo:rerun-if-changed={}",
        target_dir.join("libocipull.dylib").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        target_dir.join("libocipull.so").display()
    );
}
