//! GPU/NPU backend discovery via dynamic loading (`dlopen` on Linux/Android,
//! `LoadLibrary` on Windows).
//!
//! The `inferrs` binary is compiled with the `cuda` feature (so
//! `Device::new_cuda()` is available) but candle-core is patched to use
//! cudarc's `fallback-dynamic-loading` instead of `dynamic-linking`.  This
//! means CUDA/cuBLAS/cuRAND libraries are **not** hard-linked into the binary;
//! they are opened on demand when a CUDA device is first used.
//!
//! At startup the binary searches for backend plugin files alongside the
//! running executable and probes each one in priority order.  Each plugin
//! exports a single C-ABI function:
//!
//! ```c
//! int inferrs_backend_probe(void);  // 0 = available, non-zero = not available
//! ```
//!
//! If a probe succeeds the matching `candle_core::Device` variant is returned.
//! The caller (`resolve_device`) uses this to construct the actual device.
//!
//! ## Platform support matrix
//!
//! | Platform                  | CUDA | ROCm | CANN | Vulkan |
//! |---------------------------|------|------|------|--------|
//! | Linux x86_64              | ✓    | ✓    | ✓    | ✓      |
//! | Linux aarch64             | ✓    | ✓    | ✓    | ✓      |
//! | Windows x86_64            | ✓    | ✓    | —    | ✓      |
//! | Windows aarch64           | —    | —    | —    | —      |
//! | macOS aarch64             | —    | —    | —    | —      |
//! | Android aarch64           | —    | —    | ✓    | —      |
//!
//! ROCm on Windows is supported from ROCm 5.5+ (HIP SDK for Windows).
//! ROCm on Linux aarch64 is supported on hardware such as AMD MI300A APUs
//! and Radeon-equipped AArch64 platforms.
//! CANN (Huawei Ascend NPU) is not supported on Windows (Huawei SDK constraint).
//!
//! Plugin search order (highest priority first):
//!
//! **Linux x86_64 / aarch64:**
//!   1. CUDA   (`.so`)  → `Device::new_cuda(0)`
//!   2. ROCm   (`.so`)  → `Device::new_cuda(0)` (HIP)
//!   3. CANN   (`.so`)  → CPU fallback with info log (pending candle CANN Device)
//!   4. Vulkan (`.so`)  → CPU fallback with info log
//!   5. CPU    (always available)
//!
//! **Windows x86_64:**
//!   1. CUDA   (`.dll`) → `Device::new_cuda(0)`
//!   2. ROCm   (`.dll`) → `Device::new_cuda(0)` (HIP SDK for Windows)
//!   3. Vulkan (`.dll`) → CPU fallback with info log
//!   4. CPU    (always available)
//!
//! **Android aarch64:**
//!   1. CANN   (`.so`)  → CPU fallback with info log (pending candle CANN Device)
//!   2. CPU    (always available)
//!
//! **macOS / Windows ARM:**
//!   No plugin system needed — Metal is linked directly on macOS;
//!   CUDA/ROCm/CANN are unavailable on Windows ARM.

// ── Linux ────────────────────────────────────────────────────────────────────

#[cfg(target_os = "linux")]
mod linux {
    use std::path::PathBuf;

    use libloading::{Library, Symbol};

    /// The detected GPU/NPU backend, in priority order.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum BackendKind {
        Cuda,
        Rocm,
        /// Huawei Ascend NPU via CANN (Compute Architecture for Neural Networks).
        ///
        /// candle-core does not yet have a native CANN `Device` variant.
        /// A successful probe causes an info-level log message; the binary
        /// then falls back to CPU.  Full acceleration will be enabled once
        /// candle integrates CANN support.
        ///
        /// Supported CANN SDK architectures: x86_64, aarch64.
        Cann,
        /// Vulkan is detected but candle 0.8 has no Vulkan Device variant yet.
        /// Falls back to CPU while logging the detection.
        Vulkan,
        Cpu,
    }

    /// Probe the backend plugins and return the highest-priority available kind.
    ///
    /// The plugins are searched next to the running executable first, then in
    /// the same directory as the executable's parent (for dev builds under
    /// `target/<target>/release/`), and finally in system-wide locations.
    pub fn detect_backend() -> BackendKind {
        let search_dirs = plugin_search_dirs();

        // Priority order: CUDA → ROCm → CANN → Vulkan → CPU.
        // Both x86_64 and aarch64 Linux support CUDA and ROCm.
        //
        // CANN (Huawei Ascend NPU) is placed after CUDA and ROCm so that a
        // system with both an NVIDIA/AMD GPU and an Ascend NPU prefers the
        // GPU.  CANN is placed before Vulkan because it represents dedicated
        // neural-network silicon rather than a general graphics API.
        //
        // The CANN plugin is arch-gated at build time (x86_64 / aarch64 only)
        // so on unsupported architectures the `.so` simply won't exist.
        let candidates: &[(&str, BackendKind)] = &[
            ("libinferrs_backend_cuda.so", BackendKind::Cuda),
            ("libinferrs_backend_rocm.so", BackendKind::Rocm),
            ("libinferrs_backend_cann.so", BackendKind::Cann),
            ("libinferrs_backend_vulkan.so", BackendKind::Vulkan),
        ];

        for (lib_name, kind) in candidates {
            if probe_plugin(&search_dirs, lib_name) {
                return *kind;
            }
        }

        BackendKind::Cpu
    }

    /// Try to load `lib_name` from each search directory and call
    /// `inferrs_backend_probe()`.  Returns `true` if the probe returns 0.
    fn probe_plugin(search_dirs: &[PathBuf], lib_name: &str) -> bool {
        type ProbeFn = unsafe extern "C" fn() -> i32;

        for dir in search_dirs {
            let path = dir.join(lib_name);
            if !path.exists() {
                continue;
            }

            // SAFETY: We are loading a well-known plugin whose ABI we control.
            let lib = match unsafe { Library::new(&path) } {
                Ok(l) => l,
                Err(e) => {
                    tracing::debug!("Failed to load {}: {e}", path.display());
                    continue;
                }
            };

            // SAFETY: We know the exported symbol name and its signature.
            let probe: Symbol<ProbeFn> = match unsafe { lib.get(b"inferrs_backend_probe\0") } {
                Ok(sym) => sym,
                Err(e) => {
                    tracing::debug!("Symbol not found in {}: {e}", path.display());
                    continue;
                }
            };

            // SAFETY: Calling a C function with no arguments is safe.
            let result = unsafe { probe() };
            if result == 0 {
                tracing::debug!("Backend probe succeeded: {}", path.display());
                // Keep `lib` alive until probe result is used — it's dropped here
                // which is fine because we only needed the probe call.
                drop(lib);
                return true;
            }
            tracing::debug!(
                "Backend probe returned {result} (unavailable): {}",
                path.display()
            );
        }

        false
    }

    /// Directories to search for backend plugins, in priority order.
    fn plugin_search_dirs() -> Vec<PathBuf> {
        let mut dirs: Vec<PathBuf> = Vec::new();

        // 1. Same directory as the running executable.
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                dirs.push(parent.to_path_buf());
            }
        }

        // 2. System-wide install locations.
        dirs.push(PathBuf::from("/usr/lib/inferrs"));
        dirs.push(PathBuf::from("/usr/local/lib/inferrs"));

        dirs
    }
}

#[cfg(target_os = "linux")]
pub use linux::{detect_backend, BackendKind};

// ── Android ──────────────────────────────────────────────────────────────────
// Android aarch64 hosts Ascend NPUs in some Huawei embedded/edge devices.
// Only CANN is probed; CUDA and ROCm are not available on Android.

#[cfg(target_os = "android")]
mod android {
    use std::path::PathBuf;

    use libloading::{Library, Symbol};

    /// The detected NPU backend on Android.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum BackendKind {
        /// Huawei Ascend NPU via CANN.
        ///
        /// Only aarch64 is supported.  The CANN plugin is compiled with an
        /// arch guard so it is absent on other Android ABIs.
        Cann,
        Cpu,
    }

    /// Probe the CANN plugin and return the detected backend.
    pub fn detect_backend() -> BackendKind {
        let search_dirs = plugin_search_dirs();

        // Android only supports CANN among the current plugin backends.
        let candidates: &[(&str, BackendKind)] =
            &[("libinferrs_backend_cann.so", BackendKind::Cann)];

        for (lib_name, kind) in candidates {
            if probe_plugin(&search_dirs, lib_name) {
                return *kind;
            }
        }

        BackendKind::Cpu
    }

    fn probe_plugin(search_dirs: &[PathBuf], lib_name: &str) -> bool {
        type ProbeFn = unsafe extern "C" fn() -> i32;

        for dir in search_dirs {
            let path = dir.join(lib_name);
            if !path.exists() {
                continue;
            }

            let lib = match unsafe { Library::new(&path) } {
                Ok(l) => l,
                Err(e) => {
                    tracing::debug!("Failed to load {}: {e}", path.display());
                    continue;
                }
            };

            let probe: Symbol<ProbeFn> = match unsafe { lib.get(b"inferrs_backend_probe\0") } {
                Ok(sym) => sym,
                Err(e) => {
                    tracing::debug!("Symbol not found in {}: {e}", path.display());
                    continue;
                }
            };

            let result = unsafe { probe() };
            if result == 0 {
                tracing::debug!("Backend probe succeeded: {}", path.display());
                drop(lib);
                return true;
            }
            tracing::debug!(
                "Backend probe returned {result} (unavailable): {}",
                path.display()
            );
        }

        false
    }

    fn plugin_search_dirs() -> Vec<PathBuf> {
        let mut dirs: Vec<PathBuf> = Vec::new();

        // 1. Same directory as the running executable.
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                dirs.push(parent.to_path_buf());
            }
        }

        // 2. Common Android data-app directories (for sideloaded builds).
        dirs.push(PathBuf::from("/data/local/tmp/inferrs"));

        dirs
    }
}

#[cfg(target_os = "android")]
pub use android::{detect_backend, BackendKind};

// ── Windows x86_64 ───────────────────────────────────────────────────────────
// CUDA and ROCm are not available on Windows ARM, so the plugin system is
// x86_64-only.  ROCm on Windows is supported from ROCm 5.5+ (HIP SDK for
// Windows); the plugin DLL is named `inferrs_backend_rocm.dll` and follows
// the same ABI as the Linux `.so`.
// CANN is not supported on Windows (Huawei SDK constraint).

#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
mod windows {
    use std::path::PathBuf;

    use libloading::{Library, Symbol};

    /// The detected GPU backend, in priority order.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum BackendKind {
        Cuda,
        /// ROCm/HIP device (AMD GPU via ROCm 5.5+ HIP SDK for Windows).
        Rocm,
        /// Vulkan is detected but candle 0.8 has no Vulkan Device variant yet.
        /// Falls back to CPU while logging the detection.
        Vulkan,
        Cpu,
    }

    /// Probe the backend plugins and return the highest-priority available kind.
    ///
    /// The plugins are searched next to the running executable first, then in
    /// `%ProgramFiles%\inferrs`.
    pub fn detect_backend() -> BackendKind {
        let search_dirs = plugin_search_dirs();

        // Priority order: CUDA → ROCm → Vulkan → CPU.
        // ROCm on Windows x86_64 is supported via AMD's HIP SDK (ROCm 5.5+).
        // CANN is not supported on Windows (Huawei SDK constraint).
        let candidates: &[(&str, BackendKind)] = &[
            ("inferrs_backend_cuda.dll", BackendKind::Cuda),
            ("inferrs_backend_rocm.dll", BackendKind::Rocm),
            ("inferrs_backend_vulkan.dll", BackendKind::Vulkan),
        ];

        for (lib_name, kind) in candidates {
            if probe_plugin(&search_dirs, lib_name) {
                return *kind;
            }
        }

        BackendKind::Cpu
    }

    /// Try to load `lib_name` from each search directory and call
    /// `inferrs_backend_probe()`.  Returns `true` if the probe returns 0.
    fn probe_plugin(search_dirs: &[PathBuf], lib_name: &str) -> bool {
        type ProbeFn = unsafe extern "C" fn() -> i32;

        for dir in search_dirs {
            let path = dir.join(lib_name);
            if !path.exists() {
                continue;
            }

            // SAFETY: We are loading a well-known plugin whose ABI we control.
            let lib = match unsafe { Library::new(&path) } {
                Ok(l) => l,
                Err(e) => {
                    tracing::debug!("Failed to load {}: {e}", path.display());
                    continue;
                }
            };

            // SAFETY: We know the exported symbol name and its signature.
            let probe: Symbol<ProbeFn> = match unsafe { lib.get(b"inferrs_backend_probe\0") } {
                Ok(sym) => sym,
                Err(e) => {
                    tracing::debug!("Symbol not found in {}: {e}", path.display());
                    continue;
                }
            };

            // SAFETY: Calling a C function with no arguments is safe.
            let result = unsafe { probe() };
            if result == 0 {
                tracing::debug!("Backend probe succeeded: {}", path.display());
                drop(lib);
                return true;
            }
            tracing::debug!(
                "Backend probe returned {result} (unavailable): {}",
                path.display()
            );
        }

        false
    }

    /// Directories to search for backend plugins, in priority order.
    fn plugin_search_dirs() -> Vec<PathBuf> {
        let mut dirs: Vec<PathBuf> = Vec::new();

        // 1. Same directory as the running executable.
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                dirs.push(parent.to_path_buf());
            }
        }

        // 2. System-wide install location.
        if let Ok(pf) = std::env::var("ProgramFiles") {
            dirs.push(PathBuf::from(pf).join("inferrs"));
        }

        dirs
    }
}

#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
pub use windows::{detect_backend, BackendKind};

// ── macOS (and any other unsupported platform) ────────────────────────────────
//
// On macOS and Windows ARM, no plugin system is needed:
//   - macOS: Metal is linked directly via the `metal` feature of candle-core.
//   - Windows ARM: CUDA, ROCm, and CANN are unavailable; CPU is the only option.

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    all(target_os = "windows", target_arch = "x86_64")
)))]
#[allow(dead_code)]
pub fn detect_backend() -> BackendKind {
    BackendKind::Cpu
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    all(target_os = "windows", target_arch = "x86_64")
)))]
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
}
