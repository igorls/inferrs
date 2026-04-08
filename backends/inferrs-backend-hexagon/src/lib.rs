//! Hexagon HTP backend plugin for `inferrs`.
//!
//! # Overview
//!
//! This crate is compiled as a C dynamic library (`cdylib`) and loaded at
//! runtime by the `inferrs` binary via `dlopen` / `LoadLibraryW`.  It exports
//! the single well-known symbol:
//!
//! ```c
//! int inferrs_backend_probe(void);  // 0 = Hexagon available, non-zero = not available
//! ```
//!
//! The probe opens `libcdsprpc.so` (Linux / Android) or `libcdsprpc.dll`
//! (Windows ARM64), resolves all required FastRPC symbols, and calls
//! `DSPRPC_GET_DSP_INFO` to confirm that a Hexagon DSP compute domain is
//! reachable.  On platforms where Hexagon hardware cannot exist (macOS,
//! Windows x86_64) the probe returns non-zero immediately without attempting
//! any library load.
//!
//! # Platform support matrix
//!
//! | OS      | Arch    | Hexagon present? | Library probed            |
//! |---------|---------|------------------|---------------------------|
//! | Linux   | x86_64  | No               | fast-fail (returns 1)     |
//! | Linux   | aarch64 | Possibly (SoC)   | `libcdsprpc.so`           |
//! | Android | aarch64 | Yes (Snapdragon) | `libcdsprpc.so`           |
//! | macOS   | x86_64  | No               | fast-fail (returns 1)     |
//! | macOS   | aarch64 | No (Apple Si.)   | fast-fail (returns 1)     |
//! | Windows | x86_64  | No               | fast-fail (returns 1)     |
//! | Windows | aarch64 | Yes (Snapdragon) | `libcdsprpc.dll` via SCM  |
//!
//! # Dynamic loading approach
//!
//! The approach mirrors `ggml/src/ggml-hexagon/htp-drv.cpp` from llama.cpp:
//!
//! 1. `libdl` provides a POSIX/Windows-portable `dlopen`/`LoadLibraryW`
//!    abstraction (`DlHandle`, `dl_open`, `dl_sym`).
//! 2. `fastrpc` resolves all Qualcomm FastRPC symbols from `libcdsprpc` into
//!    a global `OnceLock<FastRpcDriver>`.
//! 3. The probe calls `get_hex_arch_ver(3)` (compute domain = 3) to confirm
//!    the DSP is reachable and to log which HTP generation is present.

mod fastrpc;
mod libdl;

/// Probe whether a Qualcomm Hexagon HTP DSP is available and accessible via
/// FastRPC on this system.
///
/// The probe is platform-aware:
///
/// - **Linux aarch64 / Android aarch64**: attempts to open `libcdsprpc.so`
///   and query the compute DSP (domain 3).  Returns `0` if a Hexagon arch
///   version ≥ 68 is found, `1` otherwise.
/// - **Windows aarch64**: locates `libcdsprpc.dll` via the SCM service path
///   of `qcnspmcdm`, then performs the same DSP query.
/// - **All other platforms** (x86_64 Linux/Windows, macOS x86_64/aarch64):
///   returns `1` immediately — Hexagon hardware does not exist on these
///   targets.
///
/// This function is `dlopen`'d by the main `inferrs` binary at runtime so
/// that the binary itself carries no link-time dependency on the Qualcomm SDK.
#[no_mangle]
pub extern "C" fn inferrs_backend_probe() -> i32 {
    probe_hexagon()
}

/// Inner implementation, separated for readability.
fn probe_hexagon() -> i32 {
    // Fast-fail on platforms that cannot have Hexagon hardware.
    if !platform_may_have_hexagon() {
        return 1;
    }

    // Attempt to load libcdsprpc and resolve all required FastRPC symbols.
    if let Err(_e) = fastrpc::htpdrv_init() {
        return 1;
    }

    // Query the compute DSP (domain 3) for its Hexagon architecture version.
    // A successful response confirms that a Hexagon NPU is reachable.
    match fastrpc::get_hex_arch_ver(3) {
        Ok(_arch) => 0,
        Err(_e) => 1,
    }
}

/// Returns `true` on CPU architectures / OS combinations where Qualcomm
/// Hexagon hardware is plausible, `false` on targets where it definitively
/// cannot exist.
///
/// This avoids pointless `dlopen` attempts on x86_64 or Apple Silicon
/// systems and keeps the probe fast.
const fn platform_may_have_hexagon() -> bool {
    // Hexagon exists only on Qualcomm Snapdragon SoCs — all of which are
    // ARM64 (aarch64).  On x86_64 machines and on Apple Silicon (which has
    // its own ANE, not Hexagon) there is no FastRPC driver at all.
    cfg!(any(
        // Linux on ARM (including Android via the linux cfg).
        all(target_os = "linux", target_arch = "aarch64"),
        // Android explicit target (cross-compilation target triple).
        all(target_os = "android", target_arch = "aarch64"),
        // Windows on ARM64 (Snapdragon X Elite / 8cx devices).
        all(target_os = "windows", target_arch = "aarch64"),
    ))
}
