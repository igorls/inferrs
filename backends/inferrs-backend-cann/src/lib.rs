/// Probe whether a Huawei Ascend NPU with the CANN (Compute Architecture for
/// Neural Networks) runtime is available on this system.
///
/// The probe works by attempting to open the CANN ACL runtime library
/// (`libascendcl.so` on Linux/Android) at runtime via `dlopen`/`LoadLibrary`.
/// The plugin itself does **not** link against CANN at compile time, so:
///
/// * Loading the plugin on a system without CANN will succeed silently.
/// * Only the probe call returns non-zero when the CANN runtime is absent.
///
/// If `libascendcl.so` opens successfully the probe additionally tries to
/// resolve and call `aclrtGetDeviceCount` to verify that at least one
/// Ascend device is actually enumerable, not just that the library is
/// installed.
///
/// Supported platforms
/// -------------------
/// | OS      | Arch          | Library                 |
/// |---------|---------------|-------------------------|
/// | Linux   | x86_64        | `libascendcl.so`        |
/// | Linux   | aarch64       | `libascendcl.so`        |
/// | Android | aarch64       | `libascendcl.so`        |
///
/// macOS and Windows are not supported by the CANN SDK and always return 1.
///
/// Returns 0 if at least one Ascend device is enumerable, 1 otherwise.
#[no_mangle]
pub extern "C" fn inferrs_backend_probe() -> i32 {
    // CANN is only available on Linux (x86_64 / aarch64) and Android (aarch64).
    // The SDK explicitly states it does not support macOS or Windows.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        return probe_cann();
    }

    // All other platforms (macOS, Windows, 32-bit, RISC-V, …) are unsupported.
    #[allow(unreachable_code)]
    1
}

/// Inner probe function gated to the supported OS/arch combination.
///
/// Uses raw `libc::dlopen` / `dlsym` rather than the `libloading` crate so
/// that the plugin does not carry an extra dependency — matching the pattern
/// used in `inferrs-backend-vulkan`.
#[cfg(any(target_os = "linux", target_os = "android"))]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn probe_cann() -> i32 {
    use std::ffi::CString;

    // The CANN ACL runtime.  The CANN installer places this under
    // $ASCEND_TOOLKIT_HOME/lib64/ and adds that to /etc/ld.so.conf.d/ (or
    // LD_LIBRARY_PATH on embedded/Android targets).
    //
    // We try the unversioned SO name first (preferred when the linker cache
    // has it), then a set of versioned fallbacks covering known CANN SDK
    // releases:
    //   - CANN 7.x / 8.x ship libascendcl.so  (no version suffix in lib64/)
    //   - Some OEM images expose a versioned symlink libascendcl.so.1
    let candidate_names: &[&str] = &["libascendcl.so", "libascendcl.so.1"];

    for name in candidate_names {
        let Ok(cname) = CString::new(*name) else {
            continue;
        };

        // SAFETY: dlopen is safe to call with a valid C string and flags.
        // RTLD_LAZY | RTLD_LOCAL: resolve symbols on-demand and do not
        // pollute the global symbol namespace.
        let handle = unsafe { libc::dlopen(cname.as_ptr(), libc::RTLD_LAZY | libc::RTLD_LOCAL) };

        if handle.is_null() {
            continue;
        }

        // Library opened — now check that at least one Ascend device exists.
        // We resolve aclrtGetDeviceCount dynamically so we never need to
        // declare the CANN SDK headers at compile time.
        //
        // Signature: aclError aclrtGetDeviceCount(uint32_t *count)
        //   aclError is a typedef for int32_t; 0 == ACL_SUCCESS.
        let result = probe_device_count(handle);

        // Close the handle regardless of success — we only needed the probe.
        // SAFETY: handle is non-null and was returned by dlopen.
        unsafe { libc::dlclose(handle) };

        if result {
            return 0;
        }

        // Library loaded but no devices — don't bother trying other names.
        return 1;
    }

    // Library not found / could not be opened.
    1
}

/// Resolves `aclrtGetDeviceCount` from an already-opened CANN handle and
/// calls it.  Returns `true` when at least one Ascend device is present.
#[cfg(any(target_os = "linux", target_os = "android"))]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn probe_device_count(handle: *mut libc::c_void) -> bool {
    use std::ffi::CString;

    // `aclrtGetDeviceCount(uint32_t *count) -> int32_t`
    // We model the return type as i32 (aclError) and count as u32.
    type AclrtGetDeviceCount = unsafe extern "C" fn(*mut u32) -> i32;

    let sym_name = match CString::new("aclrtGetDeviceCount") {
        Ok(s) => s,
        Err(_) => return false,
    };

    // SAFETY: handle is non-null and valid; we pass a valid C string.
    let sym_ptr = unsafe { libc::dlsym(handle, sym_name.as_ptr()) };
    if sym_ptr.is_null() {
        // Symbol not found — treat as unavailable rather than crashing.
        return false;
    }

    // SAFETY: we verified the symbol exists and cast it to the known
    // CANN SDK signature.  The function is called with a stack-allocated
    // u32 pointer, which is valid for the duration of the call.
    let get_device_count: AclrtGetDeviceCount = unsafe { std::mem::transmute(sym_ptr) };

    let mut count: u32 = 0;
    // aclError == 0 means ACL_SUCCESS.
    let acl_err = unsafe { get_device_count(&mut count) };
    acl_err == 0 && count > 0
}
