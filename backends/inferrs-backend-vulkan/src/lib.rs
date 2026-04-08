/// Probe whether a Vulkan-capable driver is available on this system.
///
/// This is implemented by attempting to `dlopen` `libvulkan.so.1` at runtime.
/// The backend `.so` itself does **not** link against Vulkan at compile time,
/// so loading this plugin on a system without Vulkan will not fail — only the
/// probe call will return non-zero.
///
/// NOTE: candle-core 0.8 does not yet have a Vulkan/wgpu `Device` variant.
/// The main `inferrs` binary uses a successful probe to log that Vulkan is
/// available and will accelerate inference once candle gains wgpu support.
/// Until then the binary falls back to CPU after logging the detection.
///
/// Platform coverage: Linux only. The probe always returns 1 (unavailable) on
/// Windows and macOS. When candle gains wgpu support and the Vulkan probe is
/// extended to Windows, update the cfg guard below and the release workflow.
///
/// Returns 0 if `libvulkan.so.1` can be opened, 1 otherwise.
#[no_mangle]
pub extern "C" fn inferrs_backend_probe() -> i32 {
    #[cfg(target_os = "linux")]
    {
        use std::ffi::CString;

        // Try both versioned and unversioned library names.
        for name in &["libvulkan.so.1", "libvulkan.so"] {
            let Ok(cname) = CString::new(*name) else {
                continue;
            };
            // SAFETY: dlopen is safe to call with a valid C string and flags.
            let handle =
                unsafe { libc::dlopen(cname.as_ptr(), libc::RTLD_LAZY | libc::RTLD_LOCAL) };
            if !handle.is_null() {
                unsafe { libc::dlclose(handle) };
                return 0;
            }
        }
        1
    }
    #[cfg(not(target_os = "linux"))]
    {
        1
    }
}
