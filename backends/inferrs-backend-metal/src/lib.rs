/// Probe whether an Apple Metal device is available and functional.
///
/// This is implemented by attempting to create a `candle_core::Device::Metal`
/// using `Device::new_metal(0)`.  The Metal framework is part of macOS and
/// is always present on supported hardware; the probe will only fail on
/// headless CI environments or very old hardware that pre-dates Metal support
/// (pre-2012 Macs).
///
/// The backend shared library links against the Metal framework at compile
/// time (`candle-core` with the `metal` feature), so this plugin can only
/// be built on macOS.  On all other platforms the probe immediately returns
/// non-zero.
///
/// Returns 0 if `Device::new_metal(0)` succeeds, 1 otherwise.
#[no_mangle]
pub extern "C" fn inferrs_backend_probe() -> i32 {
    #[cfg(target_os = "macos")]
    {
        match candle_core::Device::new_metal(0) {
            Ok(_) => 0,
            Err(_) => 1,
        }
    }
    #[cfg(not(target_os = "macos"))]
    1
}
