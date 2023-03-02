use crate::driver::{result, sys};

/// Enables profile collection by the active profiling tool for the current context. If profiling is already enabled, then profiler_start() has no effect.
/// ```ignore
/// use cudarc::driver::{profiler_start, profiler_stop};
///
/// profiler_start()?;
/// // Hotpath
/// profiler_stop()?;
/// // Now check your results
/// // nsys profile -c cudaProfilerApi /path/to/bin
/// // And this will profile only the hotpath.
/// ```
///
pub fn profiler_start() -> Result<(), result::DriverError> {
    unsafe { sys::cuProfilerStart() }.result()
}

/// Disables profile collection by the active profiling tool for the current context. If profiling is already disabled, then profiler_stop() has no effect.
pub fn profiler_stop() -> Result<(), result::DriverError> {
    unsafe { sys::cuProfilerStop() }.result()
}
