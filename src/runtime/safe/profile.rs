use crate::runtime::{result, sys::lib};

/// Calls [profiler_start()] in [Profiler::new()], and [profiler_stop()] in [Drop].
#[derive(Default)]
pub struct Profiler {}

impl Profiler {
    /// Enables profile collection by the active profiling tool for the current context. If profiling is already enabled, then Profiler::new() has no effect.
    /// More info in [Cuda docs](https://docs.nvidia.com/cuda/profiler-users-guide/)
    /// ```no_run
    /// use cudarc::runtime::{Profiler};
    /// # use cudarc::runtime::result;
    ///
    /// # fn run() -> Result<(), result::RuntimeError>{
    /// {
    /// let profiler = Profiler::new()?;
    /// // Hotpath
    /// // Profiler stops on drop
    /// }
    /// # Ok(())
    /// # }
    /// // Now check your results
    /// // nsys profile -c cudaProfilerApi /path/to/bin
    /// // And this will profile only the hotpath.
    /// ```
    ///
    pub fn new() -> Result<Self, result::RuntimeError> {
        profiler_start()?;
        Ok(Self {})
    }
}

impl Drop for Profiler {
    fn drop(&mut self) {
        // We don't want to panic on drop.
        profiler_stop().ok();
    }
}

/// Enables profile collection by the active profiling tool for the current context. If profiling is already enabled, then profiler_start() has no effect.
/// More info in [Cuda docs](https://docs.nvidia.com/cuda/profiler-users-guide/)
/// For RAII version see [`Profiler::new`].
/// ```no_run
/// use cudarc::runtime::{profiler_start, profiler_stop};
/// # use cudarc::runtime::result;
///
/// # fn run() -> Result<(), result::RuntimeError>{
/// profiler_start()?;
/// // Hotpath
/// profiler_stop()?;
/// # Ok(())
/// # }
/// // Now check your results
/// // nsys profile -c cudaProfilerApi /path/to/bin
/// // And this will profile only the hotpath.
/// ```
///
pub fn profiler_start() -> Result<(), result::RuntimeError> {
    unsafe { lib().cudaProfilerStart() }.result()
}

/// Disables profile collection by the active profiling tool for the current context. If profiling is already disabled, then profiler_stop() has no effect.
pub fn profiler_stop() -> Result<(), result::RuntimeError> {
    unsafe { lib().cudaProfilerStop() }.result()
}
