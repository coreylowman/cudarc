//! Helper functions to run cuda profiler range.
//! ```
//! use cudarc::profiler;
//!
//! profiler::start()?;
//! // Hotpath
//! profiler::stop()?;
//! // Now check your results
//! // nsys profile -c cudaProfilerApi /path/to/bin
//! // And this will profile only the hotpath.
//! ```
//!
use super::{result, sys};

pub use result::DriverError;

/// Description
/// Enables profile collection by the active profiling tool for the current context. If profiling is already enabled, then cuProfilerStart() has no effect.
///
/// cuProfilerStart and cuProfilerStop APIs are used to programmatically control the profiling granularity by allowing profiling to be done only on selective pieces of code.
///
/// Note:
///
/// Note that this function may also return error codes from previous, asynchronous launches.
pub fn start() -> Result<(), DriverError> {
    unsafe { sys::cuProfilerStart() }.result()
}

/// Description
/// Disables profile collection by the active profiling tool for the current context. If profiling is already disabled, then cuProfilerStop() has no effect.
/// cuProfilerStart and cuProfilerStop APIs are used to programmatically control the profiling granularity by allowing profiling to be done only on selective pieces of code.
///
/// Note:
/// Note that this function may also return error codes from previous, asynchronous launches.
pub fn stop() -> Result<(), DriverError> {
    unsafe { sys::cuProfilerStop() }.result()
}
