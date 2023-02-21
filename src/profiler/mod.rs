//! Wrappers around the [CUDA Profiler driver API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PROFILER.html),
//! in three levels. See crate documentation for description of each.

pub mod result;
pub mod safe;
#[allow(warnings)]
pub mod sys;

pub use safe::*;
