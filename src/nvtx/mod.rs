//! Wrappers around the [NVTX API](https://nvidia.github.io/NVTX/doxygen/index.html)
//! in two levels. See crate documentation for description of each.

pub mod safe;
#[allow(warnings)]
pub mod sys;

pub use safe::*;
