//! Wrappers around the [CUDA Profiling Tools Interface](https://docs.nvidia.com/cupti/index.html),
//! in two levels: an unsafe low-level API and a (still unsafe) thin wrapper around it.

pub mod result;
#[allow(warnings)]
pub mod sys;
