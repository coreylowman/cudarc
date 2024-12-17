//! Wrappers around the [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html),
//! in two levels: an unsafe low-level API and a (still unsafe) thin wrapper around it.

pub mod result;
#[allow(warnings)]
pub mod sys;
