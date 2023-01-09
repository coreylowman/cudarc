//! Wrappers around the [CUDA driver API](https://docs.nvidia.com/cuda/cuda-driver-api/index.html),
//! in two levels: [result], [sys]. See [crate::device] for safe wrappers.
//!
//! 1. [result] thin wrapper around [sys] to return [Result<_,
//! result::CudaError>] 2. [sys] the raw bindings

pub mod result;
#[allow(warnings)]
pub mod sys;
