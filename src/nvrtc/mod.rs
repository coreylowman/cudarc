//! Wrappers around the [Nvidia Runtime Compilation (nvrtc) API](https://docs.nvidia.com/cuda/nvrtc/index.html),
//! in two levels: [result], and [sys]. See [crate::jit] for safe wrapper.
//!
//! 1. [result] thin wrapper around [sys] to return [Result<_,
//! result::NvrtcError>] 2. [sys] the raw bindings

pub mod result;
#[allow(warnings)]
pub mod sys;
