//! Wrappers around the [Nvidia Runtime Compilation (nvrtc) API](https://docs.nvidia.com/cuda/nvrtc/index.html),
//! in three levels: [compile], [result], and [sys].
//!
//! 1. [compile] provides safe abstractions over [result]
//! 2. [result] thin wrapper around [sys] to return [Result<_, result::NvrtcError>]
//! 3. [sys] the raw bindings

pub mod result;
#[allow(warnings)]
pub mod sys;
