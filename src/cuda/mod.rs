//! Wrappers around the [CUDA driver API](https://docs.nvidia.com/cuda/cuda-driver-api/index.html),
//! in three levels: [rc], [result], [sys].
//!
//! 1. [rc] provides safe abstractions over [result]
//! 2. [result] thin wrapper around [sys] to return [Result<_, result::CudaError>]
//! 3. [sys] the raw bindings

pub mod rc;
pub mod result;
#[allow(warnings)]
pub mod sys;

pub mod prelude {
    pub use super::rc::*;
}
