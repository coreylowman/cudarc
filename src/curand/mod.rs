//! [CudaRng] safe bindings around [cuRAND](https://docs.nvidia.com/cuda/curand/index.html).
//!
//! Instantiate with [CudaRng::new()], and then fill existing [crate::driver::CudaSlice]/[crate::driver::CudaViewMut]
//! with three different
//! 1. Uniform - [CudaRng::fill_with_uniform()]
//! 2. Normal - [CudaRng::fill_with_normal()]
//! 3. LogNormal - [CudaRng::fill_with_log_normal()]

pub mod result;
pub mod safe;
#[allow(warnings)]
pub mod sys;

pub use safe::*;
