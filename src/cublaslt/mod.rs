//! [CudaBlasLT] wraps around [cuBLASLt](https://docs.nvidia.com/cuda/cublas/index.html#using-the-cublaslt-api) via:
//! 
//! 1. Instantiate a [CudaBlasLT] handle with [CudaBlasLT::new()]
//! 2. Execute a gemm using [CudaBlasLT::matmul()]
//! 
//! Note that all above apis work with [crate::driver::DevicePtr]/[crate::driver::DevicePtrMut], so they
//! accept [crate::driver::CudaSlice], [crate::driver::CudaView], and [crate::driver::CudaViewMut].

pub mod result;
pub mod safe;
#[allow(warnings)]
pub mod sys;

pub use safe::*;
