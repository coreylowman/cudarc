//! [CudaBlas] wraps around the [cublas API](https://docs.nvidia.com/cuda/cublas/index.html).
//! 
//! To use:
//! 
//! 1. Instantiate a [CudaBlas] handle with [CudaBlas::new()]
//! 2. Choose your operation: [Gemm], [Gemv], and [Asum] traits, which [CudaBlas] implements.
//! 3. f16/bf16/f32/f64 are all supported at the trait level.
//! 4. Instantiate your corresponding config: [GemmConfig], [StridedBatchedConfig], [GemvConfig], [AsumConfig]
//! 5. Call using [CudaBlas::gemm()], [CudaBlas::gemv()], or [CudaBlas::asum()]
//! 
//! Note that all above apis work with [crate::driver::DevicePtr]/[crate::driver::DevicePtrMut], so they
//! accept [crate::driver::CudaSlice], [crate::driver::CudaView], and [crate::driver::CudaViewMut].

pub mod result;
pub mod safe;
#[allow(warnings)]
pub mod sys;

pub use safe::*;
