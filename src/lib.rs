//! Safe abstractions over:
//! 1. [CUDA driver API](https://docs.nvidia.com/cuda/cuda-driver-api/index.html)
//! 2. [NVRTC API](https://docs.nvidia.com/cuda/nvrtc/index.html)
//! 3. [cuRAND API](https://docs.nvidia.com/cuda/curand/index.html)
#![no_std]
#![feature(generic_const_exprs)]

extern crate alloc;
extern crate no_std_compat as std;

pub mod arrays;
pub mod cudarc;
pub mod cudnn;
pub mod curand;
pub mod driver;
pub mod jit;
pub mod nvrtc;
pub mod rng;

pub mod prelude {
    pub use crate::arrays::NumElements;
    pub use crate::cudarc::{
        CudaDevice, CudaDeviceBuilder, CudaError, CudaRc, IntoKernelParam, LaunchConfig,
        LaunchCudaFunction,
    };
    pub use crate::cudnn::*;
    pub use crate::rng::CudaRng;
}
