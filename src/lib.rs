//! Safe abstractions over:
//! 1. [CUDA driver API](https://docs.nvidia.com/cuda/cuda-driver-api/index.html)
//! 2. [NVRTC API](https://docs.nvidia.com/cuda/nvrtc/index.html)
//! 3. [cuRAND API](https://docs.nvidia.com/cuda/curand/index.html)
//! 4. [cuBLAS API](https://docs.nvidia.com/cuda/cublas/index.html)
#![no_std]

extern crate alloc;
extern crate no_std_compat as std;

pub mod blas;
pub mod cublas;
pub mod cudnn;
pub mod curand;
pub mod device;
pub mod driver;
pub mod jit;
pub mod nvrtc;
pub mod rng;

pub mod prelude {
    pub use crate::device::{
        AsKernelParam, CudaDevice, CudaDeviceBuilder, CudaSlice, DriverError, LaunchAsync,
        LaunchConfig, ValidAsZeroBits,
    };
    pub use crate::cudnn::*;
}
