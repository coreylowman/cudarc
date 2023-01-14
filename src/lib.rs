//! Safe abstractions over:
//! 1. [CUDA driver API](https://docs.nvidia.com/cuda/cuda-driver-api/index.html)
//! 2. [NVRTC API](https://docs.nvidia.com/cuda/nvrtc/index.html)
//! 3. [cuRAND API](https://docs.nvidia.com/cuda/curand/index.html)
//! 4. [cuBLAS API](https://docs.nvidia.com/cuda/cublas/index.html)
//!
//! Each of the modules for the above is organized into three levels:
//! 1. A `safe` module which provides safe abstractions over the `result` module
//! 2. A `result` which is a thin wrapper around the `sys` module to ensure all functions return [Result]
//! 3. A `sys` module which contains the raw bindings
//!
//! Each module exports the safe API, and exposes each level if you want to use a different one.

#![no_std]

extern crate alloc;
extern crate no_std_compat as std;

pub mod cublas;
pub mod curand;
pub mod driver;
pub mod nvrtc;

pub mod prelude {
    pub use crate::driver::{
        AsKernelParam, CudaDevice, CudaDeviceBuilder, CudaSlice, DriverError, LaunchAsync,
        LaunchConfig, ValidAsZeroBits,
    };
}
