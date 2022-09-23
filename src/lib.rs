//! Safe abstractions over:
//! 1. [CUDA driver API](https://docs.nvidia.com/cuda/cuda-driver-api/index.html)
//! 2. [NVRTC API](https://docs.nvidia.com/cuda/nvrtc/index.html)
//! 3. [cuRAND API](https://docs.nvidia.com/cuda/curand/index.html)

pub mod arrays;
pub mod cuda;
pub mod cudarc;
pub mod curand;
pub mod jit;
pub mod nvrtc;
pub mod rng;

pub mod prelude {
    pub use crate::cudarc::{
        CudaDevice, CudaDeviceBuilder, CudaError, CudaRc, IntoKernelParam, LaunchConfig,
        LaunchCudaFunction,
    };
}
