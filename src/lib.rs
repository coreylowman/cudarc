//! Safe abstractions over the [CUDA toolkit APIs](https://docs.nvidia.com/cuda/index.html)

pub mod arrays;
pub mod compile;
pub mod cuda;
pub mod cudarc;
pub mod curand;
pub mod nvrtc;
pub mod rng;

pub mod prelude {
    pub use crate::cudarc::{
        CudaDevice, CudaDeviceBuilder, CudaError, CudaRc, IntoKernelParam, LaunchConfig,
        LaunchCudaFunction,
    };
}
