//! Safe abstractions over the [CUDA toolkit APIs](https://docs.nvidia.com/cuda/index.html)

pub mod cuda;
pub mod nvrtc;

pub mod prelude {
    pub use crate::cuda::prelude::*;
}
