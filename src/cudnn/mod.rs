//! This module is a very thin wrapper around [cudnn](https://docs.nvidia.com/deeplearning/cudnn/api/index.html).
//!
//!
//! This module tries to maximize the reuse of descriptors/structs
//! (e.g. [Activation] or [Convolution2DForward]),
//! so the initialization only takes tensor descriptors
//! while the actual api calls take the tensor data (or the tensor itself).
//! For example, every [Activation<A>] with
//! the [ActivationMode] `A` is only required to be created once.
//!
//!
//! # Descriptors
//! Every descriptor is wrapped inside a struct to prevent memory leaks
//! and every function returns a [CudaCudnnError] as every api call is fallible.
//! Additionally, all descriptors are wrapped in a [Rc] so they can be shared
//! between multiple structs.
//!
//! # Tensors
//! All tensors are fully packed 4D tensors in `NCHW layout`;
//! if you need lower-ranked tensors, you should set the following dimensions to
//! `1` (in this order): `H`, `C`, `W`, `N`.

mod activation;
mod batch_normalization;
mod conv;
mod cuda_cudnn_result;
mod custom_kernel_functions_names;
mod pooling2d;
mod result;
#[allow(warnings)]
mod sys;
mod tensor;

pub use activation::*;
pub use batch_normalization::*;
pub use conv::*;
pub use cuda_cudnn_result::*;
pub use pooling2d::*;
pub use result::*;
pub use tensor::*;

pub(crate) const CUSTOM_KERNEL_MODULE: &str = "custom_kernels";
impl crate::cudarc::CudaDeviceBuilder {
    pub fn with_cudnn_modules(self) -> Self {
        self.with_ptx_from_file(
            CUSTOM_KERNEL_MODULE,
            // is this as efficient as including bytes (`include_bytes!`) on compile time?
            "src/cudnn/custom_kernels.ptx",
            &custom_kernel_functions_names::FUNCTION_NAMES,
        )
    }
}
