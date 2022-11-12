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
mod result;
mod pooling2d;
#[allow(warnings)]
mod sys;
mod tensor;

pub use activation::*;
pub use batch_normalization::*;
pub use conv::*;
pub use cuda_cudnn_result::*;
pub use result::*;
pub use tensor::*;
pub use pooling2d::*;
