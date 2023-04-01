//! Safe wrappers around cuDNN.
//!
//! 1. Allocate tensor descriptors with [`Cudnn::create_tensor4d()`]
//! 2. Allocate filter descriptors with [`Cudnn::create_filter4d()`]
//! 3. Allocate conv descriptors with [`Cudnn::create_conv2d()`]
//! 4. Instantiate one of the following algorithms with the descriptors:
//!     a. [`Conv2dForward`]
//!     b. [`Conv2dBackwardData`] for computing gradient of image
//!     c. [`Conv2dBackwardFilter`] for computing gradient of filters
//! 5. Call the `pick_algorithm` method of the struct. Specify the number of options to compare with a const generic.
//! 6. Call the `get_workspace_size` method of the struct.
//! 7. Re-allocate the workspace to the appropriate size.
//! 8. Call the `launch` method of the struct.

mod conv;
mod core;

pub use self::conv::{
    Conv2dBackwardData, Conv2dBackwardFilter, Conv2dDescriptor, Conv2dForward, FilterDescriptor,
};
pub use self::core::{Cudnn, CudnnDataType, TensorDescriptor};
