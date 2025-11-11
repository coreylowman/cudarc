//! [CuTensor] wraps around the [cuTENSOR API](https://docs.nvidia.com/cuda/cutensor/index.html).
//!
//! cuTENSOR is a high-performance CUDA library for tensor primitives including:
//! - Tensor contractions (generalized matrix multiplications using Einstein notation)
//! - Tensor reductions
//! - Element-wise operations
//!
//! To use:
//!
//! 1. Instantiate a [CuTensor] handle with [CuTensor::new()]
//! 2. Create tensor descriptors with [TensorDescriptor::new()]
//! 3. Perform operations like contractions and reductions
//!
//! # Version Requirements
//!
//! - CUDA 11.x: cuTENSOR 1.3+ (limited API support)
//! - CUDA 12.x: cuTENSOR 2.0+ (recommended)
//! - CUDA 13.x: cuTENSOR 2.3+
//!
//! # Example
//!
//! ```rust,ignore
//! use cudarc::cutensor::CuTensor;
//! use cudarc::driver::CudaContext;
//!
//! let ctx = CudaContext::new(0)?;
//! let stream = ctx.default_stream();
//! let cutensor = CuTensor::new(stream.clone())?;
//!
//! // Check runtime version
//! let (major, minor) = cutensor.version();
//! println!("cuTENSOR version: {}.{}", major, minor);
//! ```

pub mod result;
#[allow(warnings)]
pub mod sys;
