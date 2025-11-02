//! Safe abstractions around [crate::cutensor::result] for tensor operations.
//!
//! Tensor data is stored in [crate::driver::CudaSlice], which provides:
//! - Automatic event tracking
//! - Multi-stream synchronization
//! - Memory management
//! - Type safety
//!
//! Tensor descriptors ([TensorDescriptor]) describe the shape and layout,
//! while the actual data lives in `CudaSlice<T>`.

use super::{result, result::CutensorError, sys};
use crate::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

/// Wrapper around [sys::cutensorHandle_t]
///
/// 1. Create with [CuTensor::new()]
/// 2. Use to create tensor descriptors and execute operations
///
/// Note: This maintains an instance of [`Arc<CudaStream>`], so will prevent the stream
/// from being dropped.
#[derive(Debug)]
pub struct CuTensor {
    pub(crate) handle: sys::cutensorHandle_t,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) version: (usize, usize, usize),
}

unsafe impl Send for CuTensor {}
unsafe impl Sync for CuTensor {}

impl CuTensor {
    /// Creates a new cuTENSOR handle.
    ///
    /// The handle will be associated with the CUDA device of the provided stream.
    /// Runtime version detection is performed to ensure compatibility.
    pub fn new(stream: Arc<CudaStream>) -> Result<Self, CutensorError> {
        let ctx = stream.context();
        ctx.record_err(ctx.bind_to_thread());
        let handle = result::create_handle()?;
        let version = result::get_version();

        // Validate version compatibility based on CUDA version
        #[cfg(feature = "cuda-13000")]
        {
            if version.0 < 2 || (version.0 == 2 && version.1 < 3) {
                eprintln!(
                    "Warning: cuTENSOR {}.{}.{} detected, but CUDA 13.x requires cuTENSOR 2.3+",
                    version.0, version.1, version.2
                );
            }
        }

        #[cfg(all(
            not(feature = "cuda-13000"),
            any(
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080",
                feature = "cuda-12090"
            )
        ))]
        {
            if version.0 < 2 {
                eprintln!(
                    "Warning: cuTENSOR {}.{}.{} detected, but CUDA 12.x requires cuTENSOR 2.0+",
                    version.0, version.1, version.2
                );
            }
        }

        Ok(Self {
            handle,
            stream,
            version,
        })
    }

    /// Returns a reference to the underlying cuTENSOR handle.
    pub fn handle(&self) -> &sys::cutensorHandle_t {
        &self.handle
    }

    /// Returns the cuTENSOR library version as (major, minor, patch).
    pub fn version(&self) -> (usize, usize, usize) {
        self.version
    }

    /// Returns a reference to the CUDA stream associated with this handle.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

impl Drop for CuTensor {
    fn drop(&mut self) {
        let handle = std::mem::replace(&mut self.handle, std::ptr::null_mut());
        if !handle.is_null() {
            unsafe { result::destroy_handle(handle) }.unwrap();
        }
    }
}

/// Wrapper around [sys::cutensorTensorDescriptor_t]
///
/// Describes the shape, layout, and data type of a tensor.
/// The actual tensor data is stored separately in a [CudaSlice].
///
/// # Example
///
/// ```rust,ignore
/// // Create a 3x4 matrix descriptor
/// let desc = TensorDescriptor::new(
///     &cutensor,
///     &[3, 4],           // extents (shape)
///     None,              // strides (None = packed/column-major)
///     sys::cudaDataType_t::CUDA_R_32F,
/// )?;
/// ```
#[derive(Debug)]
pub struct TensorDescriptor {
    pub(crate) desc: sys::cutensorTensorDescriptor_t,
    pub(crate) handle: Arc<CuTensor>,
}

impl TensorDescriptor {
    /// Creates a new tensor descriptor.
    ///
    /// # Arguments
    ///
    /// * `handle` - The cuTENSOR handle
    /// * `extents` - The size of each mode (dimension)
    /// * `strides` - Optional strides for each mode. If None, uses packed column-major layout
    /// * `data_type` - The data type of the tensor elements
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // 2x3 matrix, column-major (default)
    /// let desc = TensorDescriptor::new(
    ///     &cutensor,
    ///     &[2, 3],
    ///     None,
    ///     sys::cudaDataType_t::CUDA_R_32F,
    /// )?;
    ///
    /// // Custom strides (row-major)
    /// let desc = TensorDescriptor::new(
    ///     &cutensor,
    ///     &[2, 3],
    ///     Some(&[3, 1]),  // row-major: stride of 3 for rows, 1 for columns
    ///     sys::cudaDataType_t::CUDA_R_32F,
    /// )?;
    /// ```
    pub fn new(
        handle: &Arc<CuTensor>,
        extents: &[i64],
        strides: Option<&[i64]>,
        data_type: sys::cudaDataType_t,
    ) -> Result<Arc<Self>, CutensorError> {
        let num_modes = extents.len() as u32;

        // Compute default strides if not provided (packed, column-major)
        let default_strides: Vec<i64>;
        let strides_ptr = if let Some(s) = strides {
            s.as_ptr()
        } else {
            default_strides = compute_default_strides(extents);
            default_strides.as_ptr()
        };

        let desc = unsafe {
            result::create_tensor_descriptor(
                handle.handle,
                num_modes,
                extents.as_ptr(),
                strides_ptr,
                data_type,
                0, // CUTENSOR_OP_IDENTITY
            )?
        };

        Ok(Arc::new(Self {
            desc,
            handle: handle.clone(),
        }))
    }

    /// Returns the underlying descriptor handle.
    pub fn desc(&self) -> sys::cutensorTensorDescriptor_t {
        self.desc
    }
}

impl Drop for TensorDescriptor {
    fn drop(&mut self) {
        let desc = std::mem::replace(&mut self.desc, std::ptr::null_mut());
        if !desc.is_null() {
            unsafe { result::destroy_tensor_descriptor(desc) }.unwrap();
        }
    }
}

/// Computes default strides for a packed, column-major tensor.
fn compute_default_strides(extents: &[i64]) -> Vec<i64> {
    let mut strides = Vec::with_capacity(extents.len());
    let mut stride = 1i64;
    for &extent in extents {
        strides.push(stride);
        stride *= extent;
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::CudaContext;

    #[test]
    fn test_cutensor_create() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let cutensor = CuTensor::new(stream.clone()).unwrap();

        let (major, minor, patch) = cutensor.version();
        println!("cuTENSOR version: {}.{}.{}", major, minor, patch);
        assert!(major >= 2 || (major == 1 && minor >= 3));
    }
}
