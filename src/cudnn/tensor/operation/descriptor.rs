use core::mem::MaybeUninit;

use crate::cudnn::sys::*;
use crate::prelude::*;

/// A descriptor for a tensor operation. It is destroyed when it is dropped.
///
/// # See also
/// <https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnOpTensorDescriptor_t>
/// <https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnDestroyOpTensorDescriptor>
pub struct TensorOperationDescriptor(pub(crate) cudnnOpTensorDescriptor_t);
impl TensorOperationDescriptor {
    /// # See also
    /// <https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnCreateOpTensorDescriptor>
    pub fn create() -> CudaCudnnResult<Self> {
        let mut descriptor = MaybeUninit::uninit();
        unsafe {
            cudnnCreateOpTensorDescriptor(descriptor.as_mut_ptr()).result()?;
            Ok(Self(descriptor.assume_init()))
        }
    }
}
impl Drop for TensorOperationDescriptor {
    fn drop(&mut self) {
        unsafe { cudnnDestroyOpTensorDescriptor(self.0) }
            .result()
            .unwrap();
    }
}
