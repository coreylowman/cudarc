use core::mem::MaybeUninit;

use crate::prelude::*;

use super::sys::*;

/// The Error type of all cudnn function calls.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CudnnError(pub cudnnStatus_t);

impl cudnnStatus_t {
    /// Transforms into a [CudaCudnnResult], so this can be mixed with cudnn
    /// related functions that call the cuda api (that returns [CudaResult]s).
    pub fn result(self) -> CudaCudnnResult<()> {
        match self {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            _ => Err(CudnnError(self).into()),
        }
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for CudnnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CudnnError {}

/// A handle to an initialized cudnn context. When dropping this value this
/// handle will be destroyed.
///
/// This does not implement sync because of https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#thread-safety
///
/// # See also
/// <https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnHandle_t>
pub struct CudnnHandle(cudnnHandle_t);
impl CudnnHandle {
    /// This creates a new [CudnnHandle] and sets the stream to the specified
    /// `device`.
    ///
    /// # See also
    /// <https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnCreate>
    /// <https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetStream>
    pub fn create(device: &CudaDevice) -> CudaCudnnResult<Self> {
        let mut handle = MaybeUninit::uninit();
        unsafe {
            cudnnCreate(handle.as_mut_ptr()).result()?;
            let handle = handle.assume_init();
            // TODO maybe fix this cast to the "same" type?
            cudnnSetStream(handle, device.cu_stream as *mut _).result()?;
            Ok(Self(handle))
        }
    }

    /// The raw handle wrapped inside [CudnnHandle].
    #[inline(always)]
    pub fn get_handle(&self) -> cudnnHandle_t {
        self.0
    }
}
impl Drop for CudnnHandle {
    fn drop(&mut self) {
        unsafe { cudnnDestroy(self.get_handle()).result().unwrap() };
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn test_create_handle() {
        let _handle = CudnnHandle::create(&CudaDeviceBuilder::new(0).build().unwrap()).unwrap();
    }
}
