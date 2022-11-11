//! A thin wrapper around [sys] providing [Result]s with [CudnnError].

use core::mem::MaybeUninit;

use crate::cudarc::CudaDevice;

use super::sys::*;
use super::CudaCudnnResult;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CudnnError(pub cudnnStatus_t);

impl cudnnStatus_t {
    /// Transforms into a [CudaCudnnResult]
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

// No sync because of https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#thread-safety
pub struct CudnnHandle(cudnnHandle_t);
impl CudnnHandle {
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
