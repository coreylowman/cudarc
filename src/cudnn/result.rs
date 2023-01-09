//! A thin wrapper around [sys] providing [Result]s with [CudnnError].

use core::mem::MaybeUninit;

use crate::device::CudaDevice;

use super::sys::*;

pub type CudnnResult<T> = Result<T, CudnnError>;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CudnnError(pub cudnnStatus_t);

impl cudnnStatus_t {
    /// Transforms into a [Result] of [CudnnError]
    pub fn result(self) -> Result<(), CudnnError> {
        match self {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            _ => Err(CudnnError(self)),
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
pub struct CudnnHandle(pub(crate) cudnnHandle_t);
impl CudnnHandle {
    pub fn create(device: &CudaDevice) -> CudnnResult<Self> {
        let mut handle = MaybeUninit::uninit();
        unsafe {
            cudnnCreate(handle.as_mut_ptr()).result()?;
            let handle = handle.assume_init();
            cudnnSetStream(handle, device.cu_stream as *mut _).result()?;
            Ok(Self(handle))
        }
    }
}
impl Drop for CudnnHandle {
    fn drop(&mut self) {
        unsafe { cudnnDestroy(self.0).result().unwrap() };
    }
}

#[cfg(test)]
mod tests {
    use crate::device::CudaDeviceBuilder;

    use super::CudnnHandle;

    #[test]
    fn create_and_drop() {
        let _handle = CudnnHandle::create(&CudaDeviceBuilder::new(0).build().unwrap()).unwrap();
    }
}
