//! A thin wrapper around [sys] providing [Result]s with [CudnnError].

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
    pub fn create() -> CudnnResult<Self> {
        let mut handle: Self = unsafe { std::mem::zeroed() };
        unsafe { cudnnCreate(&mut handle.0 as *mut _) }.result()?;
        Ok(handle)
    }
}
impl Drop for CudnnHandle {
    fn drop(&mut self) {
        unsafe { cudnnDestroy(self.0).result().unwrap() };
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn test_create_handle() {
        let _handle = CudnnHandle::create().unwrap();
    }
}