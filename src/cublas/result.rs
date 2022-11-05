use core::ptr::null_mut;

use super::sys::*;

/// Wrapper around [sys::cublasStatus_t].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CublasError(pub cublasStatus_t);

pub type CublasResult<T> = Result<T, CublasError>;
impl cublasStatus_t {
    /// Transforms into a [Result] of [CublasError]
    pub fn result(self) -> CublasResult<()> {
        match self {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
            _ => Err(CublasError(self)),
        }
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for CublasError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CublasError {}

/// A handle to a cublas context.
pub struct CublasHandle(pub(crate) cublasHandle_t);
/// This is thread-safe according to NVIDIA (<https://docs.nvidia.com/cuda/cublas/index.html#thread-safety2>),
/// as long as no configuration is changed (which is currently impossible).
unsafe impl Send for CublasHandle {}
unsafe impl Sync for CublasHandle {}
impl Drop for CublasHandle {
    fn drop(&mut self) {
        unsafe { cublasDestroy_v2(self.0).result().unwrap(); }
    }
}

impl CublasHandle {
    pub fn create() -> CublasResult<Self> {
        let mut handle = Self(null_mut());
        unsafe { cublasCreate_v2(&mut handle.0 as *mut _) }.result()?;
        Ok(handle)
    }
}
