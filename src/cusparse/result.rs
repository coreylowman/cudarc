use std::mem::MaybeUninit;

use super::sys;

/// Wrapper around [sys::CUresult]. See
/// nvidia's [CUresult docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CusparseError(pub sys::cusparseStatus_t);

impl sys::cusparseStatus_t {
    #[inline]
    pub fn result(self) -> Result<(), CusparseError> {
        match self {
            sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS => Ok(()),
            _ => Err(CusparseError(self)),
        }
    }
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsecreate)
pub fn create() -> Result<sys::cusparseHandle_t, CusparseError> {
    let mut handle = MaybeUninit::uninit();
    unsafe {
        sys::cusparseCreate(handle.as_mut_ptr()).result()?;
        Ok(handle.assume_init())
    }
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsedestroy)
/// # Safety
/// Ensure `handle` has not been freed already
pub unsafe fn destroy(handle: sys::cusparseHandle_t) -> Result<(), CusparseError> {
    sys::cusparseDestroy(handle).result()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_destroy() -> Result<(), CusparseError> {
        let handle = create()?;
        unsafe { destroy(handle) }?;
        Ok(())
    }
}
