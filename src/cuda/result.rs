//! A thin wrapper around [sys].

use super::sys;
use std::{
    ffi::{c_void, CStr},
    mem::{size_of, MaybeUninit},
    os::raw::c_uint,
};

/// TODO
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct CudaError(pub sys::CUresult);

impl sys::CUresult {
    /// TODO
    pub fn result(self) -> Result<(), CudaError> {
        match self {
            sys::CUresult::CUDA_SUCCESS => Ok(()),
            _ => Err(CudaError(self)),
        }
    }
}

impl CudaError {
    /// TODO
    pub fn error_name(&self) -> Result<&CStr, CudaError> {
        let mut err_str = MaybeUninit::uninit();
        unsafe {
            sys::cuGetErrorName(self.0, err_str.as_mut_ptr()).result()?;
            Ok(CStr::from_ptr(err_str.assume_init()))
        }
    }

    /// TODO
    pub fn error_string(&self) -> Result<&CStr, CudaError> {
        let mut err_str = MaybeUninit::uninit();
        unsafe {
            sys::cuGetErrorString(self.0, err_str.as_mut_ptr()).result()?;
            Ok(CStr::from_ptr(err_str.assume_init()))
        }
    }
}

impl std::fmt::Debug for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let err_str = self.error_string().unwrap();
        f.debug_tuple("CudaError")
            .field(&self.0)
            .field(&err_str)
            .finish()
    }
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for CudaError {}

/// TODO
pub fn init() -> Result<(), CudaError> {
    unsafe { sys::cuInit(0).result() }
}

pub mod device {
    //! TODO

    use super::{sys, CudaError};
    use std::mem::MaybeUninit;

    /// TODO
    pub fn get(ordinal: std::os::raw::c_int) -> Result<sys::CUdevice, CudaError> {
        let mut dev: sys::CUdevice = 0;
        unsafe { sys::cuDeviceGet((&mut dev) as *mut sys::CUdevice, ordinal).result()? }
        Ok(dev)
    }

    /// TODO
    pub unsafe fn primary_ctx_retain(dev: sys::CUdevice) -> Result<sys::CUcontext, CudaError> {
        let mut ctx = MaybeUninit::uninit();
        sys::cuDevicePrimaryCtxRetain(ctx.as_mut_ptr(), dev).result()?;
        Ok(ctx.assume_init())
    }

    /// TODO
    pub unsafe fn primary_ctx_release(dev: sys::CUdevice) -> Result<(), CudaError> {
        sys::cuDevicePrimaryCtxRelease_v2(dev).result()
    }
}

pub mod ctx {
    //! TODO
    use super::{sys, CudaError};

    /// TODO
    pub unsafe fn set_current(ctx: sys::CUcontext) -> Result<(), CudaError> {
        sys::cuCtxSetCurrent(ctx).result()
    }
}

pub mod stream {
    //! TODO
    use super::{sys, CudaError};
    use std::mem::MaybeUninit;

    /// TODO
    pub enum StreamKind {
        Default,
        NonBlocking,
    }

    impl StreamKind {
        fn flags(self) -> sys::CUstream_flags {
            match self {
                Self::Default => sys::CUstream_flags::CU_STREAM_DEFAULT,
                Self::NonBlocking => sys::CUstream_flags::CU_STREAM_NON_BLOCKING,
            }
        }
    }

    /// TODO
    pub fn null() -> sys::CUstream {
        std::ptr::null_mut()
    }

    /// TODO
    pub fn create(kind: StreamKind) -> Result<sys::CUstream, CudaError> {
        let mut stream = MaybeUninit::uninit();
        unsafe {
            sys::cuStreamCreate(stream.as_mut_ptr(), kind.flags() as u32).result()?;
            Ok(stream.assume_init())
        }
    }

    /// TODO
    pub unsafe fn synchronize(stream: sys::CUstream) -> Result<(), CudaError> {
        sys::cuStreamSynchronize(stream).result()
    }

    /// TODO
    pub unsafe fn destroy(stream: sys::CUstream) -> Result<(), CudaError> {
        sys::cuStreamDestroy_v2(stream).result()
    }
}

/// TODO
pub unsafe fn malloc<T>() -> Result<sys::CUdeviceptr, CudaError> {
    let mut dev_ptr = MaybeUninit::uninit();
    sys::cuMemAlloc_v2(dev_ptr.as_mut_ptr(), size_of::<T>()).result()?;
    Ok(dev_ptr.assume_init())
}

/// TODO
pub unsafe fn malloc_async<T>(stream: sys::CUstream) -> Result<sys::CUdeviceptr, CudaError> {
    let mut dev_ptr = MaybeUninit::uninit();
    sys::cuMemAllocAsync(dev_ptr.as_mut_ptr(), size_of::<T>(), stream).result()?;
    Ok(dev_ptr.assume_init())
}

/// TODO
pub unsafe fn free(dptr: sys::CUdeviceptr) -> Result<(), CudaError> {
    sys::cuMemFree_v2(dptr).result()
}

/// TODO
pub unsafe fn free_async(dptr: sys::CUdeviceptr, stream: sys::CUstream) -> Result<(), CudaError> {
    sys::cuMemFreeAsync(dptr, stream).result()
}

/// TODO
pub unsafe fn memset_d8<T>(
    dptr: sys::CUdeviceptr,
    uc: std::os::raw::c_uchar,
) -> Result<(), CudaError> {
    sys::cuMemsetD8_v2(dptr, uc, size_of::<T>()).result()
}

/// TODO
pub unsafe fn memset_d8_async<T>(
    dptr: sys::CUdeviceptr,
    uc: std::os::raw::c_uchar,
    stream: sys::CUstream,
) -> Result<(), CudaError> {
    sys::cuMemsetD8Async(dptr, uc, size_of::<T>(), stream).result()
}

/// TODO
pub unsafe fn memcpy_htod<T>(dst: sys::CUdeviceptr, src: &T) -> Result<(), CudaError> {
    sys::cuMemcpyHtoD_v2(dst, src as *const T as *const _, size_of::<T>()).result()
}

/// TODO
pub unsafe fn memcpy_htod_async<T>(
    dst: sys::CUdeviceptr,
    src: &T,
    stream: sys::CUstream,
) -> Result<(), CudaError> {
    sys::cuMemcpyHtoDAsync_v2(dst, src as *const T as *const _, size_of::<T>(), stream).result()
}

/// TODO
pub unsafe fn memcpy_dtoh<T>(dst: &mut T, src: sys::CUdeviceptr) -> Result<(), CudaError> {
    sys::cuMemcpyDtoH_v2(dst as *mut T as *mut _, src, size_of::<T>()).result()
}

/// TODO
pub unsafe fn memcpy_dtoh_async<T>(
    dst: &mut T,
    src: sys::CUdeviceptr,
    stream: sys::CUstream,
) -> Result<(), CudaError> {
    sys::cuMemcpyDtoHAsync_v2(dst as *mut T as *mut _, src, size_of::<T>(), stream).result()
}

/// TODO
pub unsafe fn memcpy_dtod<T>(
    dst: sys::CUdeviceptr,
    src: sys::CUdeviceptr,
) -> Result<(), CudaError> {
    sys::cuMemcpyDtoD_v2(dst, src, size_of::<T>()).result()
}

/// TODO
pub unsafe fn memcpy_dtod_async<T>(
    dst: sys::CUdeviceptr,
    src: sys::CUdeviceptr,
    stream: sys::CUstream,
) -> Result<(), CudaError> {
    sys::cuMemcpyDtoDAsync_v2(dst, src, size_of::<T>(), stream).result()
}

pub mod module {
    //! TODO

    use super::{sys, CudaError};
    use std::{ffi::CString, mem::MaybeUninit};

    /// TODO
    pub fn load<S: AsRef<str>>(fname: S) -> Result<sys::CUmodule, CudaError> {
        let fname_cstr = CString::new(fname.as_ref()).unwrap();
        let fname_ptr = fname_cstr.as_c_str().as_ptr();
        let mut module = MaybeUninit::uninit();
        unsafe {
            sys::cuModuleLoad(module.as_mut_ptr(), fname_ptr).result()?;
            Ok(module.assume_init())
        }
    }

    /// TODO
    pub unsafe fn load_data(image: *const std::ffi::c_void) -> Result<sys::CUmodule, CudaError> {
        let mut module = MaybeUninit::uninit();
        sys::cuModuleLoadData(module.as_mut_ptr(), image).result()?;
        Ok(module.assume_init())
    }

    /// TODO
    pub unsafe fn get_function<S: AsRef<str>>(
        module: sys::CUmodule,
        name: S,
    ) -> Result<sys::CUfunction, CudaError> {
        let name_cstr = CString::new(name.as_ref()).unwrap();
        let name_ptr = name_cstr.as_c_str().as_ptr();
        let mut func = MaybeUninit::uninit();
        sys::cuModuleGetFunction(func.as_mut_ptr(), module, name_ptr).result()?;
        Ok(func.assume_init())
    }

    /// TODO
    pub unsafe fn unload(module: sys::CUmodule) -> Result<(), CudaError> {
        sys::cuModuleUnload(module).result()
    }
}

/// TODO
pub unsafe fn launch_kernel(
    f: sys::CUfunction,
    grid_dim: (c_uint, c_uint, c_uint),
    block_dim: (c_uint, c_uint, c_uint),
    shared_mem_bytes: c_uint,
    stream: sys::CUstream,
    kernel_params: &mut [*mut c_void],
) -> Result<(), CudaError> {
    sys::cuLaunchKernel(
        f,
        grid_dim.0,
        grid_dim.1,
        grid_dim.2,
        block_dim.0,
        block_dim.1,
        block_dim.2,
        shared_mem_bytes,
        stream,
        kernel_params.as_mut_ptr(),
        std::ptr::null_mut(),
    )
    .result()
}
