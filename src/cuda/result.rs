//! A thin wrapper around [sys].
//!
//! While all the functions here will return [Result], they are
//! mostly all still unsafe because order of operations
//! really matters.
//!
//! This also only exposes the `*_async` version of functions
//! because mixing the two is confusing and even more unsafe.
//!
//! This module also groups functions into sub-modules
//! to make naming easier. For example [sys::cuStreamCreate()]
//! turns into [stream::create()], where [stream] is a module.

use super::sys;
use std::{
    ffi::{c_void, CStr},
    mem::{size_of, MaybeUninit},
    os::raw::c_uint,
};

/// Wrapper around [sys::CUresult]. See
/// nvidia's [CUresult docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9)
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct CudaError(pub sys::CUresult);

impl sys::CUresult {
    /// Transforms into a [Result] of [CudaError]
    pub fn result(self) -> Result<(), CudaError> {
        match self {
            sys::CUresult::CUDA_SUCCESS => Ok(()),
            _ => Err(CudaError(self)),
        }
    }
}

impl CudaError {
    /// Gets the name for this error.
    ///
    /// See [cuGetErrorName() docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__ERROR.html#group__CUDA__ERROR_1g2c4ac087113652bb3d1f95bf2513c468)
    pub fn error_name(&self) -> Result<&CStr, CudaError> {
        let mut err_str = MaybeUninit::uninit();
        unsafe {
            sys::cuGetErrorName(self.0, err_str.as_mut_ptr()).result()?;
            Ok(CStr::from_ptr(err_str.assume_init()))
        }
    }

    /// Gets the error string for this error.
    ///
    /// See [cuGetErrorString() docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__ERROR.html#group__CUDA__ERROR_1g72758fcaf05b5c7fac5c25ead9445ada)
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

/// Initializes the CUDA driver API.
/// **MUST BE CALLED BEFORE ANYTHING ELSE**
///
/// See [cuInit() docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__INITIALIZE.html#group__CUDA__INITIALIZE_1g0a2f1517e1bd8502c7194c3a8c134bc3)
pub fn init() -> Result<(), CudaError> {
    unsafe { sys::cuInit(0).result() }
}

pub mod device {
    //! Device management functions (`cuDevice*`).
    //!
    //! See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE)

    use super::{sys, CudaError};
    use std::mem::MaybeUninit;

    /// Get a device for a specific ordinal.
    /// See [cuDeviceGet() docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g8bdd1cc7201304b01357b8034f6587cb).
    pub fn get(ordinal: std::os::raw::c_int) -> Result<sys::CUdevice, CudaError> {
        let mut dev = MaybeUninit::uninit();
        unsafe {
            sys::cuDeviceGet(dev.as_mut_ptr(), ordinal).result()?;
            Ok(dev.assume_init())
        }
    }

    /// Gets the number of available devices.
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g52b5ce05cb8c5fb6831b2c0ff2887c74)
    pub fn get_count() -> Result<std::os::raw::c_int, CudaError> {
        let mut count = MaybeUninit::uninit();
        unsafe {
            sys::cuDeviceGetCount(count.as_mut_ptr()).result()?;
            Ok(count.assume_init())
        }
    }

    /// Returns the total amount of memory in bytes on the device.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gc6a0d6551335a3780f9f3c967a0fde5d)
    ///
    /// # Safety
    /// Must be a device returned from [get].
    pub unsafe fn total_mem(dev: sys::CUdevice) -> Result<usize, CudaError> {
        let mut bytes = MaybeUninit::uninit();
        sys::cuDeviceTotalMem_v2(bytes.as_mut_ptr(), dev).result()?;
        Ok(bytes.assume_init())
    }
}

pub mod primary_ctx {
    //! Primary context management functions (`cuDevicePrimaryCtx*`).
    //!
    //! See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX)

    use super::{sys, CudaError};
    use std::mem::MaybeUninit;

    /// Creates a primary context on the device and pushes it onto the primary context stack.
    /// Call [release] to free it.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1g9051f2d5c31501997a6cb0530290a300)
    ///
    /// # Safety
    ///
    /// This is only safe with a device that was returned from [super::device::get].
    pub unsafe fn retain(dev: sys::CUdevice) -> Result<sys::CUcontext, CudaError> {
        let mut ctx = MaybeUninit::uninit();
        sys::cuDevicePrimaryCtxRetain(ctx.as_mut_ptr(), dev).result()?;
        Ok(ctx.assume_init())
    }

    /// Release a reference to the current primary context.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1gf2a8bc16f8df0c88031f6a1ba3d6e8ad).
    ///
    /// # Safety
    ///
    /// This is only safe with a device that was returned from [super::device::get].
    pub unsafe fn release(dev: sys::CUdevice) -> Result<(), CudaError> {
        sys::cuDevicePrimaryCtxRelease_v2(dev).result()
    }
}

pub mod ctx {
    //! Context management functions (`cuCtx*`).
    //!
    //! See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX)

    use super::{sys, CudaError};

    /// Binds the specified CUDA context to the calling CPU thread.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1gbe562ee6258b4fcc272ca6478ca2a2f7)
    ///
    /// # Safety
    ///
    /// This has weird behavior depending on the value of `ctx`. See cuda docs for more info.
    /// In general this should only be called with an already initialized context,
    /// and one that wasn't already freed.
    pub unsafe fn set_current(ctx: sys::CUcontext) -> Result<(), CudaError> {
        sys::cuCtxSetCurrent(ctx).result()
    }
}

pub mod stream {
    //! Stream management functions (`cuStream*`).
    //!
    //! See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM).

    use super::{sys, CudaError};
    use std::mem::MaybeUninit;

    /// The kind of stream to initialize.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4)
    pub enum StreamKind {
        /// From cuda docs:
        /// > Default stream creation flag.
        Default,

        /// From cuda docs:
        /// > Specifies that work running in the created stream
        /// > may run concurrently with work in stream 0 (the NULL stream),
        /// > and that the created stream should perform no implicit
        /// > synchronization with stream 0.
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

    /// The null stream, which is just a null pointer. **Recommend not using this.**
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/stream-sync-behavior.html#stream-sync-behavior__default-stream)
    pub fn null() -> sys::CUstream {
        std::ptr::null_mut()
    }

    /// Creates a stream with the specified kind.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4)
    pub fn create(kind: StreamKind) -> Result<sys::CUstream, CudaError> {
        let mut stream = MaybeUninit::uninit();
        unsafe {
            sys::cuStreamCreate(stream.as_mut_ptr(), kind.flags() as u32).result()?;
            Ok(stream.assume_init())
        }
    }

    /// Wait until a stream's tasks are completed.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad)
    ///
    /// # Safety
    ///
    /// This should only be called with stream created by [create] and not already
    /// destroyed. This follows default stream semantics, see relevant cuda docs.
    pub unsafe fn synchronize(stream: sys::CUstream) -> Result<(), CudaError> {
        sys::cuStreamSynchronize(stream).result()
    }

    /// Destroys a stream.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758)
    ///
    /// # Safety
    ///
    /// This should only be called with stream created by [create] and not already
    /// destroyed. This follows default stream semantics, see relevant cuda docs.
    pub unsafe fn destroy(stream: sys::CUstream) -> Result<(), CudaError> {
        sys::cuStreamDestroy_v2(stream).result()
    }
}

/// Allocates memory with stream ordered semantics.
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g13413273e84a641bce1929eae9e6501f)
///
/// # Safety
/// 1. The stream should be an already created stream.
/// 2. The memory return by this is unset, which may be invalid for `T`.
/// 3. All uses of this memory must be on the same stream.
pub unsafe fn malloc_async<T>(stream: sys::CUstream) -> Result<sys::CUdeviceptr, CudaError> {
    let mut dev_ptr = MaybeUninit::uninit();
    sys::cuMemAllocAsync(dev_ptr.as_mut_ptr(), size_of::<T>(), stream).result()?;
    Ok(dev_ptr.assume_init())
}

/// Frees memory with stream ordered semantics.
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g41acf4131f672a2a75cd93d3241f10cf)
///
/// # Safety
/// 1. The stream should be an already created stream.
/// 2. The memory should have been allocated on this stream.
/// 3. The memory should not have been freed already (double free)
pub unsafe fn free_async(dptr: sys::CUdeviceptr, stream: sys::CUstream) -> Result<(), CudaError> {
    sys::cuMemFreeAsync(dptr, stream).result()
}

/// Sets device memory with stream ordered semantics.
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gaef08a7ccd61112f94e82f2b30d43627)
///
/// # Safety
/// 1. The resulting memory pattern may not be valid for `T`.
/// 2. The device pointer should not have been freed already (double free)
/// 3. The stream should be the stream the memory was allocated on.
pub unsafe fn memset_d8_async<T>(
    dptr: sys::CUdeviceptr,
    uc: std::os::raw::c_uchar,
    stream: sys::CUstream,
) -> Result<(), CudaError> {
    sys::cuMemsetD8Async(dptr, uc, size_of::<T>(), stream).result()
}

/// Copies memory from Host to Device with stream ordered semantics.
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169)
///
/// # Safety
/// **This function is asynchronous** in most cases, so the data from `src`
/// will be copied at a later point after this function returns.
///
/// 1. `T` must be the type that device pointer was allocated with.
/// 2. The device pointer should not have been freed already (double free)
/// 3. The stream should be the stream the memory was allocated on.
pub unsafe fn memcpy_htod_async<T>(
    dst: sys::CUdeviceptr,
    src: &T,
    stream: sys::CUstream,
) -> Result<(), CudaError> {
    sys::cuMemcpyHtoDAsync_v2(dst, src as *const T as *const _, size_of::<T>(), stream).result()
}

/// Copies memory from Device to Host with stream ordered semantics.
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362)
///
/// # Safety
/// **This function is asynchronous** in most cases, so `dst` will be
/// mutated at a later point after this function returns.
///
/// 1. `T` must be the type that device pointer was allocated with.
/// 2. The device pointer should not have been freed already (double free)
/// 3. The stream should be the stream the memory was allocated on.
pub unsafe fn memcpy_dtoh_async<T>(
    dst: &mut T,
    src: sys::CUdeviceptr,
    stream: sys::CUstream,
) -> Result<(), CudaError> {
    sys::cuMemcpyDtoHAsync_v2(dst as *mut T as *mut _, src, size_of::<T>(), stream).result()
}

/// Copies memory from Device to Device with stream ordered semantics.
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8)
///
/// # Safety
/// 1. `T` must be the type that BOTH device pointers were allocated with.
/// 2. Neither device pointer should not have been freed already (double free)
/// 3. The stream should be the stream the memory was allocated on.
pub unsafe fn memcpy_dtod_async<T>(
    dst: sys::CUdeviceptr,
    src: sys::CUdeviceptr,
    stream: sys::CUstream,
) -> Result<(), CudaError> {
    sys::cuMemcpyDtoDAsync_v2(dst, src, size_of::<T>(), stream).result()
}

pub mod module {
    //! Module management functions (`cuModule*`).
    //!
    //! See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE)

    use super::{sys, CudaError};
    use std::{ffi::CString, mem::MaybeUninit};

    /// Loads a compute module from a given file.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g366093bd269dafd0af21f1c7d18115d3)
    pub fn load(fname: CString) -> Result<sys::CUmodule, CudaError> {
        let fname_ptr = fname.as_c_str().as_ptr();
        let mut module = MaybeUninit::uninit();
        unsafe {
            sys::cuModuleLoad(module.as_mut_ptr(), fname_ptr).result()?;
            Ok(module.assume_init())
        }
    }

    /// Load a module's data:
    ///
    /// > The pointer may be obtained by mapping a cubin or PTX or fatbin file,
    /// > passing a cubin or PTX or fatbin file as a NULL-terminated text string,
    /// > or incorporating a cubin or fatbin object into the executable resources
    /// > and using operating system calls such as Windows FindResource() to obtain the pointer.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g04ce266ce03720f479eab76136b90c0b)
    ///
    /// # Safety
    /// The image must be properly formed pointer
    pub unsafe fn load_data(image: *const std::ffi::c_void) -> Result<sys::CUmodule, CudaError> {
        let mut module = MaybeUninit::uninit();
        sys::cuModuleLoadData(module.as_mut_ptr(), image).result()?;
        Ok(module.assume_init())
    }

    /// Returns a function handle from the given module.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1ga52be009b0d4045811b30c965e1cb2cf)
    ///
    /// # Safety
    /// `module` must be a properly allocated and not freed module.
    pub unsafe fn get_function(
        module: sys::CUmodule,
        name: CString,
    ) -> Result<sys::CUfunction, CudaError> {
        let name_ptr = name.as_c_str().as_ptr();
        let mut func = MaybeUninit::uninit();
        sys::cuModuleGetFunction(func.as_mut_ptr(), module, name_ptr).result()?;
        Ok(func.assume_init())
    }

    /// Unloads a module.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g8ea3d716524369de3763104ced4ea57b)
    ///
    /// # Safety
    /// `module` must not have be unloaded already.
    pub unsafe fn unload(module: sys::CUmodule) -> Result<(), CudaError> {
        sys::cuModuleUnload(module).result()
    }
}

/// Launches a cuda functions
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15)
///
/// # Safety
/// This method is **very unsafe**.
///
/// 1. The cuda function must be a valid handle returned from a non-unloaded module.
/// 2. This is asynchronous, so the results of calling this function happen
/// at a later point after this function returns.
/// 3. All parameters used for this kernel should have been allocated by stream (I think?)
/// 4. The cuda kernel has mutable access to every parameter, that means every parameter
/// can change at a later point after callign this function. *Even non-mutable references*.
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
