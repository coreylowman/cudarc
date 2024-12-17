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
//! to make naming easier. For example [sys::cudaStreamCreate()]
//! turns into [stream::create()], where [stream] is a module.

use super::sys::{self, lib};
use crate::driver::sys::cudaError_enum;
use core::ffi::{c_uchar, c_void};
use std::ffi::CStr;
use std::mem::MaybeUninit;

/// Wrapper around `cudaError_t`. See
/// NVIDIA's [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g6cfab9c404b2678e6cd1de91ff24e799)
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct RuntimeError(pub sys::cudaError_t);

impl sys::cudaError_t {
    #[inline]
    pub fn result(self) -> Result<(), RuntimeError> {
        match self {
            sys::cudaError_t::cudaSuccess => Ok(()),
            _ => Err(RuntimeError(self)),
        }
    }
}

impl From<cudaError_enum> for RuntimeError {
    fn from(e: cudaError_enum) -> Self {
        match e {
            cudaError_enum::CUDA_SUCCESS => Self(sys::cudaError_t::cudaSuccess),
            cudaError_enum::CUDA_ERROR_INVALID_VALUE => {
                Self(sys::cudaError_t::cudaErrorInvalidValue)
            }
            cudaError_enum::CUDA_ERROR_OUT_OF_MEMORY => {
                Self(sys::cudaError_t::cudaErrorMemoryAllocation)
            }
            _ => Self(sys::cudaError_t::cudaErrorUnknown),
        }
    }
}

impl RuntimeError {
    /// Gets the name for this error.
    ///
    /// See [cudaGetErrorName() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html#group__CUDART__ERROR_1g52abef1749df1e2e0d99a07ffaf1f6cb)
    pub fn error_name(&self) -> Result<&CStr, RuntimeError> {
        unsafe {
            let err_str = lib().cudaGetErrorName(self.0);
            Ok(CStr::from_ptr(err_str))
        }
    }

    /// Gets the error string for this error.
    ///
    /// See [cudaGetErrorString() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html#group__CUDART__ERROR_1g6cfab9c404b2678e6cd1de91ff24e799)
    pub fn error_string(&self) -> Result<&CStr, RuntimeError> {
        unsafe {
            let err_str = lib().cudaGetErrorString(self.0);
            Ok(CStr::from_ptr(err_str))
        }
    }
}

impl std::fmt::Debug for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let err_str = self.error_string().unwrap();
        f.debug_tuple("RuntimeError")
            .field(&self.0)
            .field(&err_str)
            .finish()
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for RuntimeError {}

/// CUDA device flags.
///
/// See [cudaSetDeviceFlags() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g69e73c7dda3fc05306ae7c811a690fac)
pub enum CudaDeviceFlags {
    CudaDeviceScheduleAuto = sys::cudaDeviceScheduleAuto as isize,
    CudaDeviceScheduleSpin = sys::cudaDeviceScheduleSpin as isize,
    CudaDeviceScheduleYield = sys::cudaDeviceScheduleYield as isize,
    CudaDeviceScheduleBlockingSync = sys::cudaDeviceScheduleBlockingSync as isize,
    CudaDeviceMapHost = sys::cudaDeviceMapHost as isize,
    CudaDeviceLmemResizeToMax = sys::cudaDeviceLmemResizeToMax as isize,
    #[cfg(not(any(
        feature = "cuda-12000",
        feature = "cuda-11080",
        feature = "cuda-11070",
        feature = "cuda-11060",
        feature = "cuda-11050",
        feature = "cuda-11040"
    )))]
    CudaDeviceSyncMemops = sys::cudaDeviceSyncMemops as isize,
}

pub mod version {
    //! CUDA Runtime API version functions (`cudaRuntimeGetVersion`).
    //!
    //! See [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__VERSION.html)

    use super::{lib, RuntimeError};

    /// Returns the CUDA Runtime version.
    ///
    /// See [cudaRuntimeGetVersion() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART____VERSION.html#group__CUDART____VERSION_1g0e3952c7802fd730432180f1f4a6cdc6)
    pub fn get() -> Result<i32, RuntimeError> {
        let mut version = 0;
        unsafe {
            lib().cudaRuntimeGetVersion(&mut version).result()?;
        }
        Ok(version)
    }

    pub fn get_runtime_version() -> Result<i32, RuntimeError> {
        get()
    }

    /// Returns the CUDA Driver version.
    ///
    /// See [cudaDriverGetVersion() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART____VERSION.html#group__CUDART____VERSION_1g8a06ee14a0551606b7c780084d5564ab)
    pub fn get_driver_version() -> Result<i32, RuntimeError> {
        let mut version = 0;
        unsafe {
            lib().cudaDriverGetVersion(&mut version).result()?;
        }
        Ok(version)
    }
}

pub mod device {
    //! Device management functions (`cudaGetDevice*`).
    //!
    //! See [cudart docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDA__DEVICE)

    use super::{lib, sys, RuntimeError};
    use core::ffi::{c_int, c_void};
    use std::mem::MaybeUninit;

    /// Sets the current device for the calling host thread.
    /// If not already done so, it will initialize the structures and context for the device.
    ///
    /// See [cudaSetDevice(0) docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g073d3dcb31d45fc96c8ff52833a890d0)
    ///
    /// # Safety
    /// 1. Must be a valid ordinal between 0 to (cudaGetDeviceCount() - 1).
    pub fn set(ordinal: i32) -> Result<(), RuntimeError> {
        unsafe { lib().cudaSetDevice(ordinal).result() }
    }

    /// Get the device for the calling host thread.
    /// See [cudaGetDevice() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g073d3dcb31d45fc96c8ff52833a890d0)
    pub fn get() -> Result<i32, RuntimeError> {
        let mut device = MaybeUninit::uninit();
        unsafe {
            lib().cudaGetDevice(device.as_mut_ptr()).result()?;
            Ok(device.assume_init())
        }
    }

    /// Frees device memory.
    ///
    /// See [cudaFree() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g9b0ec4fcdc8894cf65d308d2b1f00223)
    ///
    /// # Safety
    /// 1. Must be a device pointer returned from `cudaMalloc()`, `cudaMallocPitch()`,
    ///    `cudaMallocManaged()`, `cudaMallocAsync()`, `cudaMallocFromPoolAsync()`.
    ///
    /// # Note
    /// if device_ptr is 0, no operation is performed.
    pub unsafe fn free(device_ptr: *mut c_void) -> Result<(), RuntimeError> {
        unsafe { lib().cudaFree(device_ptr).result() }
    }

    /// Get a device prop for a specific ordinal.
    /// See [cudaGetDeviceProperties_v2() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g073d3dcb31d45fc96c8ff52833a890d0)
    pub fn get_device_prop(ordinal: c_int) -> Result<sys::cudaDeviceProp, RuntimeError> {
        let mut prop = MaybeUninit::uninit();

        #[cfg(not(any(
            feature = "cuda-11080",
            feature = "cuda-11070",
            feature = "cuda-11060",
            feature = "cuda-11050",
            feature = "cuda-11040"
        )))]
        unsafe {
            lib()
                .cudaGetDeviceProperties_v2(prop.as_mut_ptr(), ordinal)
                .result()?;
            Ok(prop.assume_init())
        }
        #[cfg(any(
            feature = "cuda-11080",
            feature = "cuda-11070",
            feature = "cuda-11060",
            feature = "cuda-11050",
            feature = "cuda-11040"
        ))]
        unsafe {
            lib()
                .cudaGetDeviceProperties(prop.as_mut_ptr(), ordinal)
                .result()?;
            Ok(prop.assume_init())
        }
    }

    /// Gets the number of available devices.
    /// See [cudaGetDeviceCount() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g9ac74732f1585d3a1a99bb0d3ea7fc48)
    pub fn get_count() -> Result<c_int, RuntimeError> {
        let mut count = MaybeUninit::uninit();
        unsafe {
            lib().cudaGetDeviceCount(count.as_mut_ptr()).result()?;
            Ok(count.assume_init())
        }
    }

    /// Returns the total amount of memory in bytes on the device.
    ///
    /// See [cudaMemGetInfo docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g0f1786e548dd2cd70152f8b8b95cc84a)
    ///
    pub fn total_mem() -> Result<usize, RuntimeError> {
        let mut bytes = MaybeUninit::uninit();
        unsafe {
            lib()
                .cudaMemGetInfo(std::ptr::null_mut(), bytes.as_mut_ptr())
                .result()?;
            Ok(bytes.assume_init())
        }
    }

    /// Returns the amount of free memory in bytes on the device.
    ///
    /// See [cudaMemGetInfo docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g0f1786e548dd2cd70152f8b8b95cc84a)
    ///
    pub fn free_mem() -> Result<usize, RuntimeError> {
        let mut bytes = MaybeUninit::uninit();
        unsafe {
            lib()
                .cudaMemGetInfo(bytes.as_mut_ptr(), std::ptr::null_mut())
                .result()?;
            Ok(bytes.assume_init())
        }
    }

    /// Get an attribute of a device.
    ///
    /// See [cudaGetDeviceAttribute() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g42000991dfc9b8dc0f17fcd8eb6cf3ab)
    ///
    /// # Safety
    /// Must be a valid attribute and device ordinal.
    pub unsafe fn get_attribute(
        ordinal: c_int,
        attr: sys::cudaDeviceAttr,
    ) -> Result<i32, RuntimeError> {
        let mut value = MaybeUninit::uninit();
        unsafe {
            lib()
                .cudaDeviceGetAttribute(value.as_mut_ptr(), attr, ordinal)
                .result()?;
            Ok(value.assume_init())
        }
    }

    /// Reset the device and release all memory allocations.
    ///
    /// See [cudaDeviceReset() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g073d3dcb31d45fc96c8ff52833a890d0)
    /// # Safety
    /// Must be called if cudaSet*Device() or cudaInitDevice() has been called.
    pub unsafe fn reset() -> Result<(), RuntimeError> {
        lib().cudaDeviceReset().result()
    }
}

pub mod function {
    use super::{lib, sys, RuntimeError};
    use std::mem::MaybeUninit;
    use std::os::raw::c_void;

    /// Sets the specific attribute of a CUDA function.
    ///
    /// See [cudaFuncSetAttribute() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g317e77d2657abf915fd9ed03e75f3eb0)
    ///
    /// # Safety
    /// Function must exist.
    pub unsafe fn set_function_attribute(
        func: *const c_void,
        attribute: sys::cudaFuncAttribute,
        value: i32,
    ) -> Result<(), RuntimeError> {
        lib().cudaFuncSetAttribute(func, attribute, value).result()
    }

    /// Sets the specific attribute of a CUDA function.
    ///
    /// See [cudaFuncSetAttribute() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g317e77d2657abf915fd9ed03e75f3eb0)
    ///
    /// # Safety
    /// Function must exist.
    pub unsafe fn get_function_attributes(
        func: *const c_void,
    ) -> Result<sys::cudaFuncAttributes, RuntimeError> {
        let mut attr = MaybeUninit::uninit();
        unsafe {
            lib()
                .cudaFuncGetAttributes(attr.as_mut_ptr(), func)
                .result()?;
            Ok(attr.assume_init())
        }
    }

    /// Retrieves the function pointer from a symbol.
    ///
    /// See [cudaGetFuncBySymbol() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DRIVER.html#group__CUDART__DRIVER_1gaba6f8d01e745f0c8d8776ceb18be617)
    ///
    /// # Safety
    /// Function must exist.
    pub unsafe fn get_function_by_symbol(
        symbol_ptr: *const c_void,
    ) -> Result<sys::cudaFunction_t, RuntimeError> {
        let mut func = MaybeUninit::uninit();
        unsafe {
            lib()
                .cudaGetFuncBySymbol(func.as_mut_ptr(), symbol_ptr)
                .result()?;
            Ok(func.assume_init())
        }
    }
}

pub mod occupancy {
    use core::ffi::{c_int, c_uint, c_void};
    use std::mem::MaybeUninit;

    use super::{lib, sys, RuntimeError};

    // Use the functions from the driver module
    use crate::driver::result::occupancy::{
        max_potential_block_size as driver_max_potential_block_size,
        max_potential_block_size_with_flags as driver_max_potential_block_size_with_flags,
    };
    use crate::driver::{sys as driver_sys, DriverError};

    /// Returns dynamic shared memory available per block when launching numBlocks blocks on SM.
    ///
    /// See [cudaOccupancyAvailableDynamicSMemPerBlock() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OCCUPANCY.html#group__CUDART__OCCUPANCY_1g3017bec8ddb4951e89f6ba4c259bb091)
    ///
    /// # Safety
    /// Function must exist.
    pub unsafe fn available_dynamic_shared_mem_per_block(
        f: *const c_void,
        num_blocks: c_int,
        block_size: c_int,
    ) -> Result<usize, RuntimeError> {
        let mut dynamic_smem_size = MaybeUninit::uninit();
        lib()
            .cudaOccupancyAvailableDynamicSMemPerBlock(
                dynamic_smem_size.as_mut_ptr(),
                f,
                num_blocks,
                block_size,
            )
            .result()?;
        Ok(dynamic_smem_size.assume_init())
    }

    /// Returns occupancy of a function.
    ///
    /// See [cudaOccupancyMaxActiveBlocksPerMultiprocessor() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OCCUPANCY.html#group__CUDART__OCCUPANCY_1ge99bee88c427b3f8ffa8ec3e43fd877d)
    ///
    /// # Safety
    /// Function must exist.
    pub unsafe fn max_active_block_per_multiprocessor(
        f: *const c_void,
        block_size: c_int,
        dynamic_smem_size: usize,
    ) -> Result<i32, RuntimeError> {
        let mut num_blocks = MaybeUninit::uninit();
        lib()
            .cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                num_blocks.as_mut_ptr(),
                f,
                block_size,
                dynamic_smem_size,
            )
            .result()?;
        Ok(num_blocks.assume_init())
    }

    /// Returns occupancy of a function.
    ///
    /// See [cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OCCUPANCY.html#group__CUDART__OCCUPANCY_1ge2255f3637784624ea99a6d3c7885ca0)
    ///
    /// # Safety
    /// Function must exist. No invalid flags.
    pub unsafe fn max_active_block_per_multiprocessor_with_flags(
        f: *const c_void,
        block_size: c_int,
        dynamic_smem_size: usize,
        flags: c_uint,
    ) -> Result<i32, RuntimeError> {
        let mut num_blocks = MaybeUninit::uninit();
        lib()
            .cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
                num_blocks.as_mut_ptr(),
                f,
                block_size,
                dynamic_smem_size,
                flags,
            )
            .result()?;
        Ok(num_blocks.assume_init())
    }

    /// Suggest a launch configuration with reasonable occupancy.
    ///
    /// Returns (min_grid_size, block_size)
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY_1gf179c4ab78962a8468e41c3f57851f03)
    ///
    /// # Safety
    /// Function must exist and the shared memory function must be correct.  No invalid flags.
    ///
    /// # Note
    /// This function is a wrapper around the driver function. cudaOccupancyMaxPotentialBlockSize
    /// is a template inline function which cannot be generated with bindgen
    pub unsafe fn max_potential_block_size(
        f: sys::cudaFunction_t,
        block_size_to_dynamic_smem_size: driver_sys::CUoccupancyB2DSize,
        dynamic_smem_size: usize,
        block_size_limit: c_int,
    ) -> Result<(i32, i32), DriverError> {
        driver_max_potential_block_size(
            f as driver_sys::CUfunction,
            block_size_to_dynamic_smem_size,
            dynamic_smem_size,
            block_size_limit,
        )
    }

    /// Suggest a launch configuration with reasonable occupancy.
    ///
    /// Returns (min_grid_size, block_size)
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY_1g04c0bb65630f82d9b99a5ca0203ee5aa)
    ///
    /// # Safety
    /// Function must exist and the shared memory function must be correct.  No invalid flags.
    ///
    /// # Note
    /// This function is a wrapper around the driver function.
    /// cudaOccupancyMaxPotentialBlockSizeWithFlags is a template inline function which cannot be
    /// generated with bindgen
    pub unsafe fn max_potential_block_size_with_flags(
        f: sys::cudaFunction_t,
        block_size_to_dynamic_smem_size: driver_sys::CUoccupancyB2DSize,
        dynamic_smem_size: usize,
        block_size_limit: c_int,
        flags: c_uint,
    ) -> Result<(i32, i32), DriverError> {
        driver_max_potential_block_size_with_flags(
            f as driver_sys::CUfunction,
            block_size_to_dynamic_smem_size,
            dynamic_smem_size,
            block_size_limit,
            flags,
        )
    }
}

pub mod stream {
    //! Stream management functions (`cudaStream*`).
    //!
    //! See [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html).

    use super::{lib, sys, RuntimeError};
    use std::mem::MaybeUninit;

    /// The kind of stream to initialize.
    ///
    /// See [cudaStreamCreateWithFlags() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g180dc77f922e6cb192b72e43835b8070)
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
        fn flags(self) -> u32 {
            match self {
                Self::Default => sys::cudaStreamDefault,
                Self::NonBlocking => sys::cudaStreamNonBlocking,
            }
        }
    }

    /// Creates a stream with the specified kind.
    ///
    /// See [cudaStreamCreateWithFlags() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g180dc77f922e6cb192b72e43835b8070)
    pub fn create(kind: StreamKind) -> Result<sys::cudaStream_t, RuntimeError> {
        let mut stream = MaybeUninit::uninit();
        unsafe {
            lib().cudaStreamCreateWithFlags(stream.as_mut_ptr(), kind.flags());
            Ok(stream.assume_init())
        }
    }

    /// Wait until a stream's tasks are completed.
    ///
    /// See [cudaStreamSynchronize() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g1817a344209a37f3d920c0f4a43fddab)
    ///
    /// # Safety
    ///
    /// This should only be called with stream created by [create] and not already
    /// destroyed. This follows default stream semantics, see relevant cuda docs.
    pub unsafe fn synchronize(stream: sys::cudaStream_t) -> Result<(), RuntimeError> {
        lib().cudaStreamSynchronize(stream).result()
    }

    /// Destroys a stream.
    ///
    /// See [cudaStreamDestroy() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gfda584f1788ca983cb21c5f4d2033a62)
    ///
    /// # Safety
    ///
    /// This should only be called with stream created by [create] and not already destroyed.
    pub unsafe fn destroy(stream: sys::cudaStream_t) -> Result<(), RuntimeError> {
        lib().cudaStreamDestroy(stream).result()
    }

    /// Make a compute stream wait on an event.
    ///
    /// See [cudaStreamWaitEvent() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g7840e3984799941a61839de40413d1d9)
    ///
    /// # Safety
    /// 1. Both stream and event must not have been freed already
    pub unsafe fn wait_event(
        stream: sys::cudaStream_t,
        event: sys::cudaEvent_t,
        flags: u32,
    ) -> Result<(), RuntimeError> {
        lib().cudaStreamWaitEvent(stream, event, flags).result()
    }
}

/// Allocates memory with stream ordered semantics.
///
/// See [cudaMallocAsync() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html#group__CUDART__MEMORY__POOLS_1gbbf70065888d61853c047513baa14081)
///
/// # Safety
/// 1. The stream should be an already created stream.
/// 2. The memory return by this is unset, which may be invalid for `T`.
/// 3. All uses of this memory must be on the same stream.
pub unsafe fn malloc_async(
    stream: sys::cudaStream_t,
    num_bytes: usize,
) -> Result<*mut c_void, RuntimeError> {
    let mut dev_ptr = MaybeUninit::uninit();
    lib()
        .cudaMallocAsync(dev_ptr.as_mut_ptr(), num_bytes, stream)
        .result()?;
    Ok(dev_ptr.assume_init())
}

/// Allocates memory
///
/// See [cudaMalloc() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g37d37965bfb4803b6d4e59ff26856356)
///
/// # Safety
/// 1. The memory return by this is unset, which may be invalid for `T`.
pub unsafe fn malloc_sync(num_bytes: usize) -> Result<*mut c_void, RuntimeError> {
    let mut dev_ptr = MaybeUninit::uninit();
    lib().cudaMalloc(dev_ptr.as_mut_ptr(), num_bytes).result()?;
    Ok(dev_ptr.assume_init())
}

/// Returns the amount of free and total memory in bytes on the device.
///
/// See [cudaMemGetInfo() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g0f1786e548dd2cd70152f8b8b95cc84a)
pub fn get_mem_info() -> Result<(usize, usize), RuntimeError> {
    let mut free = MaybeUninit::uninit();
    let mut total = MaybeUninit::uninit();
    unsafe {
        lib()
            .cudaMemGetInfo(free.as_mut_ptr(), total.as_mut_ptr())
            .result()?;
        Ok((free.assume_init(), total.assume_init()))
    }
}

/// Frees memory with stream ordered semantics.
///
/// See [cudaFreeAsync() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g5717a06f176c7d621c6364a4181fcbbf)
///
/// # Safety
/// 1. The stream should be an already created stream.
/// 2. The memory should have been allocated on this stream.
/// 3. The memory should not have been freed already (double free)
pub unsafe fn free_async(dptr: *mut c_void, stream: sys::cudaStream_t) -> Result<(), RuntimeError> {
    lib().cudaFreeAsync(dptr, stream).result()
}

/// Allocates memory
///
/// See [cudaFree() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g5717a06f176c7d621c6364a4181fcbbf)
///
/// # Safety
/// 1. The memory should have been allocated with malloc_sync
pub unsafe fn free_sync(dptr: *mut c_void) -> Result<(), RuntimeError> {
    lib().cudaFree(dptr).result()
}

/// Frees device memory.
///
/// See [cudaFree() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g9b0ec4fcdc8894cf65d308d2b1f00223)
///
/// # Safety
/// 1. Memory must only be freed once.
/// 2. All async accesses to this pointer must have been completed.
pub unsafe fn memory_free(device_ptr: *mut c_void) -> Result<(), RuntimeError> {
    lib().cudaFree(device_ptr).result()
}

/// Sets device memory with stream ordered semantics.
///
/// See [cudaMemsetAsync() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g7c9761e21d9f0999fd136c51e7b9b2a0)
///
/// # Safety
/// 1. The resulting memory pattern may not be valid for `T`.
/// 2. The device pointer should not have been freed already (double free)
/// 3. The stream should be the stream the memory was allocated on.
pub unsafe fn memset_d8_async(
    dptr: *mut c_void,
    uc: c_uchar,
    num_bytes: usize,
    stream: sys::cudaStream_t,
) -> Result<(), RuntimeError> {
    lib()
        .cudaMemsetAsync(dptr, uc as i32, num_bytes, stream)
        .result()
}

/// Sets device memory with stream ordered semantics.
///
/// See [cudaMemset() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gf7338650f7683c51ee26aadc6973c63a)
///
/// # Safety
/// 1. The resulting memory pattern may not be valid for `T`.
/// 2. The device pointer should not have been freed already (double free)
pub unsafe fn memset_d8_sync(
    dptr: *mut c_void,
    uc: c_uchar,
    num_bytes: usize,
) -> Result<(), RuntimeError> {
    lib().cudaMemset(dptr, uc as i32, num_bytes).result()
}

/// Copies memory from Host to Device with stream ordered semantics.
///
/// See [cudaMemcpyAsync() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g85073372f776b4c4d5f89f7124b7bf79)
///
/// # Safety
/// **This function is asynchronous** in most cases, so the data from `src`
/// will be copied at a later point after this function returns.
///
/// 1. `T` must be the type that device pointer was allocated with.
/// 2. The device pointer should not have been freed already (double free)
/// 3. The stream should be the stream the memory was allocated on.
/// 4. `src` must not be moved
pub unsafe fn memcpy_htod_async<T>(
    dst: *mut c_void,
    src: &[T],
    stream: sys::cudaStream_t,
) -> Result<(), RuntimeError> {
    lib()
        .cudaMemcpyAsync(
            dst,
            src.as_ptr() as *const c_void,
            std::mem::size_of_val(src),
            sys::cudaMemcpyKind::cudaMemcpyHostToDevice,
            stream,
        )
        .result()
}

/// Copies memory from Host to Device
///
/// See [cudaMemcpy() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gc263dbe6574220cc776b45438fc351e8)
///
/// # Safety
/// **This function is synchronous**
/// 1. `T` must be the type that device pointer was allocated with.
/// 2. The device pointer should not have been freed already (double free)
/// 3. `src` must not be moved
pub unsafe fn memcpy_htod_sync<T>(dst: *mut c_void, src: &[T]) -> Result<(), RuntimeError> {
    lib()
        .cudaMemcpy(
            dst,
            src.as_ptr() as *const c_void,
            std::mem::size_of_val(src),
            sys::cudaMemcpyKind::cudaMemcpyHostToDevice,
        )
        .result()
}

/// Copies memory from Device to Host with stream ordered semantics.
///
/// See [cudaMemcpyAsync() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g85073372f776b4c4d5f89f7124b7bf79)
///
/// # Safety
/// **This function is asynchronous** in most cases, so `dst` will be
/// mutated at a later point after this function returns.
///
/// 1. `T` must be the type that device pointer was allocated with.
/// 2. The device pointer should not have been freed already (double free)
/// 3. The stream should be the stream the memory was allocated on.
pub unsafe fn memcpy_dtoh_async<T>(
    dst: &mut [T],
    src: *const c_void,
    stream: sys::cudaStream_t,
) -> Result<(), RuntimeError> {
    lib()
        .cudaMemcpyAsync(
            dst.as_mut_ptr() as *mut c_void,
            src,
            std::mem::size_of_val(dst),
            sys::cudaMemcpyKind::cudaMemcpyDeviceToHost,
            stream,
        )
        .result()
}

/// Copies memory from Device to Host
///
/// See [cudaMemcpy() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gc263dbe6574220cc776b45438fc351e8)
///
/// # Safety
/// **This function is synchronous**
///
/// 1. `T` must be the type that device pointer was allocated with.
/// 2. The device pointer should not have been freed already (double free)
pub unsafe fn memcpy_dtoh_sync<T>(dst: &mut [T], src: *const c_void) -> Result<(), RuntimeError> {
    lib()
        .cudaMemcpy(
            dst.as_mut_ptr() as *mut c_void,
            src,
            std::mem::size_of_val(dst),
            sys::cudaMemcpyKind::cudaMemcpyDeviceToHost,
        )
        .result()
}

/// Copies memory from Device to Device with stream ordered semantics.
///
/// See [cudaMemcpyAsync() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g85073372f776b4c4d5f89f7124b7bf79)
///
/// # Safety
/// 1. `T` must be the type that BOTH device pointers were allocated with.
/// 2. Neither device pointer should have been freed already (double free)
/// 3. The stream should be the stream the memory was allocated on.
pub unsafe fn memcpy_dtod_async(
    dst: *mut c_void,
    src: *const c_void,
    num_bytes: usize,
    stream: sys::cudaStream_t,
) -> Result<(), RuntimeError> {
    lib()
        .cudaMemcpyAsync(
            dst,
            src,
            num_bytes,
            sys::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            stream,
        )
        .result()
}

/// Copies memory from Device to Device
///
/// See [cudaMemcpy() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gc263dbe6574220cc776b45438fc351e8)
///
/// # Safety
/// 1. `T` must be the type that BOTH device pointers were allocated with.
/// 2. Neither device pointer should have been freed already (double free)
pub unsafe fn memcpy_dtod_sync(
    dst: *mut c_void,
    src: *const c_void,
    num_bytes: usize,
) -> Result<(), RuntimeError> {
    lib()
        .cudaMemcpy(
            dst,
            src,
            num_bytes,
            sys::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
        )
        .result()
}

/// Returns (free, total) memory in bytes.
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0)
pub fn mem_get_info() -> Result<(usize, usize), RuntimeError> {
    let mut free = 0;
    let mut total = 0;
    unsafe {
        lib().cudaMemGetInfo(&mut free, &mut total).result()?;
    }
    Ok((free, total))
}

pub mod event {
    use super::{lib, sys, RuntimeError};
    use std::mem::MaybeUninit;

    /// Creates an event.
    ///
    /// See [cudaEventCreate() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g7b317e07ff385d85aa656204b971a042)
    pub fn create(flags: u32) -> Result<sys::cudaEvent_t, RuntimeError> {
        let mut event = MaybeUninit::uninit();
        unsafe {
            lib()
                .cudaEventCreateWithFlags(event.as_mut_ptr(), flags)
                .result()?;
            Ok(event.assume_init())
        }
    }

    /// Records an event.
    ///
    /// See [cudaEventRecord() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1gf4fcb74343aa689f4159791967868446)
    ///
    /// # Safety
    /// This function is unsafe because event can be a null event, in which case
    pub unsafe fn record(
        event: sys::cudaEvent_t,
        stream: sys::cudaStream_t,
    ) -> Result<(), RuntimeError> {
        lib().cudaEventRecord(event, stream).result()
    }

    /// Computes the elapsed time (in milliseconds) between two events.
    ///
    /// See [cudaEventElapsedTime() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g40159125411db92c835edb46a0989cd6)
    ///
    /// # Safety
    /// 1. Events must have been created by [create]
    /// 2. They should be on the same stream
    /// 3. They must not have been destroyed.
    pub unsafe fn elapsed(
        start: sys::cudaEvent_t,
        end: sys::cudaEvent_t,
    ) -> Result<f32, RuntimeError> {
        let mut ms: f32 = 0.0;
        lib()
            .cudaEventElapsedTime((&mut ms) as *mut _, start, end)
            .result()?;
        Ok(ms)
    }

    /// Destroys an event.
    ///
    /// > An event may be destroyed before it is complete (i.e., while cudaEventQuery() would return cudaErrorNotReady).
    /// > In this case, the call does not block on completion of the event,
    /// > and any associated resources will automatically be released asynchronously at completion.
    ///
    /// See [cudaEventDestroy() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g2cb6baa0830a1cd0bd957bfd8705045b)
    ///
    /// # Safety
    /// 1. Event must not have been freed already
    pub unsafe fn destroy(event: sys::cudaEvent_t) -> Result<(), RuntimeError> {
        lib().cudaEventDestroy(event).result()
    }
}

/// Launches a CUDA function
///
/// See [cudaLaunchKernel() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g5064cdf5d8e6741ace56fd8be951783c)
///
/// # Safety
/// This method is **very unsafe**.
///
/// 1. The function must be a valid function.
/// 2. The grid and block dimensions must be valid.
/// 3. The shared memory size must be correct.
/// 4. The stream must be a valid stream.
/// 5. The kernel params must be valid.
#[inline]
pub unsafe fn launch_kernel(
    f: crate::driver::sys::CUfunction,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem_bytes: usize,
    stream: sys::cudaStream_t,
    kernel_params: &mut [*mut c_void],
) -> Result<(), RuntimeError> {
    lib()
        .cudaLaunchKernel(
            f as *const c_void,
            sys::dim3 {
                x: grid_dim.0,
                y: grid_dim.1,
                z: grid_dim.2,
            },
            sys::dim3 {
                x: block_dim.0,
                y: block_dim.1,
                z: block_dim.2,
            },
            kernel_params.as_mut_ptr(),
            shared_mem_bytes,
            stream,
        )
        .result()
}

pub mod external_memory {
    use core::ffi::c_void;
    use std::mem::MaybeUninit;

    use super::{lib, sys, RuntimeError};

    /// Imports an external memory object, in this case an OpaqueFd.
    ///
    /// The memory should be destroyed using [`destroy_external_memory`].
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g52aba3a7f780157d8ba12972b2481735)
    ///
    /// # Safety
    /// `size` must be the size of the size of the memory object in bytes.
    #[cfg(unix)]
    pub unsafe fn import_external_memory_opaque_fd(
        fd: std::os::fd::RawFd,
        size: u64,
    ) -> Result<sys::cudaExternalMemory_t, RuntimeError> {
        let mut external_memory = MaybeUninit::uninit();
        let handle_description = sys::cudaExternalMemoryHandleDesc {
            type_: sys::cudaExternalMemoryHandleType::cudaExternalMemoryHandleTypeOpaqueFd,
            handle: sys::cudaExternalMemoryHandleDesc__bindgen_ty_1 { fd },
            size,
            ..Default::default()
        };
        lib()
            .cudaImportExternalMemory(external_memory.as_mut_ptr(), &handle_description)
            .result()?;
        Ok(external_memory.assume_init())
    }

    /// Imports an external memory object, in this case an OpaqueWin32 handle.
    ///
    /// The memory should be destroyed using [`destroy_external_memory`].
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g52aba3a7f780157d8ba12972b2481735)
    ///
    /// # Safety
    /// `size` must be the size of the size of the memory object in bytes.
    #[cfg(windows)]
    pub unsafe fn import_external_memory_opaque_win32(
        handle: std::os::windows::io::RawHandle,
        size: u64,
    ) -> Result<sys::cudaExternalMemory_t, RuntimeError> {
        let mut external_memory = MaybeUninit::uninit();
        let handle_description = sys::cudaExternalMemoryHandleDesc {
            type_: sys::cudaExternalMemoryHandleType::cudaExternalMemoryHandleTypeOpaqueWin32,
            handle: sys::cudaExternalMemoryHandleDesc_st__bindgen_ty_1 {
                win32: sys::cudaExternalMemoryHandleDesc_st__bindgen_ty_1__bindgen_ty_1 {
                    handle,
                    name: std::ptr::null(),
                },
            },
            size,
            ..Default::default()
        };
        sys::cudaImportExternalMemory(external_memory.as_mut_ptr(), &handle_description)
            .result()?;
        Ok(external_memory.assume_init())
    }

    /// Destroys an external memory object.
    ///
    /// See [cudaDestroyExternalMemory() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXTRES__INTEROP.html#group__CUDART__EXTRES__INTEROP_1ga48e3292855a85b1ba80fa1fdf85a158)
    ///
    /// # Safety
    /// 1. Any mapped buffers onto this object must already be freed.
    /// 2. The external memory must only be destroyed once.
    pub unsafe fn destroy_external_memory(
        external_memory: sys::cudaExternalMemory_t,
    ) -> Result<(), RuntimeError> {
        lib().cudaDestroyExternalMemory(external_memory).result()
    }

    /// Maps a buffer onto an imported memory object.
    ///
    /// The buffer must be freed using [`memory_free`](super::memory_free).
    ///
    /// See [cudaExternalMemoryGetMappedBuffer() docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXTRES__INTEROP.html#group__CUDART__EXTRES__INTEROP_1g78c1876c265217cd7874e6e8a7608e41)
    ///
    /// # Safety
    /// Mapped buffers may overlap.
    pub unsafe fn get_mapped_buffer(
        external_memory: sys::cudaExternalMemory_t,
        offset: u64,
        size: u64,
    ) -> Result<*mut c_void, RuntimeError> {
        let mut device_ptr = MaybeUninit::uninit();
        let buffer_description = sys::cudaExternalMemoryBufferDesc {
            offset,
            size,
            ..Default::default()
        };
        lib()
            .cudaExternalMemoryGetMappedBuffer(
                device_ptr.as_mut_ptr(),
                external_memory,
                &buffer_description,
            )
            .result()?;
        Ok(device_ptr.assume_init())
    }
}
