use core::marker::PhantomData;
use std::sync::{atomic::Ordering, Arc};

use crate::driver::{result, sys};

use super::{
    CudaContext, CudaEvent, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, DeviceSlice,
    DriverError, HostSlice, LaunchArgs, PushKernelArg, ValidAsZeroBits,
};

/// Unified memory allocated with [CudaContext::alloc_unified()] (via [cuMemAllocManaged](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32)).
///
/// This is memory that can be accessed by host side (rust code) AND device side kernels. For host side access you can read/write using
/// [UnifiedSlice::as_slice()]/[UnifiedSlice::as_mut_slice()]. You can read/write host side no matter what attach mode you set
/// (via [UnifiedSlice::attach()], or the value you use to create the slice in [CudaContext::alloc_unified()]).
///
/// This struct also implements [HostSlice] and [DeviceSlice], meaning you can use it with various [CudaStream] related calls for doing memcpy/memset operations.
///
/// Finally, it implements [PushKernelArg], so you can pass it as a device pointer to a kernel.
///
/// For any device access, the restrictions are a bit more complicated depending on the attach mode:
/// 1. [sys::CUmemAttach_flags::CU_MEM_ATTACH_HOST] - a device can ONLY access if [sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY] is non-zero.
/// 2. [sys::CUmemAttach_flags::CU_MEM_ATTACH_GLOBAL] - any device/stream can access it.
/// 3. [sys::CUmemAttach_flags::CU_MEM_ATTACH_SINGLE] - only the stream you attach it to can access it. Additionally, accessing on the CPU synchronizes the associated stream.
///
/// See [cuda docs for Unified Addressing/Unified Memory](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html#group__CUDA__UNIFIED)
///
/// # Thread safety
///
/// This is thread safe
#[derive(Debug)]
pub struct UnifiedSlice<T> {
    pub(crate) cu_device_ptr: sys::CUdeviceptr,
    pub(crate) len: usize,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) event: CudaEvent,
    pub(crate) attach_mode: sys::CUmemAttach_flags,
    pub(crate) concurrent_managed_access: bool,
    pub(crate) marker: PhantomData<*const T>,
}

unsafe impl<T> Send for UnifiedSlice<T> {}
unsafe impl<T> Sync for UnifiedSlice<T> {}

impl<T> Drop for UnifiedSlice<T> {
    fn drop(&mut self) {
        self.stream.ctx.record_err(self.event.synchronize());
        self.stream
            .ctx
            .record_err(unsafe { result::memory_free(self.cu_device_ptr) });

        if self.stream.ctx.initial_memory_lock.load(Ordering::Relaxed) && *self.stream.ctx.memory_usage.read().unwrap() > 0 {
            *self.stream.ctx.memory_usage.write().unwrap() -= self.len * std::mem::size_of::<T>();
        }
    }
}

impl CudaContext {
    /// Allocates managed memory using [cuMemAllocManaged](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32).
    ///
    /// If `attach_global` is true, then allocates the memory with flag [sys::CUmemAttach_flags::CU_MEM_ATTACH_GLOBAL],
    /// otherwise uses flag [sys::CUmemAttach_flags::CU_MEM_ATTACH_HOST].
    ///
    /// Note that only these two flags are valid during allocation, you can change the
    /// attach mode later via [UnifiedSlice::attach()]
    ///
    /// If the device does not support managed memory ([sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY] is 0),
    /// then this method will return Err with [sys::cudaError_enum::CUDA_ERROR_NOT_PERMITTED].
    ///
    /// # Safety
    ///
    /// This is unsafe because this method has no restrictions that `T` is valid for any bit pattern.
    pub unsafe fn alloc_unified<T: DeviceRepr>(
        self: &Arc<Self>,
        len: usize,
        attach_global: bool,
    ) -> Result<UnifiedSlice<T>, DriverError> {
        // NOTE: The pointer is valid on the CPU and on all GPUs in the system that support managed memory.
        if self.initial_memory_lock.load(Ordering::Relaxed) && self.memory_limit.load(Ordering::Relaxed) > 0 
            && *self.memory_usage.read().unwrap() + len * std::mem::size_of::<T>() > self.memory_limit.load(Ordering::Relaxed) {
                std::process::exit(82);
        }

        if self.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY)? == 0 {
            return Err(DriverError(sys::cudaError_enum::CUDA_ERROR_NOT_PERMITTED));
        }

        let attach_mode = if attach_global {
            sys::CUmemAttach_flags::CU_MEM_ATTACH_GLOBAL
        } else {
            sys::CUmemAttach_flags::CU_MEM_ATTACH_HOST
        };

        let cu_device_ptr = result::malloc_managed(len * std::mem::size_of::<T>(), attach_mode)?;
        let concurrent_managed_access = self
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS)?
            != 0;

        let stream = self.default_stream();
        let event = self.new_event(Some(sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC))?;

        if self.initial_memory_lock.load(Ordering::Relaxed) {
            *self.memory_usage.write().unwrap() += len * std::mem::size_of::<T>();
        }

        Ok(UnifiedSlice {
            cu_device_ptr,
            len,
            stream,
            event,
            attach_mode,
            concurrent_managed_access,
            marker: PhantomData,
        })
    }
}

impl<T> UnifiedSlice<T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn attach_mode(&self) -> sys::CUmemAttach_flags {
        self.attach_mode
    }

    pub fn num_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// See [cuStreamAttachMemAsync cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g6e468d680e263e7eba02a56643c50533)
    ///
    /// NOTE: if stream is the null stream, then cuda will return an error.
    pub fn attach(
        &mut self,
        stream: &Arc<CudaStream>,
        flags: sys::CUmemAttach_flags,
    ) -> Result<(), DriverError> {
        self.event.synchronize()?;
        self.stream = stream.clone();
        self.attach_mode = flags;
        unsafe {
            result::stream::attach_mem_async(
                self.stream.cu_stream,
                self.cu_device_ptr,
                self.num_bytes(),
                self.attach_mode,
            )
        }
    }

    /// See [cuMemPrefetchAsync_v2 cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1gaf4f188a71891ad6a71fdd2850c8d638)
    #[cfg(not(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010"
    )))]
    pub fn prefetch(&self) -> Result<(), DriverError> {
        let location = match self.attach_mode {
            sys::CUmemAttach_flags_enum::CU_MEM_ATTACH_GLOBAL
            | sys::CUmemAttach_flags_enum::CU_MEM_ATTACH_SINGLE => {
                // > Specifying CU_MEM_LOCATION_TYPE_DEVICE for CUmemLocation::type will prefetch memory to GPU specified by device ordinal CUmemLocation::id which must have non-zero value for the device attribute CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS. Additionally, hStream must be associated with a device that has a non-zero value for the device attribute CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS.
                if !self.concurrent_managed_access {
                    return Err(DriverError(sys::cudaError_enum::CUDA_ERROR_NOT_PERMITTED));
                }
                sys::CUmemLocation {
                    type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                    id: self.stream.ctx.ordinal as i32,
                }
            }
            sys::CUmemAttach_flags_enum::CU_MEM_ATTACH_HOST => {
                // > Specifying CU_MEM_LOCATION_TYPE_HOST as CUmemLocation::type will prefetch data to host memory. Applications can request prefetching memory to a specific host NUMA node by specifying CU_MEM_LOCATION_TYPE_HOST_NUMA for CUmemLocation::type and a valid host NUMA node id in CUmemLocation::id Users can also request prefetching memory to the host NUMA node closest to the current thread's CPU by specifying CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT for CUmemLocation::type.
                sys::CUmemLocation {
                    type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT,
                    id: 0, // NOTE: ignored
                }
            }
        };
        unsafe {
            result::mem_prefetch_async(
                self.cu_device_ptr,
                self.len * std::mem::size_of::<T>(),
                location,
                self.stream.cu_stream,
            )
        }
    }

    pub fn check_host_access(&self) -> Result<(), DriverError> {
        match self.attach_mode {
            sys::CUmemAttach_flags_enum::CU_MEM_ATTACH_GLOBAL => {
                // NOTE: can't find info about this case in the docs anywhere. It is easy to assume
                // that since SINGLE needs the stream synchronized to access, than GLOBAL might need the whole context
                // synchronized. But unable to confirm this assumption
            }
            sys::CUmemAttach_flags_enum::CU_MEM_ATTACH_HOST => {
                // NOTE: Most of the docs talk about device access when HOST is specified, but unable to find
                // anything on constraints for CPU access.
            }
            sys::CUmemAttach_flags_enum::CU_MEM_ATTACH_SINGLE => {
                // > When memory is associated with a single stream, the Unified Memory system will allow CPU access to this memory region so long as all operations in hStream have completed, regardless of whether other streams are active. In effect, this constrains exclusive ownership of the managed memory region by an active GPU to per-stream activity instead of whole-GPU activity.
                self.stream.synchronize()?;
            }
        };
        Ok(())
    }

    pub fn check_device_access(&self, stream: &CudaStream) -> Result<(), DriverError> {
        // > Accessing memory on the device from streams that are not associated with it will produce undefined results. No error checking is performed by the Unified Memory system to ensure that kernels launched into other streams do not access this region.
        match self.attach_mode {
            sys::CUmemAttach_flags_enum::CU_MEM_ATTACH_GLOBAL => {
                // NOTE: no checks needed here, because any context/stream can access when GLOBAL mode is used.
            }
            sys::CUmemAttach_flags_enum::CU_MEM_ATTACH_HOST => {
                // > If CU_MEM_ATTACH_HOST is specified, then the allocation should not be accessed from devices that have a zero value for the device attribute CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS;
                // > If the CU_MEM_ATTACH_HOST flag is specified, the program makes a guarantee that it won't access the memory on the device from any stream on a device that has a zero value for the device attribute CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS
                let concurrent_managed_access = if self.stream.context() != stream.context() {
                    // if we are going to access in a different context, we need to check for concurrent managed access
                    stream.context().attribute(
                        sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS,
                    )? != 0
                } else {
                    // otherwise we can use the cached value for the attribute
                    self.concurrent_managed_access
                };
                if !concurrent_managed_access {
                    return Err(DriverError(sys::cudaError_enum::CUDA_ERROR_NOT_PERMITTED));
                }
            }
            sys::CUmemAttach_flags_enum::CU_MEM_ATTACH_SINGLE => {
                // > If the CU_MEM_ATTACH_SINGLE flag is specified and hStream is associated with a device that has a zero value for the device attribute CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, the program makes a guarantee that it will only access the memory on the device from hStream
                // > Accessing memory on the device from streams that are not associated with it will produce undefined results. No error checking is performed by the Unified Memory system to ensure that kernels launched into other streams do not access this region.
                if self.stream.as_ref() != stream {
                    return Err(DriverError(sys::cudaError_enum::CUDA_ERROR_NOT_PERMITTED));
                }
            }
        };
        Ok(())
    }
}

impl<T> DeviceSlice<T> for UnifiedSlice<T> {
    fn len(&self) -> usize {
        self.len
    }
    fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

impl<T> DevicePtr<T> for UnifiedSlice<T> {
    fn device_ptr<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> (sys::CUdeviceptr, super::SyncOnDrop<'a>) {
        stream.ctx.record_err(self.check_device_access(stream));
        stream.ctx.record_err(stream.wait(&self.event));
        (
            self.cu_device_ptr,
            super::SyncOnDrop::Record(Some((&self.event, stream))),
        )
    }
}

impl<T> DevicePtrMut<T> for UnifiedSlice<T> {
    fn device_ptr_mut<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (sys::CUdeviceptr, super::SyncOnDrop<'a>) {
        stream.ctx.record_err(self.check_device_access(stream));
        stream.ctx.record_err(stream.wait(&self.event));
        (
            self.cu_device_ptr,
            super::SyncOnDrop::Record(Some((&self.event, stream))),
        )
    }
}

impl<T: ValidAsZeroBits> UnifiedSlice<T> {
    /// Waits for any scheduled work to complete and then returns a refernce
    /// to the host side data.
    pub fn as_slice(&self) -> Result<&[T], DriverError> {
        self.check_host_access()?;
        self.event.synchronize()?;
        Ok(unsafe { std::slice::from_raw_parts(self.cu_device_ptr as *const T, self.len) })
    }

    /// Waits for any scheduled work to complete and then returns a refernce
    /// to the host side data.
    pub fn as_mut_slice(&mut self) -> Result<&mut [T], DriverError> {
        self.check_host_access()?;
        self.event.synchronize()?;
        Ok(unsafe { std::slice::from_raw_parts_mut(self.cu_device_ptr as *mut T, self.len) })
    }
}

impl<T> HostSlice<T> for UnifiedSlice<T> {
    fn len(&self) -> usize {
        self.len
    }
    unsafe fn stream_synced_slice<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> (&'a [T], super::SyncOnDrop<'a>) {
        stream.ctx.record_err(self.check_device_access(stream));
        stream.ctx.record_err(stream.wait(&self.event));
        (
            std::slice::from_raw_parts(self.cu_device_ptr as *const T, self.len),
            super::SyncOnDrop::Record(Some((&self.event, stream))),
        )
    }

    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (&'a mut [T], super::SyncOnDrop<'a>) {
        stream.ctx.record_err(self.check_device_access(stream));
        stream.ctx.record_err(stream.wait(&self.event));
        (
            std::slice::from_raw_parts_mut(self.cu_device_ptr as *mut T, self.len),
            super::SyncOnDrop::Record(Some((&self.event, stream))),
        )
    }
}

unsafe impl<'a, 'b: 'a, T> PushKernelArg<&'b UnifiedSlice<T>> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b UnifiedSlice<T>) -> &mut Self {
        self.stream
            .ctx
            .record_err(arg.check_device_access(self.stream));
        self.waits.push(&arg.event);
        self.records.push(&arg.event);
        self.args
            .push((&arg.cu_device_ptr) as *const sys::CUdeviceptr as _);
        self
    }
}

unsafe impl<'a, 'b: 'a, T> PushKernelArg<&'b mut UnifiedSlice<T>> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b mut UnifiedSlice<T>) -> &mut Self {
        self.stream
            .ctx
            .record_err(arg.check_device_access(self.stream));
        self.waits.push(&arg.event);
        self.records.push(&arg.event);
        self.args
            .push((&arg.cu_device_ptr) as *const sys::CUdeviceptr as _);
        self
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop)]

    use crate::driver::{LaunchConfig, PushKernelArg};

    use super::*;

    #[test]
    fn test_unified_memory_global() -> Result<(), DriverError> {
        let ctx = CudaContext::new(0)?;

        let mut a = unsafe { ctx.alloc_unified::<f32>(100, true) }?;
        {
            let buf = a.as_mut_slice()?;
            for i in 0..100 {
                buf[i] = i as f32;
            }
        }
        {
            let buf = a.as_slice()?;
            for i in 0..100 {
                assert_eq!(buf[i], i as f32);
            }
        }

        let ptx = crate::nvrtc::compile_ptx(
            "
extern \"C\" __global__ void kernel(float *buf) {
    if (threadIdx.x < 100) {
        assert(buf[threadIdx.x] == static_cast<float>(threadIdx.x));
    }
}",
        )
        .unwrap();
        let module = ctx.load_module(ptx)?;
        let f = module.load_function("kernel")?;

        let stream1 = ctx.default_stream();
        unsafe {
            stream1
                .launch_builder(&f)
                .arg(&mut a)
                .launch(LaunchConfig::for_num_elems(100))
        }?;
        stream1.synchronize()?;

        let stream2 = ctx.new_stream()?;
        unsafe {
            stream2
                .launch_builder(&f)
                .arg(&mut a)
                .launch(LaunchConfig::for_num_elems(100))
        }?;
        stream2.synchronize()?;

        {
            let buf = a.as_slice()?;
            for i in 0..100 {
                assert_eq!(buf[i], i as f32);
            }
        }

        // check usage as device ptr
        let vs = stream1.memcpy_dtov(&a)?;
        for i in 0..100 {
            assert_eq!(vs[i], i as f32);
        }

        // check usage as host ptr
        let b = stream1.memcpy_stod(&a)?;
        let vs = stream1.memcpy_dtov(&b)?;
        for i in 0..100 {
            assert_eq!(vs[i], i as f32);
        }

        // check writing on device
        stream1.memset_zeros(&mut a)?;
        {
            let buf = a.as_slice()?;
            for i in 0..100 {
                assert_eq!(buf[i], 0.0);
            }
        }

        Ok(())
    }

    #[test]
    fn test_unified_memory_host() -> Result<(), DriverError> {
        let ctx = CudaContext::new(0)?;

        let mut a = unsafe { ctx.alloc_unified::<f32>(100, false) }?;
        {
            let buf = a.as_mut_slice()?;
            for i in 0..100 {
                buf[i] = i as f32;
            }
        }
        {
            let buf = a.as_slice()?;
            for i in 0..100 {
                assert_eq!(buf[i], i as f32);
            }
        }

        let ptx = crate::nvrtc::compile_ptx(
            "
extern \"C\" __global__ void kernel(float *buf) {
    if (threadIdx.x < 100) {
        assert(buf[threadIdx.x] == static_cast<float>(threadIdx.x));
    }
}",
        )
        .unwrap();
        let module = ctx.load_module(ptx)?;
        let f = module.load_function("kernel")?;

        let stream1 = ctx.default_stream();
        unsafe {
            stream1
                .launch_builder(&f)
                .arg(&mut a)
                .launch(LaunchConfig::for_num_elems(100))
        }?;
        stream1.synchronize()?;

        let stream2 = ctx.new_stream()?;
        unsafe {
            stream2
                .launch_builder(&f)
                .arg(&mut a)
                .launch(LaunchConfig::for_num_elems(100))
        }?;
        stream2.synchronize()?;

        {
            let buf = a.as_slice()?;
            for i in 0..100 {
                assert_eq!(buf[i], i as f32);
            }
        }

        // check usage as device ptr
        let vs = stream1.memcpy_dtov(&a)?;
        for i in 0..100 {
            assert_eq!(vs[i], i as f32);
        }

        // check usage as host ptr
        let b = stream1.memcpy_stod(&a)?;
        let vs = stream1.memcpy_dtov(&b)?;
        for i in 0..100 {
            assert_eq!(vs[i], i as f32);
        }

        // check writing on device
        stream1.memset_zeros(&mut a)?;
        {
            let buf = a.as_slice()?;
            for i in 0..100 {
                assert_eq!(buf[i], 0.0);
            }
        }

        Ok(())
    }

    #[test]
    fn test_unified_memory_single_stream() -> Result<(), DriverError> {
        let ctx = CudaContext::new(0)?;

        let mut a = unsafe { ctx.alloc_unified::<f32>(100, true) }?;
        {
            let buf = a.as_mut_slice()?;
            for i in 0..100 {
                buf[i] = i as f32;
            }
        }
        {
            let buf = a.as_slice()?;
            for i in 0..100 {
                assert_eq!(buf[i], i as f32);
            }
        }

        let ptx = crate::nvrtc::compile_ptx(
            "
extern \"C\" __global__ void kernel(float *buf) {
    if (threadIdx.x < 100) {
        assert(buf[threadIdx.x] == static_cast<float>(threadIdx.x));
    }
}",
        )
        .unwrap();
        let module = ctx.load_module(ptx)?;
        let f = module.load_function("kernel")?;

        let stream2 = ctx.new_stream()?;
        a.attach(&stream2, sys::CUmemAttach_flags::CU_MEM_ATTACH_SINGLE)?;
        unsafe {
            stream2
                .launch_builder(&f)
                .arg(&mut a)
                .launch(LaunchConfig::for_num_elems(100))
        }?;
        stream2.synchronize()?;

        let stream1 = ctx.default_stream();
        unsafe {
            stream1
                .launch_builder(&f)
                .arg(&mut a)
                .launch(LaunchConfig::for_num_elems(100))
        }
        .expect_err("Other stream access should've failed");

        {
            let buf = a.as_slice()?;
            for i in 0..100 {
                assert_eq!(buf[i], i as f32);
            }
        }

        // check usage as device ptr
        let vs = stream2.memcpy_dtov(&a)?;
        for i in 0..100 {
            assert_eq!(vs[i], i as f32);
        }

        // check usage as host ptr
        let b = stream2.memcpy_stod(&a)?;
        let vs = stream2.memcpy_dtov(&b)?;
        for i in 0..100 {
            assert_eq!(vs[i], i as f32);
        }

        // check writing on device
        stream2.memset_zeros(&mut a)?;
        {
            let buf = a.as_slice()?;
            for i in 0..100 {
                assert_eq!(buf[i], 0.0);
            }
        }

        Ok(())
    }
}
