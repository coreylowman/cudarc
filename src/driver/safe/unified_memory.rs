use core::marker::PhantomData;
use std::ops::RangeBounds;
use std::sync::Arc;

use crate::driver::{result, sys};

use super::{
    core::to_range, CudaContext, CudaEvent, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr,
    DeviceSlice, DriverError, HostSlice, LaunchArgs, PushKernelArg, ValidAsZeroBits,
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
    }
}

/// `&[T]` on unified memory. An immutable sub-view into a [UnifiedSlice] created by [UnifiedSlice::as_view()]/[UnifiedSlice::slice()].
#[derive(Debug, Copy, Clone)]
pub struct UnifiedView<'a, T> {
    pub(crate) ptr: sys::CUdeviceptr,
    pub(crate) len: usize,
    pub(crate) event: &'a CudaEvent,
    pub(crate) stream: &'a Arc<CudaStream>,
    pub(crate) attach_mode: sys::CUmemAttach_flags,
    pub(crate) concurrent_managed_access: bool,
    marker: PhantomData<&'a [T]>,
}

/// `&mut [T]` on unified memory. A mutable sub-view into a [UnifiedSlice] created by [UnifiedSlice::as_view_mut()]/[UnifiedSlice::slice_mut()].
#[derive(Debug)]
pub struct UnifiedViewMut<'a, T> {
    pub(crate) ptr: sys::CUdeviceptr,
    pub(crate) len: usize,
    pub(crate) event: &'a CudaEvent,
    pub(crate) stream: &'a Arc<CudaStream>,
    pub(crate) attach_mode: sys::CUmemAttach_flags,
    pub(crate) concurrent_managed_access: bool,
    marker: PhantomData<&'a mut [T]>,
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
        check_device_access(
            self.attach_mode,
            &self.stream,
            self.concurrent_managed_access,
            stream,
        )
    }
}

// Consolidated device access validation function
fn check_device_access(
    attach_mode: sys::CUmemAttach_flags,
    owner_stream: &Arc<CudaStream>,
    concurrent_managed_access: bool,
    stream: &CudaStream,
) -> Result<(), DriverError> {
    match attach_mode {
        sys::CUmemAttach_flags_enum::CU_MEM_ATTACH_GLOBAL => {
            // NOTE: no checks needed here, because any context/stream can access when GLOBAL mode is used.
        }
        sys::CUmemAttach_flags_enum::CU_MEM_ATTACH_HOST => {
            let concurrent_managed_access = if owner_stream.context() != stream.context() {
                stream.context().attribute(
                    sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS,
                )? != 0
            } else {
                concurrent_managed_access
            };
            if !concurrent_managed_access {
                return Err(DriverError(sys::cudaError_enum::CUDA_ERROR_NOT_PERMITTED));
            }
        }
        sys::CUmemAttach_flags_enum::CU_MEM_ATTACH_SINGLE => {
            if owner_stream.as_ref() != stream {
                return Err(DriverError(sys::cudaError_enum::CUDA_ERROR_NOT_PERMITTED));
            }
        }
    };
    Ok(())
}

impl<T> UnifiedSlice<T> {
    /// Creates an immutable view of the entire [UnifiedSlice].
    pub fn as_view(&self) -> UnifiedView<'_, T> {
        UnifiedView {
            ptr: self.cu_device_ptr,
            len: self.len,
            event: &self.event,
            stream: &self.stream,
            attach_mode: self.attach_mode,
            concurrent_managed_access: self.concurrent_managed_access,
            marker: PhantomData,
        }
    }

    /// Creates a mutable view of the entire [UnifiedSlice].
    pub fn as_view_mut(&mut self) -> UnifiedViewMut<'_, T> {
        UnifiedViewMut {
            ptr: self.cu_device_ptr,
            len: self.len,
            event: &self.event,
            stream: &self.stream,
            attach_mode: self.attach_mode,
            concurrent_managed_access: self.concurrent_managed_access,
            marker: PhantomData,
        }
    }

    /// Creates a [UnifiedView] at the specified offset from the start of `self`.
    ///
    /// Panics if `range.start >= self.len`.
    pub fn slice(&self, bounds: impl RangeBounds<usize>) -> UnifiedView<'_, T> {
        self.as_view().slice(bounds)
    }

    /// Fallible version of [UnifiedSlice::slice()].
    pub fn try_slice(&self, bounds: impl RangeBounds<usize>) -> Option<UnifiedView<'_, T>> {
        self.as_view().try_slice(bounds)
    }

    /// Creates a [UnifiedViewMut] at the specified offset from the start of `self`.
    ///
    /// Panics if `range` and `0...self.len()` are not overlapping.
    pub fn slice_mut(&mut self, bounds: impl RangeBounds<usize>) -> UnifiedViewMut<'_, T> {
        self.try_slice_mut(bounds).unwrap()
    }

    /// Fallible version of [UnifiedSlice::slice_mut]
    pub fn try_slice_mut(
        &mut self,
        bounds: impl RangeBounds<usize>,
    ) -> Option<UnifiedViewMut<'_, T>> {
        to_range(bounds, self.len).map(|(start, end)| self.as_view_mut().resize(start, end))
    }

    pub fn split_at(&self, mid: usize) -> (UnifiedView<'_, T>, UnifiedView<'_, T>) {
        self.try_split_at(mid).unwrap()
    }

    /// Fallible version of [UnifiedSlice::split_at]. Returns `None` if `mid > self.len`.
    pub fn try_split_at(&self, mid: usize) -> Option<(UnifiedView<'_, T>, UnifiedView<'_, T>)> {
        (mid <= self.len()).then(|| {
            let view = self.as_view();
            (view.resize(0, mid), view.resize(mid, self.len))
        })
    }

    /// Splits the [UnifiedSlice] into two at the given index, returning two [UnifiedViewMut] for the two halves.
    ///
    /// Panics if `mid > self.len`.
    pub fn split_at_mut(&mut self, mid: usize) -> (UnifiedViewMut<'_, T>, UnifiedViewMut<'_, T>) {
        self.try_split_at_mut(mid).unwrap()
    }

    /// Fallible version of [UnifiedSlice::split_at_mut].
    ///
    /// Returns `None` if `mid > self.len`.
    pub fn try_split_at_mut(
        &mut self,
        mid: usize,
    ) -> Option<(UnifiedViewMut<'_, T>, UnifiedViewMut<'_, T>)> {
        let length = self.len;
        (mid <= length).then(|| {
            let view = self.as_view_mut();
            (view.resize(0, mid), view.resize(mid, length))
        })
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

impl<T> DeviceSlice<T> for UnifiedView<'_, T> {
    fn len(&self) -> usize {
        self.len
    }
    fn stream(&self) -> &Arc<CudaStream> {
        self.stream
    }
}

impl<T> DeviceSlice<T> for UnifiedViewMut<'_, T> {
    fn len(&self) -> usize {
        self.len
    }
    fn stream(&self) -> &Arc<CudaStream> {
        self.stream
    }
}

impl<T> DevicePtr<T> for UnifiedView<'_, T> {
    fn device_ptr<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> (sys::CUdeviceptr, super::SyncOnDrop<'a>) {
        stream.ctx.record_err(check_device_access(
            self.attach_mode,
            self.stream,
            self.concurrent_managed_access,
            stream,
        ));
        stream.ctx.record_err(stream.wait(self.event));
        (
            self.ptr,
            super::SyncOnDrop::Record(Some((self.event, stream))),
        )
    }
}

impl<T> DevicePtr<T> for UnifiedViewMut<'_, T> {
    fn device_ptr<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> (sys::CUdeviceptr, super::SyncOnDrop<'a>) {
        stream.ctx.record_err(check_device_access(
            self.attach_mode,
            self.stream,
            self.concurrent_managed_access,
            stream,
        ));
        stream.ctx.record_err(stream.wait(self.event));
        (
            self.ptr,
            super::SyncOnDrop::Record(Some((self.event, stream))),
        )
    }
}

impl<T> DevicePtrMut<T> for UnifiedViewMut<'_, T> {
    fn device_ptr_mut<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (sys::CUdeviceptr, super::SyncOnDrop<'a>) {
        stream.ctx.record_err(check_device_access(
            self.attach_mode,
            self.stream,
            self.concurrent_managed_access,
            stream,
        ));
        stream.ctx.record_err(stream.wait(self.event));
        (
            self.ptr,
            super::SyncOnDrop::Record(Some((self.event, stream))),
        )
    }
}

impl<T> HostSlice<T> for UnifiedView<'_, T> {
    fn len(&self) -> usize {
        self.len
    }
    unsafe fn stream_synced_slice<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> (&'a [T], super::SyncOnDrop<'a>) {
        stream.ctx.record_err(check_device_access(
            self.attach_mode,
            self.stream,
            self.concurrent_managed_access,
            stream,
        ));
        stream.ctx.record_err(stream.wait(self.event));
        (
            std::slice::from_raw_parts(self.ptr as *const T, self.len),
            super::SyncOnDrop::Record(Some((self.event, stream))),
        )
    }

    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (&'a mut [T], super::SyncOnDrop<'a>) {
        stream.ctx.record_err(check_device_access(
            self.attach_mode,
            self.stream,
            self.concurrent_managed_access,
            stream,
        ));
        stream.ctx.record_err(stream.wait(self.event));
        (
            std::slice::from_raw_parts_mut(self.ptr as *mut T, self.len),
            super::SyncOnDrop::Record(Some((self.event, stream))),
        )
    }
}

impl<T> HostSlice<T> for UnifiedViewMut<'_, T> {
    fn len(&self) -> usize {
        self.len
    }
    unsafe fn stream_synced_slice<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> (&'a [T], super::SyncOnDrop<'a>) {
        stream.ctx.record_err(check_device_access(
            self.attach_mode,
            self.stream,
            self.concurrent_managed_access,
            stream,
        ));
        stream.ctx.record_err(stream.wait(self.event));
        (
            std::slice::from_raw_parts(self.ptr as *const T, self.len),
            super::SyncOnDrop::Record(Some((self.event, stream))),
        )
    }

    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (&'a mut [T], super::SyncOnDrop<'a>) {
        stream.ctx.record_err(check_device_access(
            self.attach_mode,
            self.stream,
            self.concurrent_managed_access,
            stream,
        ));
        stream.ctx.record_err(stream.wait(self.event));
        (
            std::slice::from_raw_parts_mut(self.ptr as *mut T, self.len),
            super::SyncOnDrop::Record(Some((self.event, stream))),
        )
    }
}

unsafe impl<'a, 'b: 'a, 'c: 'b, T> PushKernelArg<&'b UnifiedView<'c, T>> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b UnifiedView<'c, T>) -> &mut Self {
        self.stream.ctx.record_err(check_device_access(
            arg.attach_mode,
            arg.stream,
            arg.concurrent_managed_access,
            self.stream,
        ));
        self.waits.push(arg.event);
        self.records.push(arg.event);
        self.args.push((&arg.ptr) as *const sys::CUdeviceptr as _);
        self
    }
}

unsafe impl<'a, 'b: 'a, 'c: 'b, T> PushKernelArg<&'b UnifiedViewMut<'c, T>> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b UnifiedViewMut<'c, T>) -> &mut Self {
        self.stream.ctx.record_err(check_device_access(
            arg.attach_mode,
            arg.stream,
            arg.concurrent_managed_access,
            self.stream,
        ));
        self.waits.push(arg.event);
        self.records.push(arg.event);
        self.args.push((&arg.ptr) as *const sys::CUdeviceptr as _);
        self
    }
}

unsafe impl<'a, 'b: 'a, 'c: 'b, T> PushKernelArg<&'b mut UnifiedViewMut<'c, T>> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b mut UnifiedViewMut<'c, T>) -> &mut Self {
        self.stream.ctx.record_err(check_device_access(
            arg.attach_mode,
            arg.stream,
            arg.concurrent_managed_access,
            self.stream,
        ));
        self.waits.push(arg.event);
        self.records.push(arg.event);
        self.args.push((&arg.ptr) as *const sys::CUdeviceptr as _);
        self
    }
}

impl<'a, T> UnifiedView<'a, T> {
    /// The number of elements `T` in this view.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn resize(&self, start: usize, end: usize) -> Self {
        assert!(start <= end && end <= self.len);
        Self {
            ptr: self.ptr + (start * std::mem::size_of::<T>()) as u64,
            len: end - start,
            event: self.event,
            stream: self.stream,
            attach_mode: self.attach_mode,
            concurrent_managed_access: self.concurrent_managed_access,
            marker: PhantomData,
        }
    }

    /// Creates a [UnifiedView] at the specified offset from the start of `self`.
    ///
    /// Panics if `range.start >= self.len`.
    pub fn slice(&self, bounds: impl RangeBounds<usize>) -> Self {
        self.try_slice(bounds).unwrap()
    }

    /// Fallible version of [UnifiedView::slice]
    pub fn try_slice(&self, bounds: impl RangeBounds<usize>) -> Option<Self> {
        to_range(bounds, self.len).map(|(start, end)| self.resize(start, end))
    }

    pub fn split_at(&self, mid: usize) -> (Self, Self) {
        self.try_split_at(mid).unwrap()
    }

    /// Fallible version of [UnifiedView::split_at].
    ///
    /// Returns `None` if `mid > self.len`.
    pub fn try_split_at(&self, mid: usize) -> Option<(Self, Self)> {
        (mid <= self.len()).then(|| (self.resize(0, mid), self.resize(mid, self.len)))
    }
}

impl<'a, T> UnifiedViewMut<'a, T> {
    /// Number of elements `T` that are in this view.
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Downgrade this to a `&[T]`
    pub fn as_view(&self) -> UnifiedView<'a, T> {
        UnifiedView {
            ptr: self.ptr,
            len: self.len,
            event: self.event,
            stream: self.stream,
            attach_mode: self.attach_mode,
            concurrent_managed_access: self.concurrent_managed_access,
            marker: PhantomData,
        }
    }

    fn resize(&self, start: usize, end: usize) -> Self {
        Self {
            ptr: self.ptr + (start * std::mem::size_of::<T>()) as u64,
            len: end - start,
            event: self.event,
            stream: self.stream,
            attach_mode: self.attach_mode,
            concurrent_managed_access: self.concurrent_managed_access,
            marker: PhantomData,
        }
    }

    /// Creates a [UnifiedView] at the specified offset from the start of `self`.
    ///
    /// Panics if `range` and `0...self.len()` are not overlapping.
    pub fn slice<'b: 'a>(&'b self, bounds: impl RangeBounds<usize>) -> UnifiedView<'a, T> {
        self.try_slice(bounds).unwrap()
    }

    /// Fallible version of [UnifiedViewMut::slice]
    pub fn try_slice<'b: 'a>(
        &'b self,
        bounds: impl RangeBounds<usize>,
    ) -> Option<UnifiedView<'a, T>> {
        to_range(bounds, self.len).map(|(start, end)| self.as_view().resize(start, end))
    }

    /// Creates a [UnifiedViewMut] at the specified offset from the start of `self`.
    ///
    /// Panics if `range` and `0...self.len()` are not overlapping.
    pub fn slice_mut(&mut self, bounds: impl RangeBounds<usize>) -> Self {
        self.try_slice_mut(bounds).unwrap()
    }

    /// Fallible version of [UnifiedViewMut::slice_mut]
    pub fn try_slice_mut(&mut self, bounds: impl RangeBounds<usize>) -> Option<Self> {
        to_range(bounds, self.len).map(|(start, end)| self.resize(start, end))
    }

    /// Splits the [UnifiedViewMut] into two at the given index.
    ///
    /// Panics if `mid > self.len`.
    pub fn split_at_mut(&mut self, mid: usize) -> (Self, Self) {
        self.try_split_at_mut(mid).unwrap()
    }

    /// Fallible version of [UnifiedViewMut::split_at_mut].
    ///
    /// Returns `None` if `mid > self.len`
    pub fn try_split_at_mut(&mut self, mid: usize) -> Option<(Self, Self)> {
        (mid <= self.len()).then(|| (self.resize(0, mid), self.resize(mid, self.len)))
    }
}

#[cfg(feature = "nvrtc")]
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
        let vs = stream1.clone_dtoh(&a)?;
        for i in 0..100 {
            assert_eq!(vs[i], i as f32);
        }

        // check usage as host ptr
        let b = stream1.clone_htod(&a)?;
        let vs = stream1.clone_dtoh(&b)?;
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
        let vs = stream1.clone_dtoh(&a)?;
        for i in 0..100 {
            assert_eq!(vs[i], i as f32);
        }

        // check usage as host ptr
        let b = stream1.clone_htod(&a)?;
        let vs = stream1.clone_dtoh(&b)?;
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
        let vs = stream2.clone_dtoh(&a)?;
        for i in 0..100 {
            assert_eq!(vs[i], i as f32);
        }

        // check usage as host ptr
        let b = stream2.clone_htod(&a)?;
        let vs = stream2.clone_dtoh(&b)?;
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

    #[test]
    fn test_unified_slice_copy_to_views() -> Result<(), DriverError> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();

        // Create multiple small unified slices with known data
        let mut smalls = [
            unsafe { ctx.alloc_unified::<f32>(2, true) }?,
            unsafe { ctx.alloc_unified::<f32>(2, true) }?,
            unsafe { ctx.alloc_unified::<f32>(2, true) }?,
            unsafe { ctx.alloc_unified::<f32>(2, true) }?,
            unsafe { ctx.alloc_unified::<f32>(2, true) }?,
        ];

        // Initialize the small slices with data
        {
            let buf = smalls[0].as_mut_slice()?;
            buf[0] = -1.0;
            buf[1] = -0.8;
        }
        {
            let buf = smalls[1].as_mut_slice()?;
            buf[0] = -0.6;
            buf[1] = -0.4;
        }
        {
            let buf = smalls[2].as_mut_slice()?;
            buf[0] = -0.2;
            buf[1] = 0.0;
        }
        {
            let buf = smalls[3].as_mut_slice()?;
            buf[0] = 0.2;
            buf[1] = 0.4;
        }
        {
            let buf = smalls[4].as_mut_slice()?;
            buf[0] = 0.6;
            buf[1] = 0.8;
        }

        // Create a large unified slice (zeroed)
        let mut big = unsafe { ctx.alloc_unified::<f32>(10, true) }?;
        stream.memset_zeros(&mut big)?;

        // Use slice_mut to get sub-views and copy data into them
        let mut offset = 0;
        for small in smalls.iter() {
            let mut sub = big.slice_mut(offset..offset + small.len());
            stream.memcpy_dtod(small, &mut sub)?;
            offset += small.len();
        }

        // Verify the result
        stream.synchronize()?;
        let result = stream.clone_dtoh(&big)?;
        assert_eq!(
            result,
            [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
        );

        Ok(())
    }

    #[test]
    fn test_unified_slice_split_at() -> Result<(), DriverError> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();

        // Create a unified slice with data [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        let mut unified = unsafe { ctx.alloc_unified::<f32>(10, true) }?;
        {
            let buf = unified.as_mut_slice()?;
            for i in 0..10 {
                buf[i] = i as f32;
            }
        }

        // Test split_at with immutable views
        let (left, right) = unified.split_at(5);
        assert_eq!(left.len(), 5);
        assert_eq!(right.len(), 5);

        let left_data = stream.clone_dtoh(&left)?;
        let right_data = stream.clone_dtoh(&right)?;
        assert_eq!(left_data, [0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(right_data, [5.0, 6.0, 7.0, 8.0, 9.0]);

        // Test split_at_mut with mutable views
        let (mut left_mut, right_mut) = unified.split_at_mut(5);
        assert_eq!(left_mut.len(), 5);
        assert_eq!(right_mut.len(), 5);

        // Modify the left half
        let zeros = std::vec![0.0f32; 5];
        stream.memcpy_htod(&zeros, &mut left_mut)?;

        // Verify only left half was modified
        stream.synchronize()?;
        let result = stream.clone_dtoh(&unified)?;
        assert_eq!(result, [0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        Ok(())
    }

    #[test]
    fn test_unified_slice_views_respect_stream_attachment() -> Result<(), DriverError> {
        let ctx = CudaContext::new(0)?;

        // Create unified slice in GLOBAL mode
        let mut unified = unsafe { ctx.alloc_unified::<f32>(100, true) }?;
        {
            let buf = unified.as_mut_slice()?;
            for i in 0..100 {
                buf[i] = i as f32;
            }
        }

        let stream1 = ctx.default_stream();
        let stream2 = ctx.new_stream()?;

        // Initially in GLOBAL mode - both streams should work
        let view1 = unified.slice(0..50);
        let data1 = stream1.clone_dtoh(&view1)?;
        assert_eq!(data1[0], 0.0);
        assert_eq!(data1[49], 49.0);

        let view2 = unified.slice(50..100);
        let data2 = stream2.clone_dtoh(&view2)?;
        assert_eq!(data2[0], 50.0);
        assert_eq!(data2[49], 99.0);

        // Test writing through a view in GLOBAL mode
        let mut view_mut = unified.slice_mut(10..20);
        let write_data = std::vec![999.0f32; 10];
        stream1.memcpy_htod(&write_data, &mut view_mut)?;
        stream1.synchronize()?;

        // Verify the write succeeded
        let verify_data = stream1.clone_dtoh(&unified)?;
        for i in 0..10 {
            assert_eq!(
                verify_data[i], i as f32,
                "Data before write range should be unchanged"
            );
        }
        for i in 10..20 {
            assert_eq!(verify_data[i], 999.0, "Data in write range should be 999.0");
        }
        for i in 20..100 {
            assert_eq!(
                verify_data[i], i as f32,
                "Data after write range should be unchanged"
            );
        }

        // Switch to SINGLE mode attached to stream2
        unified.attach(&stream2, sys::CUmemAttach_flags::CU_MEM_ATTACH_SINGLE)?;

        // Create a view - it should inherit SINGLE mode
        let view_single = unified.slice(0..50);

        // Access with attached stream (stream2) should work
        let data_ok = stream2.clone_dtoh(&view_single)?;
        assert_eq!(data_ok[0], 0.0);

        // Access with different stream (stream1) should record an error
        let _ = stream1.clone_dtoh(&view_single);
        // The error is recorded in the context, not returned synchronously
        assert!(
            ctx.check_err().is_err(),
            "Expected error to be recorded when accessing SINGLE mode view from wrong stream"
        );

        // Test writing through a view in SINGLE mode with correct stream
        let mut view_single_mut = unified.slice_mut(30..40);
        let write_data2 = std::vec![777.0f32; 10];
        stream2.memcpy_htod(&write_data2, &mut view_single_mut)?;
        stream2.synchronize()?;

        // Verify the write succeeded
        let verify_data2 = stream2.clone_dtoh(&unified)?;
        for i in 30..40 {
            assert_eq!(
                verify_data2[i], 777.0,
                "Data written through SINGLE mode view should be 777.0"
            );
        }

        // Verify writing with wrong stream records an error
        let mut view_wrong_stream = unified.slice_mut(40..50);
        let _ = stream1.memcpy_htod(&write_data2, &mut view_wrong_stream);
        // The error is recorded in the context
        assert!(
            ctx.check_err().is_err(),
            "Expected error to be recorded when writing to SINGLE mode view from wrong stream"
        );

        Ok(())
    }
}
