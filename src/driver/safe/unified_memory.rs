use core::marker::PhantomData;
use std::sync::Arc;

use crate::driver::{result, sys};

use super::{
    CudaContext, CudaEvent, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, DeviceSlice,
    DriverError, HostSlice,
};

/// Unified memory allocated with [CudaContext::alloc_unified()] (via [cuMemAllocManaged](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32)).
///
/// See [cuda docs for Unified Addressing/Unified Memory](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html#group__CUDA__UNIFIED)
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

impl CudaContext {
    /// Allocates managed memory using [cuMemAllocManaged](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32).
    ///
    /// If `attach_global` is true, then allocates the memory with flag [sys::CUmemAttach_flags::CU_MEM_ATTACH_GLOBAL],
    /// otherwise uses flag [sys::CUmemAttach_flags::CU_MEM_ATTACH_HOST].
    ///
    /// Note that only these two flags are valid during allocation, you can change the
    /// attach mode later via [UnifiedSlice::attach()]
    pub unsafe fn alloc_unified<T: DeviceRepr>(
        self: &Arc<Self>,
        len: usize,
        attach_global: bool,
    ) -> Result<UnifiedSlice<T>, DriverError> {
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
    /// See [cuStreamAttachMemAsync cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g6e468d680e263e7eba02a56643c50533)
    pub fn attach(
        &mut self,
        stream: &Arc<CudaStream>,
        flags: sys::CUmemAttach_flags,
    ) -> Result<(), DriverError> {
        self.event.synchronize()?;
        self.stream = stream.clone();
        unsafe {
            result::stream::attach_mem_async(
                self.stream.cu_stream,
                self.cu_device_ptr,
                self.len * std::mem::size_of::<T>(),
                flags,
            )
        }
    }

    /// See [cuMemPrefetchAsync_v2 cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1gaf4f188a71891ad6a71fdd2850c8d638)
    pub fn prefetch(&self) -> Result<(), DriverError> {
        let location = match self.attach_mode {
            sys::CUmemAttach_flags_enum::CU_MEM_ATTACH_GLOBAL
            | sys::CUmemAttach_flags_enum::CU_MEM_ATTACH_SINGLE => {
                // From the docs
                // > Specifying CU_MEM_LOCATION_TYPE_DEVICE for CUmemLocation::type will prefetch memory to GPU specified by device ordinal CUmemLocation::id which must have non-zero value for the device attribute CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS. Additionally, hStream must be associated with a device that has a non-zero value for the device attribute CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS.
                assert!(self.concurrent_managed_access);
                sys::CUmemLocation {
                    type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                    id: self.stream.ctx.ordinal as i32,
                }
            }
            sys::CUmemAttach_flags_enum::CU_MEM_ATTACH_HOST => {
                // From the docs:
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
        // If the CU_MEM_ATTACH_HOST flag is specified, the program makes a guarantee that it won't access the memory on the device from any stream on a device that has a zero value for the device attribute
        // If the CU_MEM_ATTACH_SINGLE flag is specified and hStream is associated with a device that has a zero value for the device attribute CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, the program makes a guarantee that it will only access the memory on the device from hStream
        todo!()
    }
}

impl<T> DevicePtrMut<T> for UnifiedSlice<T> {
    fn device_ptr_mut<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (sys::CUdeviceptr, super::SyncOnDrop<'a>) {
        // If the CU_MEM_ATTACH_HOST flag is specified, the program makes a guarantee that it won't access the memory on the device from any stream on a device that has a zero value for the device attribute
        // If the CU_MEM_ATTACH_SINGLE flag is specified and hStream is associated with a device that has a zero value for the device attribute CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, the program makes a guarantee that it will only access the memory on the device from hStream
        todo!()
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
        // When memory is associated with a single stream, the Unified Memory system will allow CPU access to this memory region so long as all operations in hStream have completed, regardless of whether other streams are active. In effect, this constrains exclusive ownership of the managed memory region by an active GPU to per-stream activity instead of whole-GPU activity.
        // Accessing memory on the device from streams that are not associated with it will produce undefined results. No error checking is performed by the Unified Memory system to ensure that kernels launched into other streams do not access this region.
        todo!("prefetch?")
    }

    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (&'a mut [T], super::SyncOnDrop<'a>) {
        // When memory is associated with a single stream, the Unified Memory system will allow CPU access to this memory region so long as all operations in hStream have completed, regardless of whether other streams are active. In effect, this constrains exclusive ownership of the managed memory region by an active GPU to per-stream activity instead of whole-GPU activity.
        // Accessing memory on the device from streams that are not associated with it will produce undefined results. No error checking is performed by the Unified Memory system to ensure that kernels launched into other streams do not access this region.
        todo!("prefetch?")
    }
}
