use std::{sync::Arc, vec::Vec};

use super::{CudaContext, CudaEvent, CudaStream, DeviceRepr, DriverError, ValidAsZeroBits};

use crate::driver::{result, sys};

pub trait HostSlice<T> {
    /// # Safety
    /// This is **only** safe if the resulting slice is used with `stream`. Otherwise
    /// You may run into device synchronization errors
    unsafe fn stream_synced_slice(&self, stream: &CudaStream) -> Result<&[T], DriverError>;
    /// # Safety
    /// This is **only** safe if the resulting slice is used with `stream`. Otherwise
    /// You may run into device synchronization errors
    unsafe fn stream_synced_mut_slice(
        &mut self,
        stream: &CudaStream,
    ) -> Result<&mut [T], DriverError>;
    fn record_use(&self, stream: &CudaStream) -> Result<(), DriverError>;
}

impl<T, const N: usize> HostSlice<T> for [T; N] {
    unsafe fn stream_synced_slice(&self, _stream: &CudaStream) -> Result<&[T], DriverError> {
        Ok(self)
    }

    unsafe fn stream_synced_mut_slice(
        &mut self,
        _stream: &CudaStream,
    ) -> Result<&mut [T], DriverError> {
        Ok(self)
    }

    fn record_use(&self, stream: &CudaStream) -> Result<(), DriverError> {
        stream.synchronize()
    }
}

impl<T> HostSlice<T> for [T] {
    unsafe fn stream_synced_slice(&self, _stream: &CudaStream) -> Result<&[T], DriverError> {
        Ok(self)
    }
    unsafe fn stream_synced_mut_slice(
        &mut self,
        _stream: &CudaStream,
    ) -> Result<&mut [T], DriverError> {
        Ok(self)
    }
    fn record_use(&self, stream: &CudaStream) -> Result<(), DriverError> {
        stream.synchronize()
    }
}

impl<T> HostSlice<T> for Vec<T> {
    unsafe fn stream_synced_slice(&self, _stream: &CudaStream) -> Result<&[T], DriverError> {
        Ok(self)
    }
    unsafe fn stream_synced_mut_slice(
        &mut self,
        _stream: &CudaStream,
    ) -> Result<&mut [T], DriverError> {
        Ok(self)
    }
    fn record_use(&self, stream: &CudaStream) -> Result<(), DriverError> {
        stream.synchronize()
    }
}

#[derive(Debug)]
pub struct PinnedHostSlice<T> {
    pub(crate) ptr: *mut T,
    pub(crate) len: usize,
    pub(crate) event: CudaEvent,
}

impl<T> Drop for PinnedHostSlice<T> {
    fn drop(&mut self) {
        self.event.synchronize().unwrap();
        unsafe { result::free_host(self.ptr as _) }.unwrap();
    }
}

impl CudaContext {
    /// Allocates page locked host memory with [sys::CU_MEMHOSTALLOC_WRITECOMBINED] flags.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9)
    ///
    /// # Safety
    /// 1. This is unsafe because the memory is unset after this call.
    pub unsafe fn alloc_pinned<T: DeviceRepr>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<PinnedHostSlice<T>, DriverError> {
        self.bind_to_thread()?;
        let ptr = result::malloc_host(
            len * std::mem::size_of::<T>(),
            sys::CU_MEMHOSTALLOC_WRITECOMBINED,
        )?;
        let event = self.empty_event(None)?;
        Ok(PinnedHostSlice::new(ptr as _, len, event))
    }
}

impl<T> PinnedHostSlice<T> {
    /// Creates a new pinned host slice.
    ///
    /// # Safety
    /// 1. `ptr` should be returned from [result::malloc_host()].
    pub unsafe fn new(ptr: *mut T, len: usize, event: CudaEvent) -> Self {
        assert!(!ptr.is_null());
        assert!(len * std::mem::size_of::<T>() < isize::MAX as usize);
        assert!(ptr.is_aligned());
        Self { ptr, len, event }
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.event.ctx
    }

    /// The size of the slice
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: ValidAsZeroBits> PinnedHostSlice<T> {
    /// Waits for any scheduled work to complete and then returns a refernce
    /// to the host side data.
    pub fn as_ptr(&self) -> Result<*const T, DriverError> {
        self.event.synchronize()?;
        Ok(self.ptr)
    }

    /// Waits for any scheduled work to complete and then returns a refernce
    /// to the host side data.
    pub fn as_mut_ptr(&mut self) -> Result<*mut T, DriverError> {
        self.event.synchronize()?;
        Ok(self.ptr)
    }

    /// Waits for any scheduled work to complete and then returns a refernce
    /// to the host side data.
    pub fn as_slice(&self) -> Result<&[T], DriverError> {
        self.event.synchronize()?;
        Ok(unsafe { std::slice::from_raw_parts(self.ptr, self.len) })
    }

    /// Waits for any scheduled work to complete and then returns a refernce
    /// to the host side data.
    pub fn as_mut_slice(&mut self) -> Result<&mut [T], DriverError> {
        self.event.synchronize()?;
        Ok(unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) })
    }
}

impl<T> HostSlice<T> for PinnedHostSlice<T> {
    unsafe fn stream_synced_slice(&self, stream: &CudaStream) -> Result<&[T], DriverError> {
        stream.wait(&self.event)?;
        Ok(std::slice::from_raw_parts(self.ptr, self.len))
    }

    unsafe fn stream_synced_mut_slice(
        &mut self,
        stream: &CudaStream,
    ) -> Result<&mut [T], DriverError> {
        stream.wait(&self.event)?;
        Ok(std::slice::from_raw_parts_mut(self.ptr, self.len))
    }

    fn record_use(&self, stream: &CudaStream) -> Result<(), DriverError> {
        self.event.record(stream)
    }
}
