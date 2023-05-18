use core::marker::PhantomData;
use std::fs::File;
use std::ops::Range;
use std::sync::Arc;

use super::{CudaDevice, DevicePtr, DeviceSlice};
use crate::driver::{result, sys, DriverError};

impl CudaDevice {
    #[cfg(any(unix, windows))]
    pub unsafe fn import_external_memory(
        self: &Arc<Self>,
        file: File,
        size: u64,
    ) -> Result<ExternalMemory, DriverError> {
        #[cfg(unix)]
        let external_memory = unsafe {
            use std::os::fd::AsRawFd;
            result::external_memory::import_external_memory_opaque_fd(file.as_raw_fd(), size)
        }?;
        #[cfg(windows)]
        let external_memory = unsafe {
            use std::os::windows::io::AsRawHandle;
            result::external_memory::import_external_memory_opaque_win32(file.as_raw_handle(), size)
        }?;
        Ok(ExternalMemory {
            external_memory,
            size,
            _device: self.clone(),
            _file: file,
        })
    }
}

pub struct ExternalMemory {
    external_memory: sys::CUexternalMemory,
    size: u64,
    _device: Arc<CudaDevice>,
    _file: std::fs::File,
}

impl Drop for ExternalMemory {
    fn drop(&mut self) {
        unsafe { result::external_memory::destroy_external_memory(self.external_memory) }.unwrap();
        // From CUDA docs, when successfully importing UNIX file descriptor:
        // Ownership of the file descriptor is transferred to the CUDA driver when the handle is imported successfully.
        // Performing any operations on the file descriptor after it is imported results in undefined behavior.
        #[cfg(unix)]
        core::mem::forget(self._file);
    }
}

impl ExternalMemory {
    pub fn map_all(&self) -> Result<MappedBuffer<'_>, DriverError> {
        self.map_range(0..self.size as usize)
    }

    pub fn map_range(&self, range: Range<usize>) -> Result<MappedBuffer<'_>, DriverError> {
        assert!(range.start as u64 <= self.size);
        assert!(range.end as u64 <= self.size);
        let device_ptr = unsafe {
            result::external_memory::get_mapped_buffer(
                self.external_memory,
                range.start as u64,
                range.len() as u64,
            )
        }?;
        Ok(MappedBuffer {
            device_ptr,
            len: range.len(),
            _marker: PhantomData,
        })
    }
}

pub struct MappedBuffer<'a> {
    device_ptr: sys::CUdeviceptr,
    len: usize,
    _marker: PhantomData<&'a ExternalMemory>,
}

impl Drop for MappedBuffer<'_> {
    fn drop(&mut self) {
        unsafe { result::external_memory::memory_free(self.device_ptr) }.unwrap()
    }
}

impl DeviceSlice<u8> for MappedBuffer<'_> {
    fn len(&self) -> usize {
        self.len
    }
}

impl DevicePtr<u8> for MappedBuffer<'_> {
    fn device_ptr(&self) -> &sys::CUdeviceptr {
        &self.device_ptr
    }
}
