use crate::driver::{result::DriverError, sys};

use super::{
    core::{CudaSlice, CudaStream, CudaView, CudaViewMut},
    CudaEvent,
};

pub trait DeviceSlice<T> {
    fn len(&self) -> usize;
    fn num_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn stream(&self) -> &CudaStream;
}

impl<T> DeviceSlice<T> for CudaSlice<T> {
    fn len(&self) -> usize {
        self.len
    }
    fn stream(&self) -> &CudaStream {
        &self.stream
    }
}

impl<T> DeviceSlice<T> for CudaView<'_, T> {
    fn len(&self) -> usize {
        self.len
    }
    fn stream(&self) -> &CudaStream {
        self.stream
    }
}

impl<T> DeviceSlice<T> for CudaViewMut<'_, T> {
    fn len(&self) -> usize {
        self.len
    }
    fn stream(&self) -> &CudaStream {
        self.stream
    }
}

/// Abstraction over [CudaSlice]/[CudaView]
pub trait DevicePtr<T>: DeviceSlice<T> {
    fn device_ptr(&self) -> &sys::CUdeviceptr;
    fn read_event(&self) -> &CudaEvent;
    fn block_for_read(&self, stream: &CudaStream) -> Result<(), DriverError>;
    fn record_read(&self, stream: &CudaStream) -> Result<(), DriverError> {
        if self.stream() != stream {
            self.read_event().record(stream)?;
        }
        Ok(())
    }
}

impl<T> DevicePtr<T> for CudaSlice<T> {
    fn device_ptr(&self) -> &sys::CUdeviceptr {
        &self.cu_device_ptr
    }
    fn read_event(&self) -> &CudaEvent {
        &self.read
    }
    fn block_for_read(&self, stream: &CudaStream) -> Result<(), DriverError> {
        stream.wait(&self.write)
    }
}

impl<T> DevicePtr<T> for CudaView<'_, T> {
    fn device_ptr(&self) -> &sys::CUdeviceptr {
        &self.ptr
    }
    fn read_event(&self) -> &CudaEvent {
        self.read
    }
    fn block_for_read(&self, stream: &CudaStream) -> Result<(), DriverError> {
        stream.wait(self.write)
    }
}

impl<T> DevicePtr<T> for CudaViewMut<'_, T> {
    fn device_ptr(&self) -> &sys::CUdeviceptr {
        &self.ptr
    }
    fn read_event(&self) -> &CudaEvent {
        self.read
    }
    fn block_for_read(&self, stream: &CudaStream) -> Result<(), DriverError> {
        stream.wait(self.write)
    }
}

/// Abstraction over [CudaSlice]/[CudaViewMut]
pub trait DevicePtrMut<T>: DevicePtr<T> {
    fn device_ptr_mut(&mut self) -> &mut sys::CUdeviceptr;
    fn write_event(&self) -> &CudaEvent;
    fn block_for_write(&self, stream: &CudaStream) -> Result<(), DriverError>;
    fn record_write(&mut self, stream: &CudaStream) -> Result<(), DriverError> {
        if self.stream() != stream {
            self.read_event().record(stream)?;
            self.write_event().record(stream)?;
        }
        Ok(())
    }
}

impl<T> DevicePtrMut<T> for CudaSlice<T> {
    fn device_ptr_mut(&mut self) -> &mut sys::CUdeviceptr {
        &mut self.cu_device_ptr
    }
    fn write_event(&self) -> &CudaEvent {
        &self.write
    }
    fn block_for_write(&self, stream: &CudaStream) -> Result<(), DriverError> {
        stream.wait(&self.read)?;
        stream.wait(&self.write)
    }
}

impl<T> DevicePtrMut<T> for CudaViewMut<'_, T> {
    fn device_ptr_mut(&mut self) -> &mut sys::CUdeviceptr {
        &mut self.ptr
    }
    fn write_event(&self) -> &CudaEvent {
        self.write
    }
    fn block_for_write(&self, stream: &CudaStream) -> Result<(), DriverError> {
        stream.wait(self.read)?;
        stream.wait(self.write)
    }
}
