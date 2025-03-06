use crate::driver::{result::DriverError, sys};

use super::core::{CudaEvent, CudaSlice, CudaStream, CudaView, CudaViewMut};

pub trait DeviceSlice<T> {
    fn len(&self) -> usize;
    fn num_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn cu_device_ptr(&self) -> sys::CUdeviceptr;
    fn event(&self) -> &CudaEvent;
    fn stream(&self) -> &CudaStream;
    fn record_use(&self, stream: &CudaStream) -> Result<(), DriverError> {
        if self.stream() != stream {
            self.event().record(stream)?;
        }
        Ok(())
    }
}

impl<T> DeviceSlice<T> for CudaSlice<T> {
    fn len(&self) -> usize {
        self.len
    }
    fn cu_device_ptr(&self) -> sys::CUdeviceptr {
        self.cu_device_ptr
    }
    fn event(&self) -> &CudaEvent {
        &self.event
    }
    fn stream(&self) -> &CudaStream {
        &self.stream
    }
}

impl<T> DeviceSlice<T> for CudaView<'_, T> {
    fn len(&self) -> usize {
        self.len
    }
    fn cu_device_ptr(&self) -> sys::CUdeviceptr {
        self.ptr
    }
    fn event(&self) -> &CudaEvent {
        self.event
    }
    fn stream(&self) -> &CudaStream {
        self.stream
    }
}

impl<T> DeviceSlice<T> for CudaViewMut<'_, T> {
    fn len(&self) -> usize {
        self.len
    }
    fn cu_device_ptr(&self) -> sys::CUdeviceptr {
        self.ptr
    }
    fn event(&self) -> &CudaEvent {
        self.event
    }
    fn stream(&self) -> &CudaStream {
        self.stream
    }
}

/// Abstraction over [CudaSlice]/[CudaView]
pub trait DevicePtr<T>: DeviceSlice<T> {
    fn device_ptr(&self) -> &sys::CUdeviceptr;
}

impl<T> DevicePtr<T> for CudaSlice<T> {
    fn device_ptr(&self) -> &sys::CUdeviceptr {
        &self.cu_device_ptr
    }
}

impl<T> DevicePtr<T> for CudaView<'_, T> {
    fn device_ptr(&self) -> &sys::CUdeviceptr {
        &self.ptr
    }
}

impl<T> DevicePtr<T> for CudaViewMut<'_, T> {
    fn device_ptr(&self) -> &sys::CUdeviceptr {
        &self.ptr
    }
}

/// Abstraction over [CudaSlice]/[CudaViewMut]
pub trait DevicePtrMut<T>: DeviceSlice<T> {
    fn device_ptr_mut(&mut self) -> &mut sys::CUdeviceptr;
}

impl<T> DevicePtrMut<T> for CudaSlice<T> {
    fn device_ptr_mut(&mut self) -> &mut sys::CUdeviceptr {
        &mut self.cu_device_ptr
    }
}

impl<T> DevicePtrMut<T> for CudaViewMut<'_, T> {
    fn device_ptr_mut(&mut self) -> &mut sys::CUdeviceptr {
        &mut self.ptr
    }
}
