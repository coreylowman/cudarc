use crate::driver::sys;

use super::core::{CudaSlice, CudaView, CudaViewMut};

pub trait DeviceSlice<T> {
    fn len(&self) -> usize;
    fn num_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> DeviceSlice<T> for CudaSlice<T> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<T> DeviceSlice<T> for CudaView<'_, T> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<T> DeviceSlice<T> for CudaViewMut<'_, T> {
    fn len(&self) -> usize {
        self.len
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
