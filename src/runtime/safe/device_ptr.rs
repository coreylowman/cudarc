use super::core::{CudaSlice, CudaView, CudaViewMut};

use core::ffi::c_void;

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

impl<'a, T> DeviceSlice<T> for CudaView<'a, T> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<'a, T> DeviceSlice<T> for CudaViewMut<'a, T> {
    fn len(&self) -> usize {
        self.len
    }
}

/// Abstraction over [CudaSlice]/[CudaView]
pub trait DevicePtr<T>: DeviceSlice<T> {
    fn device_ptr(&self) -> *const c_void;
}

impl<T> DevicePtr<T> for CudaSlice<T> {
    fn device_ptr(&self) -> *const c_void {
        self.device_ptr as *const c_void
    }
}

impl<'a, T> DevicePtr<T> for CudaView<'a, T> {
    fn device_ptr(&self) -> *const c_void {
        self.ptr as *const c_void
    }
}

impl<'a, T> DevicePtr<T> for CudaViewMut<'a, T> {
    fn device_ptr(&self) -> *const c_void {
        self.ptr as *const c_void
    }
}

/// Abstraction over [CudaSlice]/[CudaViewMut]
pub trait DevicePtrMut<T>: DeviceSlice<T> {
    fn device_ptr_mut(&mut self) -> *mut c_void;
}

impl<T> DevicePtrMut<T> for CudaSlice<T> {
    fn device_ptr_mut(&mut self) -> *mut c_void {
        self.device_ptr
    }
}

impl<'a, T> DevicePtrMut<T> for CudaViewMut<'a, T> {
    fn device_ptr_mut(&mut self) -> *mut c_void {
        self.ptr
    }
}
