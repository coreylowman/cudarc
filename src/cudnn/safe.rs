// No sync because of https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#thread-safety
use super::{result, result::CudnnError, sys};
use crate::driver::{CudaDevice, CudaStream};
use std::sync::Arc;

pub struct Cudnn {
    pub(crate) handle: sys::cudnnHandle_t,
    pub(crate) device: Arc<CudaDevice>,
}
impl Cudnn {
    /// Creates a new cudnn handle and sets the stream to the `device`'s stream.
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, CudnnError> {
        let handle = result::create_handle()?;
        let blas = Self { handle, device };
        unsafe { result::set_stream(handle, blas.device.stream as *mut _) }?;
        Ok(blas)
    }

    /// Sets the handle's current to either the stream specified, or the device's default work
    /// stream.
    ///
    /// # Safety
    /// This is unsafe because you can end up scheduling multiple concurrent kernels that all
    /// write to the same memory address.
    pub unsafe fn set_stream(&self, opt_stream: Option<&CudaStream>) -> Result<(), CudnnError> {
        match opt_stream {
            Some(s) => result::set_stream(self.handle, s.stream as *mut _),
            None => result::set_stream(self.handle, self.device.stream as *mut _),
        }
    }
}

impl Drop for Cudnn {
    fn drop(&mut self) {
        let handle = std::mem::replace(&mut self.handle, std::ptr::null_mut());
        if !handle.is_null() {
            unsafe { result::destroy_handle(handle) }.unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::driver::CudaDevice;

    use super::Cudnn;

    #[test]
    fn create_and_drop() {
        let _handle = Cudnn::new(CudaDevice::new(0).unwrap()).unwrap();
    }
}