use super::{CudaDevice, RuntimeError};

use crate::driver::result;
use crate::runtime::sys;

impl CudaDevice {
    /// Binds the device to the calling thread. You must call this before
    /// using the device on a separate thread!
    pub fn bind_to_thread(&self) -> Result<(), RuntimeError> {
        let cuda_result = unsafe { result::ctx::set_current(self.cu_primary_ctx) };
        if cuda_result.is_err() {
            return Err(RuntimeError(sys::cudaError::cudaErrorInvalidValue));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::thread;

    #[test]
    fn test_threading() {
        let dev1 = CudaDevice::new(0).unwrap();
        let dev2 = dev1.clone();

        let thread1 = thread::spawn(move || {
            dev1.bind_to_thread()?;
            dev1.alloc_zeros::<f32>(10)
        });
        let thread2 = thread::spawn(move || {
            dev2.bind_to_thread()?;
            dev2.alloc_zeros::<f32>(10)
        });

        let _: crate::runtime::CudaSlice<f32> = thread1.join().unwrap().unwrap();
        let _: crate::runtime::CudaSlice<f32> = thread2.join().unwrap().unwrap();
    }
}
