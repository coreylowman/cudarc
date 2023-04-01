// No sync because of https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#thread-safety

use crate::{
    cudnn::{result, result::CudnnError, sys},
    driver::{CudaDevice, CudaStream},
};

use std::{marker::PhantomData, sync::Arc};

#[derive(Debug)]
pub struct Cudnn {
    pub(crate) handle: sys::cudnnHandle_t,
    pub(crate) device: Arc<CudaDevice>,
}

impl Cudnn {
    /// Creates a new cudnn handle and sets the stream to the `device`'s stream.
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, CudnnError> {
        let handle = result::create_handle()?;
        unsafe { result::set_stream(handle, device.stream as *mut _) }?;
        Ok(Self { handle, device })
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

/// Maps a rust type to a [sys::cudnnDataType_t]
pub trait CudnnDataType {
    const DATA_TYPE: sys::cudnnDataType_t;
}

macro_rules! cudnn_dtype {
    ($RustTy:ty, $CudnnTy:tt) => {
        impl CudnnDataType for $RustTy {
            const DATA_TYPE: sys::cudnnDataType_t = sys::cudnnDataType_t::$CudnnTy;
        }
    };
}

#[cfg(feature = "half")]
cudnn_dtype!(half::f16, CUDNN_DATA_HALF);
#[cfg(feature = "half")]
cudnn_dtype!(half::bf16, CUDNN_DATA_BFLOAT16);
cudnn_dtype!(f32, CUDNN_DATA_FLOAT);
cudnn_dtype!(f64, CUDNN_DATA_DOUBLE);
cudnn_dtype!(i8, CUDNN_DATA_INT8);
cudnn_dtype!(i32, CUDNN_DATA_INT32);
cudnn_dtype!(i64, CUDNN_DATA_INT64);
cudnn_dtype!(u8, CUDNN_DATA_UINT8);
cudnn_dtype!(bool, CUDNN_DATA_BOOLEAN);

#[derive(Debug)]
pub struct TensorDescriptor<T> {
    pub(crate) desc: sys::cudnnTensorDescriptor_t,
    #[allow(unused)]
    pub(crate) handle: Arc<Cudnn>,
    pub(crate) marker: PhantomData<T>,
}

impl Cudnn {
    pub fn create_tensor4d<T: CudnnDataType>(
        self: &Arc<Cudnn>,
        dims: [std::ffi::c_int; 4],
        strides: [std::ffi::c_int; 4],
    ) -> Result<TensorDescriptor<T>, CudnnError> {
        let desc = result::create_tensor_descriptor()?;
        let desc = TensorDescriptor {
            desc,
            handle: self.clone(),
            marker: PhantomData,
        };
        unsafe { result::set_tensor4d_descriptor_ex(desc.desc, T::DATA_TYPE, dims, strides) }?;
        Ok(desc)
    }

    pub fn create_tensornd<T: CudnnDataType>(
        self: &Arc<Cudnn>,
        dims: &[std::ffi::c_int],
        strides: &[std::ffi::c_int],
    ) -> Result<TensorDescriptor<T>, CudnnError> {
        assert_eq!(dims.len(), strides.len());
        let desc = result::create_tensor_descriptor()?;
        let desc = TensorDescriptor {
            desc,
            handle: self.clone(),
            marker: PhantomData,
        };
        unsafe {
            result::set_tensornd_descriptor(
                desc.desc,
                T::DATA_TYPE,
                dims.len() as std::ffi::c_int,
                dims.as_ptr(),
                strides.as_ptr(),
            )
        }?;
        Ok(desc)
    }
}

impl<T> Drop for TensorDescriptor<T> {
    fn drop(&mut self) {
        let desc = std::mem::replace(&mut self.desc, std::ptr::null_mut());
        if !desc.is_null() {
            unsafe { result::destroy_tensor_descriptor(desc) }.unwrap()
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
