use core::mem::{size_of, MaybeUninit};

use alloc::rc::Rc;

use super::sys::*;
use crate::cudarc::CudaUniquePtr;
use crate::prelude::*;

pub struct TensorDescriptor(pub(crate) cudnnTensorDescriptor_t);
impl TensorDescriptor {
    pub fn create() -> CudnnResult<Self> {
        let mut descriptor = MaybeUninit::uninit();
        unsafe {
            cudnnCreateTensorDescriptor(descriptor.as_mut_ptr()).result()?;
            Ok(Self(descriptor.assume_init()))
        }
    }
}
impl Drop for TensorDescriptor {
    fn drop(&mut self) {
        unsafe { cudnnDestroyTensorDescriptor(self.0) }
            .result()
            .unwrap();
    }
}
/// A 4D-tensor with the `NCHW`-layout. Cloning this tensor only clones the
/// point and thus increases the reference count.
pub struct Tensor4D<T, const N: usize, const C: usize, const H: usize, const W: usize> {
    pub(crate) descriptor: Rc<TensorDescriptor>,
    pub(crate) data: CudaRc<[[[[T; W]; H]; C]; N]>,
}
impl<T, const N: usize, const C: usize, const H: usize, const W: usize> Clone
    for Tensor4D<T, N, C, H, W>
{
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            descriptor: Rc::clone(&self.descriptor),
        }
    }
}
impl<T: TensorDataType, const N: usize, const C: usize, const H: usize, const W: usize>
    Tensor4D<T, N, C, H, W>
{
    /// # Safety
    /// Tensor must be initialized or not be read from until it is so.
    pub fn create(allocation: CudaRc<[[[[T; W]; H]; C]; N]>) -> CudnnResult<Self> {
        let descriptor = TensorDescriptor::create()?;
        unsafe {
            cudnnSetTensor4dDescriptor(
                descriptor.0,
                T::get_tensor_format(),
                T::get_data_type(),
                N as _,
                C as _,
                H as _,
                W as _,
            )
        }
        .result()?;
        Ok(Self {
            descriptor: Rc::new(descriptor),
            data: allocation,
        })
    }

    pub unsafe fn alloc_uninit(device: &Rc<CudaDevice>) -> CudaCudnnResult<Self> {
        Self::create(CudaRc {
            t_cuda: Rc::new(CudaUniquePtr::alloc(device).into_cuda_cudnn_result()?),
            t_host: None,
        })
        .into_cuda_cudnn_result()
    }

    pub fn alloc_with(
        device: &Rc<CudaDevice>,
        value: [[[[T; W]; H]; C]; N],
    ) -> CudaCudnnResult<Self> {
        Self::create(device.take(Rc::new(value)).into_cuda_cudnn_result()?).into_cuda_cudnn_result()
    }

    pub fn alloc_all_same(
        device: &Rc<CudaDevice>,
        cudnn_handle: &CudnnHandle,
        value: &T,
    ) -> CudaCudnnResult<Self> {
        let s = unsafe { Self::alloc_uninit(device) }?;
        s.set_all(cudnn_handle, value)?;
        Ok(s)
    }

    pub fn set_all(&self, cudnn_handle: &CudnnHandle, v: &T) -> CudaCudnnResult<()> {
        unsafe {
            cudnnSetTensor(
                cudnn_handle.0,
                self.descriptor.0,
                self.data.t_cuda.cu_device_ptr as *mut _,
                v as *const _ as *const _,
            )
        }
        .result()
        .into_cuda_cudnn_result()
    }

    pub const fn size(&self) -> usize {
        size_of::<T>() * N * C * H * W
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn test_create_descriptor() {
        let _descriptor = TensorDescriptor::create().unwrap();
    }

    #[test]
    fn test_create_tensor() {
        let data = [[[[0.0, 1.0]]], [[[2.0, 3.0]]]];
        let t = Tensor4D::alloc_with(&CudaDeviceBuilder::new(0).build().unwrap(), data).unwrap();
        let on_gpu = *t.data.sync_release().unwrap().unwrap();
        assert_eq!(data, on_gpu);
    }
}
