use core::marker::PhantomData;
use core::mem::{size_of, MaybeUninit};

use alloc::rc::Rc;

use super::sys::*;
use crate::cudarc::CudaUniquePtr;
use crate::prelude::*;

/// recommended by docs <https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetTensorNdDescriptor>
pub struct Tensor4DDescriptor<T, const N: usize, const C: usize, const H: usize, const W: usize> {
    pub(crate) descriptor: TensorDescriptor,
    data_type: PhantomData<T>,
}
impl<T: TensorDataType, const N: usize, const C: usize, const H: usize, const W: usize>
    Tensor4DDescriptor<T, N, C, H, W>
{
    /// Sets the tensor to a NCHW layout.
    pub fn create(descriptor: TensorDescriptor) -> CudnnResult<Self> {
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
            descriptor,
            data_type: PhantomData,
        })
    }
}

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
pub type Tensor2D<T, const N: usize, const W: usize> = Tensor4D<T, N, 1, 1, W>;
pub type Tensor3D<T, const N: usize, const H: usize, const W: usize> = Tensor4D<T, N, 1, H, W>;
/// A 4D-tensor with the `NCHW`-layout. Cloning this tensor only clones the
/// point and thus increases the reference count.
pub struct Tensor4D<T, const N: usize, const C: usize, const H: usize, const W: usize> {
    pub(crate) descriptor: Rc<Tensor4DDescriptor<T, N, C, H, W>>,
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
        Ok(Self {
            descriptor: Rc::new(Tensor4DDescriptor::create(TensorDescriptor::create()?)?),
            data: allocation,
        })
    }

    pub unsafe fn alloc_uninit(device: &Rc<CudaDevice>) -> CudnnResult<Self> {
        Self::create(CudaRc {
            t_cuda: Rc::new(CudaUniquePtr::alloc(device).unwrap()),
            t_host: None,
        })
    }

    pub fn alloc_with(device: &Rc<CudaDevice>, value: [[[[T; W]; H]; C]; N]) -> CudnnResult<Self> {
        Self::create(device.take(Rc::new(value)).unwrap())
    }

    pub fn alloc_all_same(
        device: &Rc<CudaDevice>,
        cudnn_handle: &CudnnHandle,
        value: &T,
    ) -> CudnnResult<Self> {
        let s = unsafe { Self::alloc_uninit(device) }?;
        s.set_all(cudnn_handle, value)?;
        Ok(s)
    }

    pub fn set_all(&self, cudnn_handle: &CudnnHandle, v: &T) -> CudnnResult<()> {
        unsafe {
            cudnnSetTensor(
                cudnn_handle.0,
                self.descriptor.descriptor.0,
                self.data.t_cuda.cu_device_ptr as *mut _,
                v as *const _ as *const _,
            )
        }
        .result()
    }

    pub const fn size(&self) -> usize {
        size_of::<T>() * N * C * H * W
    }
}

#[cfg(test)]
mod tests {
    use core::mem::zeroed;

    use super::super::sys::*;
    use crate::prelude::*;

    #[test]
    fn test_create_descriptor() {
        let _descriptor = TensorDescriptor::create().unwrap();
    }

    #[test]
    fn test_descriptor_parameters() {
        let cut =
            Tensor4DDescriptor::<f32, 1, 2, 3, 4>::create(TensorDescriptor::create().unwrap())
                .unwrap();
        unsafe {
            let mut data_type = zeroed();
            let mut n = zeroed();
            let mut h = zeroed();
            let mut c = zeroed();
            let mut w = zeroed();
            let mut n_stride = zeroed();
            let mut h_stride = zeroed();
            let mut c_stride = zeroed();
            let mut w_stride = zeroed();
            cudnnGetTensor4dDescriptor(
                cut.descriptor.0,
                &mut data_type,
                &mut n,
                &mut c,
                &mut h,
                &mut w,
                &mut n_stride,
                &mut c_stride,
                &mut h_stride,
                &mut w_stride,
            )
            .result()
            .unwrap();
            assert_eq!(data_type, cudnnDataType_t::CUDNN_DATA_FLOAT);
            assert_eq!(n, 1);
            assert_eq!(c, 2);
            assert_eq!(h, 3);
            assert_eq!(w, 4);
            assert_eq!(n_stride, 2 * 3 * 4);
            assert_eq!(c_stride, 3 * 4);
            assert_eq!(h_stride, 4);
            assert_eq!(w_stride, 1);
        }
    }

    #[test]
    fn test_create_tensor() {
        let data = [[[[0.0, 1.0]]], [[[2.0, 3.0]]]];
        let t = Tensor2D::alloc_with(&CudaDeviceBuilder::new(0).build().unwrap(), data).unwrap();
        let on_gpu = *t.data.sync_release().unwrap().unwrap();
        assert_eq!(data, on_gpu);
    }
}
