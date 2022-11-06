use core::{marker::PhantomData, mem::size_of};

use super::sys::*;
use crate::{prelude::*, driver::result::memcpy_htod_async};

/// recommended by docs <https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetTensorNdDescriptor>
pub struct Tensor4DDescriptor<T, const N: usize, const C: usize, const H: usize, const W: usize> {
    pub(crate) descriptor: TensorDescriptor,
    data_type:  PhantomData<T>,
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
        let mut descriptor: Self = unsafe { std::mem::zeroed() };
        unsafe { cudnnCreateTensorDescriptor(&mut descriptor.0 as *mut _) }.result()?;
        Ok(descriptor)
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
pub struct Tensor4D<T, const N: usize, const C: usize, const H: usize, const W: usize> {
    pub(crate) descriptor: Tensor4DDescriptor<T, N, C, H, W>,
    pub(crate) data: CudaRc<[[[[T; W]; H]; C]; N]>,
}
impl<T: TensorDataType, const N: usize, const C: usize, const H: usize, const W: usize> Tensor4D<T, N, C, H, W> {
    pub fn create_async(allocation: CudaRc<[[[[T; W]; H]; C]; N]>, data: &[[[[T; W]; H]; C]; N]) -> CudnnResult<Self> {
        let t = unsafe { Self::uninit(allocation) }?;
        unsafe { memcpy_htod_async(t.data.t_cuda.cu_device_ptr, data, t.data.device().cu_stream).unwrap();}
        Ok(t)
    }

    ///
    /// # Safety
    /// Tensor must be not be read from as it is uninitialized.
    pub unsafe fn uninit(allocation: CudaRc<[[[[T; W]; H]; C]; N]>) -> CudnnResult<Self> {
        Ok(Self {
            descriptor: Tensor4DDescriptor::create(TensorDescriptor::create()?)?,
            data: allocation
        })
    }

    pub const fn size(&self) -> usize {
        size_of::<T>() * N * C * H * W
    }
}

#[cfg(test)]
mod tests {
    use core::mem::zeroed;

    use alloc::rc::Rc;

    use super::super::sys::*;
    use crate::prelude::*;

    #[test]
    fn test_create_descriptor() {
        let _descriptor = TensorDescriptor::create().unwrap();
    }

    #[test]
    fn test_descriptor_parameters() {
        let cut = Tensor4DDescriptor::<f32, 1, 2, 3, 4>::create(
            TensorDescriptor::create().unwrap(),
        )
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
        let t = Tensor2D::create_async(CudaDeviceBuilder::new(0).build().unwrap().take(Rc::new(unsafe { zeroed() })).unwrap(), &data).unwrap();
        let on_gpu = *t.data.sync_release().unwrap().unwrap();
        assert_eq!(data, on_gpu);
    }
}
