use core::{marker::PhantomData, mem::size_of};

use super::sys::*;
use crate::{prelude::*, driver::sys::cuMemcpyHtoD_v2};

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
                cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
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
    pub fn create(allocation: CudaRc<[[[[T; W]; H]; C]; N]>, data: [[[[T; W]; H]; C]; N]) -> CudnnResult<Self> {
        let t = unsafe { Self::uninit(allocation) }?;
        unsafe { cuMemcpyHtoD_v2(t.data.t_cuda.cu_device_ptr, data.as_ptr() as *const _, t.size())}.result().unwrap();
        Ok(t)
    }

    pub unsafe fn uninit(allocation: CudaRc<[[[[T; W]; H]; C]; N]>) -> CudnnResult<Self> {
        Ok(Self {
            descriptor: Tensor4DDescriptor::create(TensorDescriptor::create()?)?,
            data: allocation
        })
    }

    pub fn size(&self) -> usize {
        size_of::<T>() * N * C * H * W
    }
}

#[cfg(test)]
mod tests {
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
            let mut data_type = std::mem::zeroed();
            let mut n = std::mem::zeroed();
            let mut h = std::mem::zeroed();
            let mut c = std::mem::zeroed();
            let mut w = std::mem::zeroed();
            let mut n_stride = std::mem::zeroed();
            let mut h_stride = std::mem::zeroed();
            let mut c_stride = std::mem::zeroed();
            let mut w_stride = std::mem::zeroed();
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
        Tensor4D::create(CudaDeviceBuilder::new(0).build().unwrap().alloc_zeros().unwrap(), [[[[-3.0, -2.0], [-1.0, -0.0]], [[0.0, 1.0], [2.0, 3.0]]]]).unwrap();
    }
}
