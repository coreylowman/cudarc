use core::ffi::c_void;

use alloc::rc::Rc;

use crate::prelude::*;

/// The data of a Tensor with the size of `NxCxHxW`. This is just a [CudaRc].
pub struct Tensor4DData<T, const N: usize, const C: usize, const H: usize, const W: usize>(
    CudaRc<[[[[T; W]; H]; C]; N]>,
);
impl<T, const N: usize, const C: usize, const H: usize, const W: usize> Clone
    for Tensor4DData<T, N, C, H, W>
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T, const N: usize, const C: usize, const H: usize, const W: usize>
    Tensor4DData<T, N, C, H, W>
{
    pub fn new(allocation: CudaRc<[[[[T; W]; H]; C]; N]>) -> Self {
        Self(allocation)
    }

    /// A pointer to the device memory.
    #[inline(always)]
    pub fn get_data_ptr(&self) -> *const c_void {
        self.0.t_cuda.cu_device_ptr as *const _
    }

    /// A mutable pointer to the device memory.
    #[inline(always)]
    pub fn get_data_ptr_mut(&self) -> *mut c_void {
        self.0.t_cuda.cu_device_ptr as *mut _
    }
}
impl<T: TensorDataType, const N: usize, const C: usize, const H: usize, const W: usize>
    Tensor4DData<T, N, C, H, W>
{
    /// The data of the device memory on the host memory (clones and then calls
    /// `into_host` on [CudaRc]).
    #[inline(always)]
    pub fn as_host(&self) -> CudaCudnnResult<Rc<[[[[T; W]; H]; C]; N]>> {
        self.0.clone().into_host().into_cuda_cudnn_result()
    }
}
