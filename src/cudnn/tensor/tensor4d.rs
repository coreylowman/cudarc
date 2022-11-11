use core::ffi::c_void;
use core::mem::size_of;

use alloc::rc::Rc;

use super::super::sys::*;
use super::descriptor::TensorDescriptor;
use crate::cudarc::CudaUniquePtr;
use crate::prelude::*;

/// A 4D-tensor with the `NCHW`-layout. Cloning this tensor only clones the
/// point and thus increases the reference count.
pub struct Tensor4D<T, const N: usize, const C: usize, const H: usize, const W: usize> {
    descriptor: Rc<TensorDescriptor<T, N, C, H, W>>,
    data: CudaRc<[[[[T; W]; H]; C]; N]>,
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
    #[inline(always)]
    pub fn get_descriptor(&self) -> cudnnTensorDescriptor_t {
        self.descriptor.get_descriptor()
    }

    #[inline(always)]
    pub fn get_descriptor_rc(&self) -> Rc<TensorDescriptor<T, N, C, H, W>> {
        Rc::clone(&self.descriptor)
    }

    #[inline(always)]
    pub fn get_data_ptr(&self) -> *const c_void {
        self.data.t_cuda.cu_device_ptr as *const _
    }

    #[inline(always)]
    pub fn get_data_ptr_mut(&self) -> *mut c_void {
        self.data.t_cuda.cu_device_ptr as *mut _
    }

    #[inline(always)]
    pub fn get_data(&self) -> CudaCudnnResult<Rc<[[[[T; W]; H]; C]; N]>> {
        self.data.clone().into_host().into_cuda_cudnn_result()
    }

    pub fn create(allocation: CudaRc<[[[[T; W]; H]; C]; N]>) -> CudaCudnnResult<Self> {
        Ok(Self {
            descriptor: Rc::new(TensorDescriptor::create()?),
            data: allocation,
        })
    }

    pub unsafe fn alloc_uninit(device: &Rc<CudaDevice>) -> CudaCudnnResult<Self> {
        Self::create(CudaRc {
            t_cuda: Rc::new(CudaUniquePtr::alloc(device)?),
            t_host: None,
        })
    }

    pub fn alloc_with(
        device: &Rc<CudaDevice>,
        value: [[[[T; W]; H]; C]; N],
    ) -> CudaCudnnResult<Self> {
        Self::create(device.take(Rc::new(value))?)
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
                cudnn_handle.get_handle(),
                self.get_descriptor(),
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
