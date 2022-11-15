use core::ffi::c_void;

use alloc::rc::Rc;

use crate::cudarc::CudaUniquePtr;
use crate::driver::result::memcpy_dtod_async;
use crate::prelude::*;

pub type DataType<T, const N: usize, const C: usize, const H: usize, const W: usize> =
    [[[[T; W]; H]; C]; N];

/// The data of a Tensor with the size of `NxCxHxW`. This is just a [CudaRc].
pub struct Tensor4DData<T, const N: usize, const C: usize, const H: usize, const W: usize>(
    CudaRc<DataType<T, N, C, H, W>>,
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
    /// Creates a new [Tensor4DData] by allocating new device memory on the
    /// [Rc<CudaDevice>] **without** initializing it.
    ///
    /// # Safety
    /// The data should not be read until it is initialized.
    pub unsafe fn alloc_uninit(device: &Rc<CudaDevice>) -> CudaCudnnResult<Self> {
        Ok(Self(CudaRc {
            t_cuda: Rc::new(CudaUniquePtr::alloc(device)?),
            t_host: None,
        }))
    }

    /// Creates a new [Tensor4DData] by allocating new device memory on the
    /// [Rc<CudaDevice>] initializing it with the value of `self`.
    pub fn clone_into_new(&self, device: &Rc<CudaDevice>) -> CudaCudnnResult<Self> {
        unsafe {
            let new = Self::alloc_uninit(device)?;
            memcpy_dtod_async::<DataType<T, N, C, H, W>>(
                new.get_address(),
                self.get_address(),
                device.cu_stream,
            )?;
            Ok(new)
        }
    }

    pub fn new(allocation: CudaRc<DataType<T, N, C, H, W>>) -> Self {
        Self(allocation)
    }

    /// The address in device memory.
    #[inline(always)]
    pub fn get_address(&self) -> u64 {
        self.0.t_cuda.cu_device_ptr
    }

    /// A pointer to the device memory.
    #[inline(always)]
    pub fn get_data_ptr(&self) -> *const c_void {
        self.0.t_cuda.cu_device_ptr as *const _
    }

    /// A mutable pointer to the device memory.
    #[inline(always)]
    pub fn get_data_ptr_mut(&mut self) -> *mut c_void {
        self.0.t_cuda.cu_device_ptr as *mut _
    }

    #[inline(always)]
    pub fn get_data(&self) -> CudaRc<DataType<T, N, C, H, W>> {
        self.0.clone()
    }

    #[inline(always)]
    pub fn get_data_ref(&self) -> &CudaRc<DataType<T, N, C, H, W>> {
        &self.0
    }

    #[inline(always)]
    pub fn get_data_ref_mut(&mut self) -> &mut CudaRc<DataType<T, N, C, H, W>> {
        &mut self.0
    }
}
impl<T: TensorDataType, const N: usize, const C: usize, const H: usize, const W: usize>
    Tensor4DData<T, N, C, H, W>
{
    /// The data of the device memory on the host memory (clones and then calls
    /// `into_host` on [CudaRc]).
    #[inline(always)]
    pub fn as_host(&self) -> CudaCudnnResult<Rc<DataType<T, N, C, H, W>>> {
        self.0.clone().into_host().into_cuda_cudnn_result()
    }
}
impl<T: NumElements, const N: usize, const C: usize, const H: usize, const W: usize>
    Tensor4DData<T, N, C, H, W>
{
    pub const fn get_numel(&self) -> u32 {
        Self::NUMEL as _
    }
}
impl<T: NumElements, const N: usize, const C: usize, const H: usize, const W: usize> NumElements
    for Tensor4DData<T, N, C, H, W>
{
    type Dtype = T;

    const NUMEL: usize = T::NUMEL * N * C * H * W;
}
unsafe impl<T: NumElements, const N: usize, const C: usize, const H: usize, const W: usize>
    IntoKernelParam for &Tensor4DData<T, N, C, H, W>
{
    fn into_kernel_param(self) -> *mut c_void {
        (&self.0).into_kernel_param()
    }
}
unsafe impl<T: NumElements, const N: usize, const C: usize, const H: usize, const W: usize>
    IntoKernelParam for &mut Tensor4DData<T, N, C, H, W>
{
    fn into_kernel_param(self) -> *mut c_void {
        (&mut self.0).into_kernel_param()
    }
}
