use core::ffi::c_void;
use core::mem::size_of;

use alloc::rc::Rc;

use super::super::sys::*;
use super::descriptor::TensorDescriptor;
use crate::cudarc::CudaUniquePtr;
use crate::prelude::*;

/// A 4D-tensor with `NCHW layout`. Cloning this tensor only clones the
/// pointer to the [TensorDescriptor] and [Tensor4DData] and thus increases the
/// reference count.
pub struct Tensor4D<T, const N: usize, const C: usize, const H: usize, const W: usize> {
    descriptor: Rc<TensorDescriptor<T, N, C, H, W>>,
    data: Tensor4DData<T, N, C, H, W>,
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
    /// The pointer to the descriptor.
    #[inline(always)]
    pub fn get_descriptor(&self) -> cudnnTensorDescriptor_t {
        self.descriptor.get_descriptor()
    }

    /// Returns the [TensorDescriptor] after cloning it.
    #[inline(always)]
    pub fn get_descriptor_rc(&self) -> Rc<TensorDescriptor<T, N, C, H, W>> {
        Rc::clone(&self.descriptor)
    }

    /// A pointer to the device memory.
    #[inline(always)]
    pub fn get_data_ptr(&self) -> *const c_void {
        self.data.get_data_ptr()
    }

    /// A mutable pointer to the device memory.
    #[inline(always)]
    pub fn get_data_ptr_mut(&self) -> *mut c_void {
        self.data.get_data_ptr_mut()
    }

    /// Returns the [Tensor4DData] after cloning it.
    #[inline(always)]
    pub fn get_data(&self) -> Tensor4DData<T, N, C, H, W> {
        self.data.clone()
    }

    /// Split the `Tensor4D` into a `TensorDescriptor` and a `Tensor4DData`,
    /// after cloning them.
    #[inline(always)]
    pub fn as_split(
        &self,
    ) -> (
        Rc<TensorDescriptor<T, N, C, H, W>>,
        Tensor4DData<T, N, C, H, W>,
    ) {
        (self.get_descriptor_rc(), self.get_data())
    }

    /// Creates a new [Tensor4D] by a [Rc<TensorDescriptor>] and a
    /// [Tensor4DData].
    pub fn new(
        descriptor: Rc<TensorDescriptor<T, N, C, H, W>>,
        data: Tensor4DData<T, N, C, H, W>,
    ) -> Self {
        Self { descriptor, data }
    }

    /// Creates a new [Tensor4D] by a [Tensor4DData] (so this creates a new
    /// [TensorDescriptor]).
    pub fn create(allocation: Tensor4DData<T, N, C, H, W>) -> CudaCudnnResult<Self> {
        Ok(Self::new(Rc::new(TensorDescriptor::create()?), allocation))
    }

    /// Creates a new [Tensor4D] by allocating new device memory on the
    /// [Rc<CudaDevice>] **without** initializing it (this also creates a new
    /// [TensorDescriptor]).
    ///
    /// # Safety
    /// The data on this tensor should not be read until it is initialized.
    pub unsafe fn alloc_uninit(device: &Rc<CudaDevice>) -> CudaCudnnResult<Self> {
        Self::create(Tensor4DData::new(CudaRc {
            t_cuda: Rc::new(CudaUniquePtr::alloc(device)?),
            t_host: None,
        }))
    }

    /// Creates a new [Tensor4D] by allocating new device memory on the
    /// [Rc<CudaDevice>] and initializing it with `value` (this also creates a
    /// new [TensorDescriptor]).
    pub fn alloc_with(
        device: &Rc<CudaDevice>,
        value: [[[[T; W]; H]; C]; N],
    ) -> CudaCudnnResult<Self> {
        Self::create(Tensor4DData::new(device.take(Rc::new(value))?))
    }

    /// Creates a new [Tensor4D] by allocating new device memory on the
    /// [Rc<CudaDevice>] and setting everything to `value` (this also creates a
    /// new [TensorDescriptor]).
    pub fn alloc_all_same(
        device: &Rc<CudaDevice>,
        cudnn_handle: &CudnnHandle,
        value: &T,
    ) -> CudaCudnnResult<Self> {
        let s = unsafe { Self::alloc_uninit(device) }?;
        s.set_all(cudnn_handle, value)?;
        Ok(s)
    }

    /// Sets all the data of this tensor to `value`.
    ///
    /// # See also
    /// <https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetTensor>
    pub fn set_all(&self, cudnn_handle: &CudnnHandle, value: &T) -> CudaCudnnResult<()> {
        unsafe {
            cudnnSetTensor(
                cudnn_handle.get_handle(),
                self.get_descriptor(),
                self.data.get_data_ptr_mut(),
                value as *const _ as *const _,
            )
        }
        .result()
    }

    /// The size of this tensor is known at compile time.
    pub const fn size(&self) -> usize {
        size_of::<T>() * N * C * H * W
    }
}
