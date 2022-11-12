mod data;
mod data_type;
mod descriptor;
mod operation;

pub use data::*;
pub use data_type::*;
pub use descriptor::*;
pub use operation::*;

use core::ffi::c_void;
use core::mem::size_of;

use alloc::rc::Rc;

use crate::cudarc::CudaUniquePtr;
use crate::cudnn::sys::*;
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

    /// Split the [Tensor4D] into a [TensorDescriptor] and a [Tensor4DData],
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
    pub fn create(data: Tensor4DData<T, N, C, H, W>) -> CudaCudnnResult<Self> {
        Ok(Self::new(Rc::new(TensorDescriptor::create()?), data))
    }

    /// Creates a new [Tensor4D] by a [CudaRc] (so this creates a new
    /// [TensorDescriptor]).
    pub fn create_with(allocation: CudaRc<DataType<T, N, C, H, W>>) -> CudaCudnnResult<Self> {
        Ok(Self::new(
            Rc::new(TensorDescriptor::create()?),
            Tensor4DData::new(allocation),
        ))
    }

    /// Creates a new [Tensor4D] by allocating new device memory on the
    /// [Rc<CudaDevice>] **without** initializing it (this also creates a new
    /// [TensorDescriptor]).
    ///
    /// # Safety
    /// The data on this tensor should not be read until it is initialized.
    pub unsafe fn alloc_uninit(device: &Rc<CudaDevice>) -> CudaCudnnResult<Self> {
        Self::create_with(CudaRc {
            t_cuda: Rc::new(CudaUniquePtr::alloc(device)?),
            t_host: None,
        })
    }

    /// Creates a new [Tensor4D] by allocating new device memory on the
    /// [Rc<CudaDevice>] and initializing it with `value` (this also creates a
    /// new [TensorDescriptor]).
    pub fn alloc_with(
        device: &Rc<CudaDevice>,
        value: [[[[T; W]; H]; C]; N],
    ) -> CudaCudnnResult<Self> {
        Self::create_with(device.take(Rc::new(value))?)
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

    /// Scales all the data of this tensor with `value`.
    ///
    /// # See also
    /// <https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnScaleTensor>
    pub fn scale(&self, cudnn_handle: &CudnnHandle, factor: &T) -> CudaCudnnResult<()> {
        unsafe {
            cudnnScaleTensor(
                cudnn_handle.get_handle(),
                self.get_descriptor(),
                self.get_data_ptr_mut(),
                factor as *const _ as *const _,
            )
        }
        .result()
    }

    /// The size of this tensor is known at compile time.
    pub const fn size(&self) -> usize {
        size_of::<T>() * N * C * H * W
    }
}
impl<T: NumElements, const N: usize, const C: usize, const H: usize, const W: usize>
    Tensor4D<T, N, C, H, W>
{
    pub const fn get_numel(&self) -> u32 {
        Self::NUMEL as _
    }
}
impl<T: NumElements, const N: usize, const C: usize, const H: usize, const W: usize> NumElements
    for Tensor4D<T, N, C, H, W>
{
    type Dtype = T;

    const NUMEL: usize = T::NUMEL * N * C * H * W;
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn test_create_tensor() {
        let data = [[[[0.0, 1.0]]], [[[2.0, 3.0]]]];
        let t = Tensor4D::alloc_with(&CudaDeviceBuilder::new(0).build().unwrap(), data).unwrap();
        let on_gpu = *t.get_data().as_host().unwrap();
        assert_eq!(data, on_gpu);
    }
}
