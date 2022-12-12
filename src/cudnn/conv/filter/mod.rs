mod descriptor;

pub use descriptor::*;

use core::ffi::c_void;

use alloc::rc::Rc;

use crate::cudarc::CudaUniquePtr;
use crate::cudnn::sys::*;
use crate::prelude::*;

/// A convolution filter. Cloning this filter only clones the
/// pointer to the [FilterDescriptor] and [Tensor4DData] and thus increases the
/// reference count.
pub struct Filter<T, const C_OUT: usize, const C_IN: usize, const H: usize, const W: usize> {
    descriptor: Rc<FilterDescriptor<T, C_OUT, C_IN, H, W>>,
    data: Tensor4DData<T, C_OUT, C_IN, H, W>,
}
impl<T, const C_OUT: usize, const C_IN: usize, const H: usize, const W: usize> Clone
    for Filter<T, C_OUT, C_IN, H, W>
{
    fn clone(&self) -> Self {
        Self {
            descriptor: Rc::clone(&self.descriptor),
            data: self.data.clone(),
        }
    }
}
impl<T: TensorDataType, const C_OUT: usize, const C_IN: usize, const H: usize, const W: usize>
    Filter<T, C_OUT, C_IN, H, W>
where
    [(); W * H * C_IN * C_OUT]:,
{
    /// The pointer to the descriptor.
    #[inline(always)]
    pub fn get_descriptor(&self) -> cudnnFilterDescriptor_t {
        self.descriptor.get_descriptor()
    }

    /// Returns the [FilterDescriptor] after cloning it.
    #[inline(always)]
    pub fn get_descriptor_rc(&self) -> Rc<FilterDescriptor<T, C_OUT, C_IN, H, W>> {
        Rc::clone(&self.descriptor)
    }

    /// A pointer to the device memory.
    #[inline(always)]
    pub fn get_data_ptr(&self) -> *const c_void {
        self.data.get_data_ptr()
    }

    /// A mutable pointer to the device memory.
    #[inline(always)]
    pub fn get_data_ptr_mut(&mut self) -> *mut c_void {
        self.data.get_data_ptr_mut()
    }

    /// Returns the [Tensor4DData] after cloning it.
    #[inline(always)]
    pub fn get_data(&self) -> Tensor4DData<T, C_OUT, C_IN, H, W> {
        self.data.clone()
    }

    /// Returns a reference to the [Tensor4DData].
    #[inline(always)]
    pub fn get_data_ref(&self) -> &Tensor4DData<T, C_OUT, C_IN, H, W> {
        &self.data
    }

    /// Returns a mutable reference to the [Tensor4DData].
    #[inline(always)]
    pub fn get_data_ref_mut(&mut self) -> &mut Tensor4DData<T, C_OUT, C_IN, H, W> {
        &mut self.data
    }

    /// Split the [Filter] into a [FilterDescriptor] and a [Tensor4DData],
    /// after cloning them.
    #[inline(always)]
    pub fn as_split(
        &self,
    ) -> (
        Rc<FilterDescriptor<T, C_OUT, C_IN, H, W>>,
        Tensor4DData<T, C_OUT, C_IN, H, W>,
    ) {
        (self.get_descriptor_rc(), self.get_data())
    }

    /// Creates a new [Filter] by a [Rc<FilterDescriptor>] and a
    /// [Tensor4DData].
    pub fn new(
        descriptor: Rc<FilterDescriptor<T, C_OUT, C_IN, H, W>>,
        data: Tensor4DData<T, C_OUT, C_IN, H, W>,
    ) -> Self {
        Self { descriptor, data }
    }

    /// Creates a new [Filter] by a [Tensor4DData] (so this creates a new
    /// [FilterDescriptor]).
    pub fn create(allocation: CudaRc<[[[[T; W]; H]; C_IN]; C_OUT]>) -> CudaCudnnResult<Self> {
        Ok(Self::new(
            Rc::new(FilterDescriptor::create()?),
            Tensor4DData::new(allocation),
        ))
    }

    /// Creates a new [Filter] by allocating new device memory on the
    /// [Rc<CudaDevice>] initializing it with the value of `self.data`.
    ///
    /// This also uses the same descriptor (and thus increases the memory
    /// count).
    pub fn clone_into_new(&self, device: &Rc<CudaDevice>) -> CudaCudnnResult<Self> {
        Ok(Self::new(
            Rc::clone(&self.descriptor),
            self.data.clone_into_new(device)?,
        ))
    }

    /// Creates a new [Filter] by allocating new device memory on the
    /// [Rc<CudaDevice>] **without** initializing it (this also creates a new
    /// [FilterDescriptor]).
    ///
    /// # Safety
    /// The data on this filter should not be read until it is initialized.
    pub unsafe fn alloc_uninit(device: &Rc<CudaDevice>) -> CudaCudnnResult<Self> {
        Self::create(CudaRc {
            t_cuda: Rc::new(CudaUniquePtr::alloc(device)?),
            t_host: None,
        })
    }

    /// Creates a new [Filter] by allocating new device memory on the
    /// [Rc<CudaDevice>] and initializing it with `value` (this also creates a
    /// new [FiterDescriptor]).
    pub fn alloc_with(
        device: &Rc<CudaDevice>,
        value: [[[[T; W]; H]; C_IN]; C_OUT],
    ) -> CudaCudnnResult<Self> {
        Self::create(device.take(Rc::new(value))?)
    }

    pub fn as_tensor(&self) -> CudaCudnnResult<Tensor4D<T, C_OUT, C_IN, H, W>> {
        Tensor4D::create(self.data.clone())
    }
}

#[cfg(test)]
mod tests {
    use alloc::rc::Rc;

    use crate::prelude::*;

    #[test]
    fn test_create_filter() {
        let data = [[[[0.0, 1.0]]], [[[2.0, 3.0]]]];
        let f = Filter::create(
            CudaDeviceBuilder::new(0)
                .build()
                .unwrap()
                .take(Rc::new(data))
                .unwrap(),
        )
        .unwrap();
        let on_gpu = *f.get_data().as_host().unwrap();
        assert_eq!(data, on_gpu);
    }
}
