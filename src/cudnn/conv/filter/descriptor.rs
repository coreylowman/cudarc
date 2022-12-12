use core::marker::PhantomData;
use core::mem::MaybeUninit;

use super::super::super::sys::*;
use crate::prelude::*;

/// A descriptor for a convolution filter. It is destroyed when it is dropped.
///
/// # See also
/// <https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnFilterDescriptor_t>
/// <https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnDestroyFilterDescriptor>
pub struct FilterDescriptor<
    T,
    const C_OUT: usize,
    const C_IN: usize,
    const H: usize,
    const W: usize,
> {
    descriptor: cudnnFilterDescriptor_t,
    data_type: PhantomData<T>,
}
impl<T: TensorDataType, const C_OUT: usize, const C_IN: usize, const H: usize, const W: usize>
    FilterDescriptor<T, C_OUT, C_IN, H, W>
{
    /// The pointer to the descriptor.
    #[inline(always)]
    pub fn get_descriptor(&self) -> cudnnFilterDescriptor_t {
        self.descriptor
    }

    /// Creates a new [FilterDescriptor] with the data type `T`.
    ///
    /// # See also
    /// <https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnCreateFilterDescriptor>
    /// <https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetFilterDescriptor>
    pub fn create() -> CudaCudnnResult<Self> {
        let descriptor = unsafe {
            let mut descriptor = MaybeUninit::uninit();
            cudnnCreateFilterDescriptor(descriptor.as_mut_ptr()).result()?;
            descriptor.assume_init()
        };
        unsafe {
            cudnnSetFilter4dDescriptor(
                descriptor,
                T::get_data_type(),
                TENSOR_FORMAT,
                C_OUT as _,
                C_IN as _,
                H as _,
                W as _,
            )
            .result()?;
        }
        Ok(Self {
            descriptor,
            data_type: PhantomData,
        })
    }
}
impl<T, const C_OUT: usize, const C_IN: usize, const H: usize, const W: usize> Drop
    for FilterDescriptor<T, C_OUT, C_IN, H, W>
{
    fn drop(&mut self) {
        unsafe { cudnnDestroyFilterDescriptor(self.descriptor) }
            .result()
            .unwrap();
    }
}
