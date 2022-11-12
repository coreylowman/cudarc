use super::super::sys::*;
use crate::prelude::*;
use core::marker::PhantomData;
use core::mem::MaybeUninit;

/// A descriptor of a tensor. This can be reused as all tensors are fully packed
/// and in NCHW layout.
///
/// This descriptor is destroyed when it is dropped.
///
/// # See also
/// <https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnTensorDescriptor_t>
/// <https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnDestroyTensorDescriptor>
pub struct TensorDescriptor<T, const N: usize, const C: usize, const H: usize, const W: usize> {
    descriptor: cudnnTensorDescriptor_t,
    data_type:  PhantomData<T>,
}
impl<T: TensorDataType, const N: usize, const C: usize, const H: usize, const W: usize>
    TensorDescriptor<T, N, C, H, W>
{
    /// The pointer to the descriptor.
    #[inline(always)]
    pub fn get_descriptor(&self) -> cudnnTensorDescriptor_t {
        self.descriptor
    }

    /// Creates a new [TensorDescriptor] and sets it to `NCHW layout` with the
    /// data type `T`.
    ///
    /// # See also
    /// <https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnCreateTensorDescriptor>
    /// <https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetTensor4dDescriptor>
    pub fn create() -> CudaCudnnResult<Self> {
        let descriptor = unsafe {
            let mut descriptor = MaybeUninit::uninit();
            cudnnCreateTensorDescriptor(descriptor.as_mut_ptr()).result()?;
            descriptor.assume_init()
        };
        unsafe {
            cudnnSetTensor4dDescriptor(
                descriptor,
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
impl<T, const N: usize, const C: usize, const H: usize, const W: usize> Drop
    for TensorDescriptor<T, N, C, H, W>
{
    fn drop(&mut self) {
        unsafe { cudnnDestroyTensorDescriptor(self.descriptor) }
            .result()
            .unwrap();
    }
}
