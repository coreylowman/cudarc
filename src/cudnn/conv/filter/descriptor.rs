use core::marker::PhantomData;
use core::mem::MaybeUninit;

use super::super::super::sys::*;
use crate::prelude::*;

pub struct FilterDescriptor<
    T,
    const C_OUT: usize,
    const C_IN: usize,
    const H: usize,
    const W: usize,
> {
    descriptor: cudnnFilterDescriptor_t,
    data_type:  PhantomData<T>,
}
impl<T: TensorDataType, const C_OUT: usize, const C_IN: usize, const H: usize, const W: usize>
    FilterDescriptor<T, C_OUT, C_IN, H, W>
{
    #[inline(always)]
    pub fn get_descriptor(&self) -> cudnnFilterDescriptor_t {
        self.descriptor
    }

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
                T::get_tensor_format(),
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
