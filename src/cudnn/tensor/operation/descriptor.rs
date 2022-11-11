use core::mem::MaybeUninit;

use crate::cudnn::sys::*;
use crate::prelude::*;

pub struct TensorOpsDescriptor(pub(crate) cudnnOpTensorDescriptor_t);
impl TensorOpsDescriptor {
    pub fn create() -> CudaCudnnResult<Self> {
        let mut descriptor = MaybeUninit::uninit();
        unsafe {
            cudnnCreateOpTensorDescriptor(descriptor.as_mut_ptr()).result()?;
            Ok(Self(descriptor.assume_init()))
        }
    }
}
impl Drop for TensorOpsDescriptor {
    fn drop(&mut self) {
        unsafe { cudnnDestroyOpTensorDescriptor(self.0) }
            .result()
            .unwrap();
    }
}
