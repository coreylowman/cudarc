use crate::prelude::*;
use super::sys::*;

pub struct ConvolutionDescriptor(pub(crate) cudnnConvolutionDescriptor_t);
impl ConvolutionDescriptor {
    pub fn create() -> CudnnResult<Self> {
        let mut descriptor: Self = unsafe { std::mem::zeroed() };
        unsafe { cudnnCreateConvolutionDescriptor(&mut descriptor.0 as *mut _) }.result()?;
        Ok(descriptor)
    }
}
impl Drop for ConvolutionDescriptor {
    fn drop(&mut self) {
        unsafe { cudnnDestroyConvolutionDescriptor(self.0) }.result().unwrap();
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn test_create_descriptor() {
        let _descriptor = ConvolutionDescriptor::create().unwrap();
    }
}