mod descriptor;

pub use descriptor::*;

use core::ffi::c_void;

use alloc::rc::Rc;

use crate::cudarc::CudaUniquePtr;
use crate::cudnn::sys::*;
use crate::prelude::*;

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
    #[inline(always)]
    pub fn get_descriptor(&self) -> cudnnFilterDescriptor_t {
        self.descriptor.get_descriptor()
    }

    #[inline(always)]
    pub fn get_descriptor_rc(&self) -> Rc<FilterDescriptor<T, C_OUT, C_IN, H, W>> {
        Rc::clone(&self.descriptor)
    }

    #[inline(always)]
    pub fn get_data_ptr(&self) -> *const c_void {
        self.data.get_data_ptr()
    }

    #[inline(always)]
    pub fn get_data_ptr_mut(&self) -> *mut c_void {
        self.data.get_data_ptr_mut()
    }

    #[inline(always)]
    pub fn get_data(&self) -> Tensor4DData<T, C_OUT, C_IN, H, W> {
        self.data.clone()
    }

    #[inline(always)]
    pub fn as_split(
        &self,
    ) -> (
        Rc<FilterDescriptor<T, C_OUT, C_IN, H, W>>,
        Tensor4DData<T, C_OUT, C_IN, H, W>,
    ) {
        (self.get_descriptor_rc(), self.get_data())
    }

    pub fn create(allocation: CudaRc<[[[[T; W]; H]; C_IN]; C_OUT]>) -> CudaCudnnResult<Self> {
        Ok(Self {
            descriptor: Rc::new(FilterDescriptor::create()?),
            data: Tensor4DData::new(allocation),
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
        value: [[[[T; W]; H]; C_IN]; C_OUT],
    ) -> CudaCudnnResult<Self> {
        Self::create(device.take(Rc::new(value))?)
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
