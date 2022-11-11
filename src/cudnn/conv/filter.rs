use core::mem::MaybeUninit;

use alloc::rc::Rc;

use super::super::sys::*;
use crate::cudarc::CudaUniquePtr;
use crate::prelude::*;

pub struct FilterDescriptor(pub(crate) cudnnFilterDescriptor_t);
impl FilterDescriptor {
    pub fn create() -> CudaCudnnResult<Self> {
        let mut descriptor = MaybeUninit::uninit();
        unsafe {
            cudnnCreateFilterDescriptor(descriptor.as_mut_ptr()).result()?;
            Ok(Self(descriptor.assume_init()))
        }
    }
}
impl Drop for FilterDescriptor {
    fn drop(&mut self) {
        unsafe { cudnnDestroyFilterDescriptor(self.0) }
            .result()
            .unwrap();
    }
}
pub struct Filter<T, const C_OUT: usize, const C_IN: usize, const H: usize, const W: usize> {
    pub(crate) descriptor: Rc<FilterDescriptor>,
    pub(crate) data: CudaRc<[[[[T; W]; H]; C_IN]; C_OUT]>,
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
    pub fn create(allocation: CudaRc<[[[[T; W]; H]; C_IN]; C_OUT]>) -> CudaCudnnResult<Self> {
        let descriptor = Rc::new(FilterDescriptor::create()?);
        unsafe {
            cudnnSetFilter4dDescriptor(
                descriptor.0,
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
            data: allocation,
        })
    }

    pub unsafe fn alloc_uninit(device: &Rc<CudaDevice>) -> CudaCudnnResult<Self> {
        Self::create(CudaRc {
            t_cuda: Rc::new(CudaUniquePtr::alloc(device).into_cuda_cudnn_result()?),
            t_host: None,
        })
    }

    pub fn alloc_with(
        device: &Rc<CudaDevice>,
        value: [[[[T; W]; H]; C_IN]; C_OUT],
    ) -> CudaCudnnResult<Self> {
        Self::create(device.take(Rc::new(value)).into_cuda_cudnn_result()?)
    }
}

#[cfg(test)]
mod tests {
    use alloc::rc::Rc;

    use crate::cudnn::sys::*;
    use crate::prelude::*;

    #[test]
    fn test_create_descriptor() {
        let descriptor = Filter::<f64, 1, 2, 3, 4>::alloc_with(
            &CudaDeviceBuilder::new(0).build().unwrap(),
            [[[[1.0f64; 4]; 3]; 2]; 1],
        )
        .unwrap();
        unsafe {
            let mut data_type: cudnnDataType_t = std::mem::zeroed();
            let mut data_format: cudnnTensorFormat_t = std::mem::zeroed();
            let mut k: i32 = std::mem::zeroed();
            let mut c: i32 = std::mem::zeroed();
            let mut h: i32 = std::mem::zeroed();
            let mut w: i32 = std::mem::zeroed();
            cudnnGetFilter4dDescriptor(
                descriptor.descriptor.0,
                &mut data_type,
                &mut data_format,
                &mut k,
                &mut c,
                &mut h,
                &mut w,
            )
            .result()
            .unwrap();
            assert_eq!(data_type, cudnnDataType_t::CUDNN_DATA_DOUBLE);
            assert_eq!(data_format, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW);
            assert_eq!(k, 1);
            assert_eq!(c, 2);
            assert_eq!(h, 3);
            assert_eq!(w, 4);
        }
    }

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
        let on_gpu = *f.data.sync_release().unwrap().unwrap();
        assert_eq!(data, on_gpu);
    }
}
