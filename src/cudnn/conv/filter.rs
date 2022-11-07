use core::mem::MaybeUninit;

use super::super::sys::*;
use crate::prelude::*;

pub struct FilterDescriptor(pub(crate) cudnnFilterDescriptor_t);
impl FilterDescriptor {
    pub fn create() -> CudnnResult<Self> {
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
    pub(crate) descriptor: FilterDescriptor,
    pub(crate) data: CudaRc<[[[[T; W]; H]; C_IN]; C_OUT]>,
}
impl<T: TensorDataType, const C_OUT: usize, const C_IN: usize, const H: usize, const W: usize>
    Filter<T, C_OUT, C_IN, H, W>
where
    [(); W * H * C_IN * C_OUT]:,
{
    pub fn create(allocation: CudaRc<[[[[T; W]; H]; C_IN]; C_OUT]>) -> CudnnResult<Self> {
        let descriptor = FilterDescriptor::create()?;
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
}

#[cfg(test)]
mod tests {
    use alloc::rc::Rc;

    use crate::cudnn::sys::*;
    use crate::prelude::*;

    #[test]
    fn test_create_descriptor() {
        let descriptor = Filter::<f64, 1, 2, 3, 4>::create(
            CudaDeviceBuilder::new(0)
                .build()
                .unwrap()
                .take(Rc::new([[[[1.0f64; 4]; 3]; 2]; 1]))
                .unwrap(),
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
