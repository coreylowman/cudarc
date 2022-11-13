use core::ops::Div;

use alloc::rc::Rc;

use crate::prelude::*;

pub trait DivisionOperand:
    TensorDataType + Div<Output = Self> + Copy + PartialEq + NumElements
where
    for<'a> &'a Self: IntoKernelParam,
{
    const FUNCTION_NAME_NO_SCALE: &'static str;
    const FUNCTION_NAME_WITH_SCALE: &'static str;
}
macro_rules! impl_div_op {
    ($type:ident) => {
        impl DivisionOperand for $type {
            const FUNCTION_NAME_NO_SCALE: &'static str = concat!("division_", stringify!($type));
            const FUNCTION_NAME_WITH_SCALE: &'static str =
                concat!("division_with_scale_", stringify!($type));
        }
    };
}
impl_div_op!(f32);
impl_div_op!(f64);

pub struct OperationDiv;
impl OperationDiv {
    pub fn divide_with_scale<
        T: DivisionOperand,
        const N: usize,
        const C: usize,
        const H: usize,
        const W: usize,
    >(
        &self,
        device: &Rc<CudaDevice>,
        a: &Tensor4DData<T, N, C, H, W>,
        a_scale: &T,
        b: &Tensor4DData<T, N, C, H, W>,
        b_scale: &T,
        out: &mut Tensor4DData<T, N, C, H, W>,
    ) -> CudaCudnnResult<()>
    where
        for<'a> &'a T: IntoKernelParam,
    {
        let numel = out.get_numel();
        let f = *a_scale / *b_scale;
        if f == T::ONE {
            return self.divide(device, a, b, out);
        }
        // TODO use directly `T` as a parameter
        let factor = device.take(Rc::new(f))?;
        unsafe {
            device.launch_cuda_function(
                device
                    .get_module(CUSTOM_KERNEL_MODULE)
                    .and_then(|m| m.get_fn(T::FUNCTION_NAME_WITH_SCALE))
                    .ok_or(CudaCudnnError::CudaError(CudaError(
                        crate::driver::sys::CUresult::CUDA_ERROR_NOT_FOUND,
                    )))?,
                LaunchConfig::for_num_elems(numel),
                (out, a, b, &factor, &numel),
            )
        }
        .into_cuda_cudnn_result()
    }

    pub fn divide<
        T: DivisionOperand,
        const N: usize,
        const C: usize,
        const H: usize,
        const W: usize,
    >(
        &self,
        device: &Rc<CudaDevice>,
        a: &Tensor4DData<T, N, C, H, W>,
        b: &Tensor4DData<T, N, C, H, W>,
        out: &mut Tensor4DData<T, N, C, H, W>,
    ) -> CudaCudnnResult<()>
    where
        for<'a> &'a T: IntoKernelParam,
    {
        let numel = out.get_numel();
        unsafe {
            device.launch_cuda_function(
                device
                    .get_module(CUSTOM_KERNEL_MODULE)
                    .and_then(|m| m.get_fn(T::FUNCTION_NAME_NO_SCALE))
                    .ok_or(CudaCudnnError::CudaError(CudaError(
                        crate::driver::sys::CUresult::CUDA_ERROR_NOT_FOUND,
                    )))?,
                LaunchConfig::for_num_elems(numel),
                (out, a, b, &numel),
            )
        }
        .into_cuda_cudnn_result()
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn test_f32() {
        let device = CudaDeviceBuilder::new(0)
            .with_cudnn_modules()
            .build()
            .unwrap();

        let a = Tensor4D::alloc_with(&device, [[[[1.0, 2.0, 3.0, 4.0, f32::NAN]]]]).unwrap();
        let b = Tensor4D::alloc_with(&device, [[[[3.0, 1.0, 1.5, 2.0, 0.0]]]]).unwrap();
        let mut out = unsafe { Tensor4D::alloc_uninit(&device) }.unwrap();

        OperationDiv
            .divide_with_scale(
                &device,
                a.get_data_ref(),
                &1.0,
                b.get_data_ref(),
                &1.0,
                out.get_data_ref_mut(),
            )
            .unwrap();

        let data = out.get_data().as_host().unwrap()[0][0][0];
        assert!((data[0] - 1.0 / 3.0).abs() < f32::EPSILON);
        assert!((data[1] - 2.0).abs() < f32::EPSILON);
        assert!((data[2] - 2.0).abs() < f32::EPSILON);
        assert!((data[3] - 2.0).abs() < f32::EPSILON);
        assert!(data[4].is_nan());
    }

    #[test]
    fn test_f64() {
        let device = CudaDeviceBuilder::new(0)
            .with_cudnn_modules()
            .build()
            .unwrap();

        let a = Tensor4D::alloc_with(&device, [[[[1.0, 2.0, 3.0, 4.0, f64::NAN]]]]).unwrap();
        let b = Tensor4D::alloc_with(&device, [[[[3.0, 1.0, 1.5, 2.0, 0.0]]]]).unwrap();
        let mut out = unsafe { Tensor4D::alloc_uninit(&device) }.unwrap();

        OperationDiv
            .divide(
                &device,
                a.get_data_ref(),
                b.get_data_ref(),
                out.get_data_ref_mut(),
            )
            .unwrap();

        let data = out.get_data().as_host().unwrap()[0][0][0];
        assert!((data[0] - 1.0 / 3.0).abs() < f64::EPSILON);
        assert!((data[1] - 2.0).abs() < f64::EPSILON);
        assert!((data[2] - 2.0).abs() < f64::EPSILON);
        assert!((data[3] - 2.0).abs() < f64::EPSILON);
        assert!(data[4].is_nan());
    }
}
