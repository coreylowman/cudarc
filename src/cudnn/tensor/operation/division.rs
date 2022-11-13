use alloc::rc::Rc;

use crate::prelude::*;

pub trait DivisionOperand: TensorDataType + NumElements {
    const NAME: &'static str;
}
impl DivisionOperand for f32 {
    const NAME: &'static str = "division_f32";
}
impl DivisionOperand for f64 {
    const NAME: &'static str = "division_f64";
}

pub struct OperationDiv;
impl OperationDiv {
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
    ) -> CudaCudnnResult<()> {
        let numel = out.get_numel();
        unsafe {
            device.launch_cuda_function(
                device
                    .get_module(CUSTOM_KERNEL_MODULE)
                    .and_then(|m| m.get_fn(T::NAME))
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
            .divide(&device, a.as_data(), b.as_data(), out.as_data_mut())
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
            .divide(&device, a.as_data(), b.as_data(), out.as_data_mut())
            .unwrap();

        let data = out.get_data().as_host().unwrap()[0][0][0];
        assert!((data[0] - 1.0 / 3.0).abs() < f64::EPSILON);
        assert!((data[1] - 2.0).abs() < f64::EPSILON);
        assert!((data[2] - 2.0).abs() < f64::EPSILON);
        assert!((data[3] - 2.0).abs() < f64::EPSILON);
        assert!(data[4].is_nan());
    }
}
