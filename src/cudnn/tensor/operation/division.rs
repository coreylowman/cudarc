use crate::prelude::*;

pub struct OperationDiv;
impl_tensor_operation!(@multi_parameter OperationDiv: "division");

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
            .execute_with_scale(&device, &a, &1.0, &b, &2.0, &mut out)
            .unwrap();

        let data = out.get_data().as_host().unwrap()[0][0][0];
        assert!((data[0] - 1.0 / 6.0).abs() < 0.001);
        assert!((data[1] - 1.0).abs() < 0.001);
        assert!((data[2] - 1.0).abs() < 0.001);
        assert!((data[3] - 1.0).abs() < 0.001);
        assert!(data[4].is_nan());
    }

    #[test]
    fn test_f64() {
        let device = CudaDeviceBuilder::new(0)
            .with_cudnn_modules()
            .build()
            .unwrap();

        let mut a = Tensor4D::alloc_with(&device, [[[[1.0, 2.0, 3.0, 4.0, f64::NAN]]]]).unwrap();
        let b = Tensor4D::alloc_with(&device, [[[[3.0, 1.0, 1.5, 2.0, 0.0]]]]).unwrap();

        OperationDiv.execute_in_place(&device, &mut a, &b).unwrap();

        let data = a.get_data().as_host().unwrap()[0][0][0];
        assert!((data[0] - 1.0 / 3.0).abs() < f64::EPSILON);
        assert!((data[1] - 2.0).abs() < f64::EPSILON);
        assert!((data[2] - 2.0).abs() < f64::EPSILON);
        assert!((data[3] - 2.0).abs() < f64::EPSILON);
        assert!(data[4].is_nan());
    }
}
