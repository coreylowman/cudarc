use crate::prelude::*;

pub struct OperationRecip;
impl_tensor_operation!(OperationRecip: "recip");

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
        let mut out = unsafe { Tensor4D::<_, 1, 1, 1, 5>::alloc_uninit(&device) }.unwrap();

        OperationRecip
            .execute_with_scale(&device, &a, &1.0, &mut out)
            .unwrap();

        let data = out.get_data().as_host().unwrap()[0][0][0];
        assert!((data[0] - 1.0f32.recip()).abs() < 0.001);
        assert!((data[1] - 2.0f32.recip()).abs() < 0.001);
        assert!((data[2] - 3.0f32.recip()).abs() < 0.001);
        assert!((data[3] - 4.0f32.recip()).abs() < 0.001);
        assert!(data[4].is_nan());
    }

    #[test]
    fn test_f64() {
        let device = CudaDeviceBuilder::new(0)
            .with_cudnn_modules()
            .build()
            .unwrap();

        let mut a = Tensor4D::alloc_with(&device, [[[[1.0, 2.0, 3.0, 4.0, f64::NAN]]]]).unwrap();

        OperationRecip.execute_in_place(&device, &mut a).unwrap();

        let data = a.get_data().as_host().unwrap()[0][0][0];
        assert!((data[0] - 1.0f64.recip()).abs() < f64::EPSILON);
        assert!((data[1] - 2.0f64.recip()).abs() < f64::EPSILON);
        assert!((data[2] - 3.0f64.recip()).abs() < f64::EPSILON);
        assert!((data[3] - 4.0f64.recip()).abs() < f64::EPSILON);
        assert!(data[4].is_nan());
    }

    #[test]
    fn test_big_tensor() {
        let device = CudaDeviceBuilder::new(0)
            .with_cudnn_modules()
            .build()
            .unwrap();

        type InnerData = [f64; 5000];
        let mut a = Tensor4D::alloc_with(
            &device,
            [[[
                InnerData::try_from((0..5000).map(|n| n as f64).collect::<std::vec::Vec<_>>())
                    .unwrap(),
            ]]],
        )
        .unwrap();

        OperationRecip.execute_in_place(&device, &mut a).unwrap();

        let expected: InnerData = (0..5000)
            .map(|n| (n as f64).recip())
            .collect::<std::vec::Vec<_>>()
            .try_into()
            .unwrap();
        assert_eq!(a.get_data().as_host().unwrap()[0][0][0], expected);
    }
}
