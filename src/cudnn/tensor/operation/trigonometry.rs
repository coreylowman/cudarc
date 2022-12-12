use crate::prelude::*;

pub struct OperationSin;
impl_tensor_operation!(OperationSin: "sin");
pub struct OperationCos;
impl_tensor_operation!(OperationCos: "cos");

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    // #[test]
    // fn test_sinf32() {
    //     let device = CudaDeviceBuilder::new(0)
    //         .with_cudnn_modules()
    //         .build()
    //         .unwrap();

    //     let mut a = Tensor4D::alloc_with(&device, [[[[1.0f32, 2.0,
    // 3.0]]]]).unwrap();

    //     OperationSin
    //         .execute_in_place_with_scale(&device, &mut a, &2.0)
    //         .unwrap();

    //     let data = a.get_data().as_host().unwrap()[0][0][0];
    //     assert!((data[0] - 1.0f32.sin() * 2.0).abs() < 0.001);
    //     assert!((data[1] - 2.0f32.sin() * 2.0).abs() < 0.001);
    //     assert!((data[2] - 3.0f32.sin() * 2.0).abs() < 0.001);
    // }

    #[test]
    fn test_cosf64() {
        let device = CudaDeviceBuilder::new(0)
            .with_cudnn_modules()
            .build()
            .unwrap();

        let a = Tensor4D::alloc_with(&device, [[[[1.0f64, 2.0, 3.0]]]]).unwrap();
        let mut out = unsafe { Tensor4D::<_, 1, 1, 1, 3>::alloc_uninit(&device) }.unwrap();

        OperationCos.execute(&device, &a, &mut out).unwrap();

        let data = out.get_data().as_host().unwrap()[0][0][0];
        assert!((data[0] - 1.0f64.cos()).abs() < 0.001);
        assert!((data[1] - 2.0f64.cos()).abs() < 0.001);
        assert!((data[2] - 3.0f64.cos()).abs() < 0.001);
    }
}
