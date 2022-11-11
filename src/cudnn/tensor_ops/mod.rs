mod descriptor;
mod operation;
mod tensor_ops;
pub use operation::*;
pub use tensor_ops::*;

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    fn get_information<O: TensorOperation>() -> (
        CudnnHandle,
        TensorOp<f64, O>,
        Tensor4D<f64, 1, 1, 1, 6>,
        Tensor4D<f64, 1, 1, 1, 6>,
        Tensor4D<f64, 1, 1, 1, 6>,
    ) {
        let cuda = CudaDeviceBuilder::new(0).build().unwrap();
        let cudnn = CudnnHandle::create(&cuda).unwrap();
        let op = TensorOp::create().unwrap();
        let a = Tensor4D::alloc_with(&cuda, [[[[
            1.0,
            2.0,
            -1.0,
            0.0,
            f64::NAN,
            f64::NEG_INFINITY,
        ]]]])
        .unwrap();
        let b =
            Tensor4D::alloc_with(&cuda, [[[[3.0, 0.0, -2.0, f64::INFINITY, 0.0, 0.4]]]]).unwrap();
        let out = unsafe { Tensor4D::alloc_uninit(&cuda) }.unwrap();
        (cudnn, op, a, b, out)
    }

    #[test]
    fn test_add() {
        let (cudnn, op, a, b, mut out) = get_information::<OperationAdd>();
        op.execute(&cudnn, &a, &b, &mut out).unwrap();
        let output = out.get_data().unwrap();
        assert_eq!(output[0][0][0][0], 4.0);
        assert_eq!(output[0][0][0][1], 2.0);
        assert_eq!(output[0][0][0][2], -3.0);
        assert_eq!(output[0][0][0][3], f64::INFINITY);
        assert!(output[0][0][0][4].is_nan());
        assert_eq!(output[0][0][0][5], f64::NEG_INFINITY);
    }

    #[test]
    fn test_mul() {
        let (cudnn, op, a, b, mut out) = get_information::<OperationMul>();
        op.execute(&cudnn, &a, &b, &mut out).unwrap();
        let output = out.get_data().unwrap();
        assert_eq!(output[0][0][0][0], 3.0);
        assert_eq!(output[0][0][0][1], 0.0);
        assert_eq!(output[0][0][0][2], 2.0);
        assert!(output[0][0][0][3].is_nan());
        assert!(output[0][0][0][4].is_nan());
        assert_eq!(output[0][0][0][5], f64::NEG_INFINITY);
    }

    #[test]
    fn test_min() {
        let (cudnn, op, a, b, mut out) = get_information::<OperationMin>();
        op.execute(&cudnn, &a, &b, &mut out).unwrap();
        let output = out.get_data().unwrap();
        assert_eq!(output[0][0][0][0], 1.0);
        assert_eq!(output[0][0][0][1], 0.0);
        assert_eq!(output[0][0][0][2], -2.0);
        assert_eq!(output[0][0][0][3], 0.0);
        assert!(output[0][0][0][4].is_nan());
        assert_eq!(output[0][0][0][5], f64::NEG_INFINITY);
    }

    #[test]
    fn test_max() {
        let (cudnn, op, a, b, mut out) = get_information::<OperationMax>();
        op.execute(&cudnn, &a, &b, &mut out).unwrap();
        let output = out.get_data().unwrap();
        assert_eq!(output[0][0][0][0], 3.0);
        assert_eq!(output[0][0][0][1], 2.0);
        assert_eq!(output[0][0][0][2], -1.0);
        assert_eq!(output[0][0][0][3], f64::INFINITY);
        assert!(output[0][0][0][4].is_nan());
        assert_eq!(output[0][0][0][5], 0.4);
    }

    #[test]
    fn test_sqrt() {
        let (cudnn, op, a, _, mut out) = get_information::<OperationSqrt>();
        op.execute(&cudnn, &a, &mut out).unwrap();
        let output = out.get_data().unwrap();
        assert_eq!(output[0][0][0][0], 1.0);
        assert_eq!(output[0][0][0][1], 2.0f64.sqrt());
        assert!(output[0][0][0][2].is_nan());
        assert_eq!(output[0][0][0][3], 0.0);
        assert!(output[0][0][0][4].is_nan());
        assert!(output[0][0][0][5].is_nan());
    }

    #[test]
    fn test_not() {
        let (cudnn, op, a, _, mut out) = get_information::<OperationNot>();
        op.execute(&cudnn, &a, &mut out).unwrap();
        let output = out.get_data().unwrap();
        // TODO check if this is really right
        assert_eq!(output[0][0][0][0], 0.0);
        assert_eq!(output[0][0][0][1], -1.0);
        assert_eq!(output[0][0][0][2], 2.0);
        assert_eq!(output[0][0][0][3], 1.0);
        assert!(output[0][0][0][4].is_nan());
        assert_eq!(output[0][0][0][5], f64::INFINITY);
    }
}
