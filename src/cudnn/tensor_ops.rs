use core::marker::PhantomData;
use core::mem::MaybeUninit;

use super::sys::*;
use crate::prelude::*;

pub trait TensorOperation {
    fn get_tensor_operation() -> cudnnOpTensorOp_t;
}
macro_rules! impl_tensor_operation {
    ($type:ident : $name:ident) => {
        pub struct $type;
        impl TensorOperation for $type {
            fn get_tensor_operation() -> cudnnOpTensorOp_t {
                cudnnOpTensorOp_t::$name
            }
        }
    };
}
impl_tensor_operation!(OperationAdd: CUDNN_OP_TENSOR_ADD);
impl_tensor_operation!(OperationMul: CUDNN_OP_TENSOR_MUL);
impl_tensor_operation!(OperationMin: CUDNN_OP_TENSOR_MIN);
impl_tensor_operation!(OperationMax: CUDNN_OP_TENSOR_MAX);
impl_tensor_operation!(OperationSqrt: CUDNN_OP_TENSOR_SQRT);
impl_tensor_operation!(OperationNot: CUDNN_OP_TENSOR_NOT);

const NAN_PROPAGATION: cudnnNanPropagation_t = cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN;

pub struct TensorOpsDescriptor(pub(crate) cudnnOpTensorDescriptor_t);
impl TensorOpsDescriptor {
    pub fn create() -> CudnnResult<Self> {
        let mut descriptor = MaybeUninit::uninit();
        unsafe {
            cudnnCreateOpTensorDescriptor(descriptor.as_mut_ptr()).result()?;
            Ok(Self(descriptor.assume_init()))
        }
    }
}
impl Drop for TensorOpsDescriptor {
    fn drop(&mut self) {
        unsafe { cudnnDestroyOpTensorDescriptor(self.0) }
            .result()
            .unwrap();
    }
}

pub struct TensorOp<T, O> {
    descriptor: TensorOpsDescriptor,
    op: PhantomData<O>,
    data_type: PhantomData<T>,
}
impl<T: TensorDataType, O: TensorOperation> TensorOp<T, O> {
    pub fn create() -> CudnnResult<Self> {
        let descriptor = TensorOpsDescriptor::create()?;
        unsafe {
            cudnnSetOpTensorDescriptor(
                descriptor.0,
                O::get_tensor_operation(),
                T::get_data_type(),
                NAN_PROPAGATION,
            )
        }
        .result()?;
        Ok(Self {
            descriptor,
            op: PhantomData,
            data_type: PhantomData,
        })
    }

    fn execute_op<const N: usize, const C: usize, const H: usize, const W: usize>(
        &self,
        cudnn_handle: &CudnnHandle,
        a: &Tensor4D<T, N, C, H, W>,
        b: &Tensor4D<T, N, C, H, W>,
        out: &mut Tensor4D<T, N, C, H, W>,
    ) -> CudnnResult<()> {
        unsafe {
            cudnnOpTensor(
                cudnn_handle.0,
                self.descriptor.0,
                &T::ONE as *const _ as *const _,
                a.descriptor.0,
                a.data.t_cuda.cu_device_ptr as *const _,
                &T::ONE as *const _ as *const _,
                b.descriptor.0,
                b.data.t_cuda.cu_device_ptr as *const _,
                &T::ZERO as *const _ as *const _,
                out.descriptor.0,
                out.data.t_cuda.cu_device_ptr as *mut _,
            )
        }
        .result()
    }
}
pub trait SingleParameterOp<T> {
    fn execute<const N: usize, const C: usize, const H: usize, const W: usize>(
        &self,
        cudnn_handle: &CudnnHandle,
        a: &Tensor4D<T, N, C, H, W>,
        out: &mut Tensor4D<T, N, C, H, W>,
    ) -> CudnnResult<()>;
}
pub trait MultiParameterOp<T> {
    fn execute<const N: usize, const C: usize, const H: usize, const W: usize>(
        &self,
        cudnn_handle: &CudnnHandle,
        a: &Tensor4D<T, N, C, H, W>,
        b: &Tensor4D<T, N, C, H, W>,
        out: &mut Tensor4D<T, N, C, H, W>,
    ) -> CudnnResult<()>;
}
macro_rules! impl_tensor_op_execution {
    ($tensor_op:ty) => {
        impl<T: TensorDataType> MultiParameterOp<T> for TensorOp<T, $tensor_op> {
            fn execute<const N: usize, const C: usize, const H: usize, const W: usize>(
                &self,
                cudnn_handle: &CudnnHandle,
                a: &Tensor4D<T, N, C, H, W>,
                b: &Tensor4D<T, N, C, H, W>,
                out: &mut Tensor4D<T, N, C, H, W>,
            ) -> CudnnResult<()> {
                self.execute_op(cudnn_handle, a, b, out)
            }
        }
    };
    (@no_second_param: $tensor_op:ty) => {
        impl<T: TensorDataType> SingleParameterOp<T> for TensorOp<T, $tensor_op> {
            fn execute<const N: usize, const C: usize, const H: usize, const W: usize>(
                &self,
                cudnn_handle: &CudnnHandle,
                a: &Tensor4D<T, N, C, H, W>,
                out: &mut Tensor4D<T, N, C, H, W>,
            ) -> CudnnResult<()> {
                self.execute_op(cudnn_handle, a, a, out)
            }
        }
    };
}
impl_tensor_op_execution!(OperationAdd);
impl_tensor_op_execution!(OperationMul);
impl_tensor_op_execution!(OperationMin);
impl_tensor_op_execution!(OperationMax);
impl_tensor_op_execution!(@no_second_param: OperationSqrt);
impl_tensor_op_execution!(@no_second_param: OperationNot);

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
        let output = out.data.into_host().unwrap();
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
        let output = out.data.into_host().unwrap();
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
        let output = out.data.into_host().unwrap();
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
        let output = out.data.into_host().unwrap();
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
        let output = out.data.into_host().unwrap();
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
        let output = out.data.into_host().unwrap();
        // TODO check if this is really right
        assert_eq!(output[0][0][0][0], 0.0);
        assert_eq!(output[0][0][0][1], -1.0);
        assert_eq!(output[0][0][0][2], 2.0);
        assert_eq!(output[0][0][0][3], 1.0);
        assert!(output[0][0][0][4].is_nan());
        assert_eq!(output[0][0][0][5], f64::INFINITY);
    }
}
