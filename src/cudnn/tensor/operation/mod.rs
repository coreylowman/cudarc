mod descriptor;
mod division;
mod mode;

pub use descriptor::*;
pub use division::*;
pub use mode::*;

use core::marker::PhantomData;

use crate::cudnn::sys::*;
use crate::prelude::*;

const NAN_PROPAGATION: cudnnNanPropagation_t = cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN;

/// A [TensorOperation] for type `T` and mode `O`.
pub struct TensorOperation<T, O> {
    descriptor: TensorOperationDescriptor,
    op: PhantomData<O>,
    data_type: PhantomData<T>,
}
impl<T: TensorDataType, O: TensorOperationMode> TensorOperation<T, O> {
    /// Creates a new [TensorOperation] for type `T` and mode `O`.
    pub fn create() -> CudaCudnnResult<Self> {
        let descriptor = TensorOperationDescriptor::create()?;
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

    /// Executes on the tensors `a` and `b`, for [OperationSqrt] and
    /// [OperationNot] these have to be the same (not null!).
    ///
    /// # See also
    /// <https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnOpTensor>
    fn execute_op<const N: usize, const C: usize, const H: usize, const W: usize>(
        &self,
        cudnn_handle: &CudnnHandle,
        a: &Tensor4D<T, N, C, H, W>,
        b: &Tensor4D<T, N, C, H, W>,
        out: &mut Tensor4D<T, N, C, H, W>,
    ) -> CudaCudnnResult<()> {
        unsafe {
            cudnnOpTensor(
                cudnn_handle.get_handle(),
                self.descriptor.0,
                &T::ONE as *const _ as *const _,
                a.get_descriptor(),
                a.get_data_ptr(),
                &T::ONE as *const _ as *const _,
                b.get_descriptor(),
                b.get_data_ptr(),
                &T::ZERO as *const _ as *const _,
                out.get_descriptor(),
                out.get_data_ptr_mut(),
            )
        }
        .result()
    }

    fn execute_op_in_place<const N: usize, const C: usize, const H: usize, const W: usize>(
        &self,
        cudnn_handle: &CudnnHandle,
        a: &mut Tensor4D<T, N, C, H, W>,
        b: &Tensor4D<T, N, C, H, W>,
    ) -> CudaCudnnResult<()> {
        unsafe {
            cudnnOpTensor(
                cudnn_handle.get_handle(),
                self.descriptor.0,
                &T::ONE as *const _ as *const _,
                a.get_descriptor(),
                a.get_data_ptr(),
                &T::ONE as *const _ as *const _,
                b.get_descriptor(),
                b.get_data_ptr(),
                &T::ZERO as *const _ as *const _,
                a.get_descriptor(),
                a.get_data_ptr_mut(),
            )
        }
        .result()
    }
}
/// A trait for single parameter [TensorOperationMode]s ([CudnnOperationSqrt]
/// and [CudnnOperationNot]).
pub trait SingleParameterOp<T> {
    /// Executes the [TensorOperation] on the tensor `a`.
    fn execute<const N: usize, const C: usize, const H: usize, const W: usize>(
        &self,
        cudnn_handle: &CudnnHandle,
        a: &Tensor4D<T, N, C, H, W>,
        out: &mut Tensor4D<T, N, C, H, W>,
    ) -> CudaCudnnResult<()>;
    /// Executes the [TensorOperation] on the tensor `a` in place.
    fn execute_in_place<const N: usize, const C: usize, const H: usize, const W: usize>(
        &self,
        cudnn_handle: &CudnnHandle,
        a: &mut Tensor4D<T, N, C, H, W>,
    ) -> CudaCudnnResult<()>;
}
/// A trait for multi parameter [TensorOperationMode]s (all except
/// [CudnnOperationSqrt] and [CudnnOperationNot]).
pub trait MultiParameterOp<T> {
    /// Executes the [TensorOperation] on the tensor `a` and `b`.
    fn execute<const N: usize, const C: usize, const H: usize, const W: usize>(
        &self,
        cudnn_handle: &CudnnHandle,
        a: &Tensor4D<T, N, C, H, W>,
        b: &Tensor4D<T, N, C, H, W>,
        out: &mut Tensor4D<T, N, C, H, W>,
    ) -> CudaCudnnResult<()>;
    /// Executes the [TensorOperation] on the tensor `a` and `b` in place.
    fn execute_in_place<const N: usize, const C: usize, const H: usize, const W: usize>(
        &self,
        cudnn_handle: &CudnnHandle,
        a: &mut Tensor4D<T, N, C, H, W>,
        b: &Tensor4D<T, N, C, H, W>,
    ) -> CudaCudnnResult<()>;
}
/// implements single and multi parameter [TensorOperation]s
macro_rules! impl_tensor_op_execution {
    ($tensor_op:ty) => {
        impl<T: TensorDataType> MultiParameterOp<T> for TensorOperation<T, $tensor_op> {
            fn execute<const N: usize, const C: usize, const H: usize, const W: usize>(
                &self,
                cudnn_handle: &CudnnHandle,
                a: &Tensor4D<T, N, C, H, W>,
                b: &Tensor4D<T, N, C, H, W>,
                out: &mut Tensor4D<T, N, C, H, W>,
            ) -> CudaCudnnResult<()> {
                self.execute_op(cudnn_handle, a, b, out)
            }

            fn execute_in_place<const N: usize, const C: usize, const H: usize, const W: usize>(
                &self,
                cudnn_handle: &CudnnHandle,
                a: &mut Tensor4D<T, N, C, H, W>,
                b: &Tensor4D<T, N, C, H, W>,
            ) -> CudaCudnnResult<()> {
                self.execute_op_in_place(cudnn_handle, a, b)
            }
        }
    };
    (@no_second_param: $tensor_op:ty) => {
        impl<T: TensorDataType> SingleParameterOp<T> for TensorOperation<T, $tensor_op> {
            fn execute<const N: usize, const C: usize, const H: usize, const W: usize>(
                &self,
                cudnn_handle: &CudnnHandle,
                a: &Tensor4D<T, N, C, H, W>,
                out: &mut Tensor4D<T, N, C, H, W>,
            ) -> CudaCudnnResult<()> {
                self.execute_op(cudnn_handle, a, a, out)
            }

            fn execute_in_place<const N: usize, const C: usize, const H: usize, const W: usize>(
                &self,
                cudnn_handle: &CudnnHandle,
                a: &mut Tensor4D<T, N, C, H, W>,
            ) -> CudaCudnnResult<()> {
                self.execute_op_in_place(cudnn_handle, a, unsafe { &*(a as *const _) })
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

    fn get_information<O: TensorOperationMode>() -> (
        CudnnHandle,
        TensorOperation<f64, O>,
        Tensor4D<f64, 1, 1, 1, 6>,
        Tensor4D<f64, 1, 1, 1, 6>,
        Tensor4D<f64, 1, 1, 1, 6>,
    ) {
        let cuda = CudaDeviceBuilder::new(0).build().unwrap();
        let cudnn = CudnnHandle::create(&cuda).unwrap();
        let op = TensorOperation::create().unwrap();
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
        let output = out.get_data().as_host().unwrap();
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
        let output = out.get_data().as_host().unwrap();
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
        let output = out.get_data().as_host().unwrap();
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
        let output = out.get_data().as_host().unwrap();
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
        let output = out.get_data().as_host().unwrap();
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
        let output = out.get_data().as_host().unwrap();
        assert_eq!(output[0][0][0][0], 0.0);
        assert_eq!(output[0][0][0][1], -1.0);
        assert_eq!(output[0][0][0][2], 2.0);
        assert_eq!(output[0][0][0][3], 1.0);
        assert!(output[0][0][0][4].is_nan());
        assert_eq!(output[0][0][0][5], f64::INFINITY);
    }
}
