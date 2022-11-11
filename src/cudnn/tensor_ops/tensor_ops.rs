use core::marker::PhantomData;

use super::super::sys::*;
use super::descriptor::*;
use crate::prelude::*;

const NAN_PROPAGATION: cudnnNanPropagation_t = cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN;

pub struct TensorOp<T, O> {
    descriptor: TensorOpsDescriptor,
    op: PhantomData<O>,
    data_type: PhantomData<T>,
}
impl<T: TensorDataType, O: TensorOperation> TensorOp<T, O> {
    pub fn create() -> CudaCudnnResult<Self> {
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
}
pub trait SingleParameterOp<T> {
    fn execute<const N: usize, const C: usize, const H: usize, const W: usize>(
        &self,
        cudnn_handle: &CudnnHandle,
        a: &Tensor4D<T, N, C, H, W>,
        out: &mut Tensor4D<T, N, C, H, W>,
    ) -> CudaCudnnResult<()>;
}
pub trait MultiParameterOp<T> {
    fn execute<const N: usize, const C: usize, const H: usize, const W: usize>(
        &self,
        cudnn_handle: &CudnnHandle,
        a: &Tensor4D<T, N, C, H, W>,
        b: &Tensor4D<T, N, C, H, W>,
        out: &mut Tensor4D<T, N, C, H, W>,
    ) -> CudaCudnnResult<()>;
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
            ) -> CudaCudnnResult<()> {
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
            ) -> CudaCudnnResult<()> {
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
