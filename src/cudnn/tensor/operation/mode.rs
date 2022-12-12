use crate::cudnn::sys::*;

/// A mode for a tensor operation. Currently supported modes are:
///     - [OperationAdd]:  Elementwise addition.
///     - [OperationMul]:  Elementwise multiplication.
///     - [OperationMin]:  Elementwise minimum.
///     - [OperationMax]:  Elementwise maximum.
///     - [OperationSqrt]: Elementwise square root; this only uses one tensor.
///     - [OperationNot]:  Elementwise y = (1 - x); this only uses one tensor.
pub trait TensorOperationMode {
    fn get_tensor_operation() -> cudnnOpTensorOp_t;
}
macro_rules! impl_tensor_operation {
    ($type:ident : $name:ident) => {
        pub struct $type;
        impl TensorOperationMode for $type {
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
