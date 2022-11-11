use super::super::sys::*;

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
