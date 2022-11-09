use crate::cudarc::ValidAsZeroBits;

use super::sys::{cudnnDataType_t, cudnnTensorFormat_t};

/// Convert a Rust type to a [cudnnDataType_t].
/// # Supported rust-types
/// [f32] and [f64]
///
///
/// # Unsupported Types
/// [bool], [u8], [i8], [i32] and [i64], as they (somehow?) aren't supported in
/// some tensor functions.
///
/// Other (cudnn) types ([core::simd::i8x4], [core::simd::u8x4],
/// [core::simd::i8x32], f16 and bf16) are unsupported as they require special
/// conditions/aren't availabe in Rust.
///
/// # See also
/// <https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnDataType_t>
pub trait TensorDataType: ValidAsZeroBits {
    const ZERO: Self;
    const ONE: Self;
    const MAX: Self;

    fn get_data_type() -> cudnnDataType_t;
    fn get_tensor_format() -> cudnnTensorFormat_t {
        cudnnTensorFormat_t::CUDNN_TENSOR_NCHW
    }
}
macro_rules! impl_data_type {
    ($type:ty : $name:ident, $zero:expr, $one:expr, $max:expr) => {
        impl TensorDataType for $type {
            const MAX: Self = $max;
            const ONE: Self = $one;
            const ZERO: Self = $zero;

            fn get_data_type() -> cudnnDataType_t {
                cudnnDataType_t::$name
            }
        }
    };
}
impl_data_type!(f32: CUDNN_DATA_FLOAT, 0.0f32, 1.0f32, f32::MAX);
impl_data_type!(f64: CUDNN_DATA_DOUBLE, 0.0f64, 1.0f64, f64::MAX);

#[cfg(test)]
mod tests {
    use super::super::sys::cudnnDataType_t;
    use super::TensorDataType;

    #[test]
    fn test() {
        assert_eq!(f32::get_data_type(), cudnnDataType_t::CUDNN_DATA_FLOAT);
        assert_eq!(f64::get_data_type(), cudnnDataType_t::CUDNN_DATA_DOUBLE);
    }
}
