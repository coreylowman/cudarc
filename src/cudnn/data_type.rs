use super::sys::cudnnDataType_t;

/// Convert a Rust type to a [cudnnDataType_t].
/// # Supported rust-types
/// [f32] and [f64]
/// 
/// 
/// # Unsupported Types
/// [bool], [u8], [i8], [i32] and [i64], as they (somehow?) aren't supported in some tensor functions.
///
/// Other (cudnn) types ([core::simd::i8x4], [core::simd::u8x4],
/// [core::simd::i8x32], f16 and bf16) are unsupported as they require special
/// conditions/aren't availabe in Rust.
///
/// # See also
/// <https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnDataType_t>
pub trait TensorDataType {
    fn get_data_type() -> cudnnDataType_t;
}
macro_rules! impl_data_type {
    ($type:ty : $name:ident) => {
        impl TensorDataType for $type {
            fn get_data_type() -> cudnnDataType_t {
                cudnnDataType_t::$name
            }
        }
    };
}
impl_data_type!(f32: CUDNN_DATA_FLOAT);
impl_data_type!(f64: CUDNN_DATA_DOUBLE);

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