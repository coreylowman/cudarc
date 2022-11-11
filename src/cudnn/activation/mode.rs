use super::super::sys::*;

/// A Marker for an [ActivationMode].
///
/// # Supported modes
/// [Sigmoid], [Relu], [Tanh], [Elu], [Swish]
///
/// [Relu] has its upper bound set to `f64::MAX`.
///
/// Other modes are currently not supported as they require additional
/// parameters.
///
/// # See also
/// <https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnActivationMode_t>
pub trait ActivationMode {
    fn get_activation_mode() -> cudnnActivationMode_t;
    fn get_additional_parameter() -> f64;
}
macro_rules! impl_activation_mode {
    ($name:ident : $mode:ident) => {
        pub struct $name;
        impl ActivationMode for $name {
            fn get_activation_mode() -> cudnnActivationMode_t {
                cudnnActivationMode_t::$mode
            }

            fn get_additional_parameter() -> f64 {
                f64::MAX
            }
        }
    };
}

impl_activation_mode!(Sigmoid: CUDNN_ACTIVATION_SIGMOID);
impl_activation_mode!(Relu: CUDNN_ACTIVATION_RELU);
impl_activation_mode!(Tanh: CUDNN_ACTIVATION_TANH);
impl_activation_mode!(Elu: CUDNN_ACTIVATION_ELU);
