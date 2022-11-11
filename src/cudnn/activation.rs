use core::marker::PhantomData;
use core::mem::MaybeUninit;

use super::sys::*;
use crate::prelude::*;

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

const NAN_PROPAGATION: cudnnNanPropagation_t = cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN;
pub struct Activation<A> {
    descriptor:      cudnnActivationDescriptor_t,
    activation_mode: PhantomData<A>,
}
impl<A: ActivationMode> Activation<A> {
    pub fn create() -> CudaCudnnResult<Self> {
        let descriptor = unsafe {
            let mut descriptor = MaybeUninit::uninit();
            cudnnCreateActivationDescriptor(descriptor.as_mut_ptr()).result()?;
            descriptor.assume_init()
        };
        unsafe {
            cudnnSetActivationDescriptor(
                descriptor,
                A::get_activation_mode(),
                NAN_PROPAGATION,
                A::get_additional_parameter(),
            )
        }
        .result()?;
        Ok(Self {
            descriptor,
            activation_mode: PhantomData,
        })
    }

    pub fn forward<
        T: TensorDataType,
        const N: usize,
        const C: usize,
        const H: usize,
        const W: usize,
    >(
        &self,
        cudnn_handle: &CudnnHandle,
        input: &Tensor4D<T, N, C, H, W>,
        output: &mut Tensor4D<T, N, C, H, W>,
    ) -> CudaCudnnResult<()> {
        unsafe {
            cudnnActivationForward(
                cudnn_handle.get_handle(),
                self.descriptor,
                &T::ONE as *const _ as *const _,
                input.get_descriptor(),
                input.get_data_ptr(),
                &T::ZERO as *const _ as *const _,
                output.get_descriptor(),
                output.get_data_ptr_mut(),
            )
        }
        .result()
    }

    pub fn backward<
        T: TensorDataType,
        const N: usize,
        const C: usize,
        const H: usize,
        const W: usize,
    >(
        &self,
        cudnn_handle: &CudnnHandle,
        input: &Tensor4D<T, N, C, H, W>,
        d_input: &Tensor4D<T, N, C, H, W>,
        output: &Tensor4D<T, N, C, H, W>,
        d_output: &mut Tensor4D<T, N, C, H, W>,
    ) -> CudaCudnnResult<()> {
        unsafe {
            cudnnActivationBackward(
                cudnn_handle.get_handle(),
                self.descriptor,
                &T::ONE as *const _ as *const _,
                input.get_descriptor(),
                input.get_data_ptr(),
                d_input.get_descriptor(),
                d_input.get_data_ptr(),
                output.get_descriptor(),
                output.get_data_ptr(),
                &T::ZERO as *const _ as *const _,
                d_output.get_descriptor(),
                d_output.get_data_ptr_mut(),
            )
        }
        .result()
    }
}
impl<A> Drop for Activation<A> {
    fn drop(&mut self) {
        unsafe { cudnnDestroyActivationDescriptor(self.descriptor) }
            .result()
            .unwrap();
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn test_relu_activation_forward_backward() {
        let cuda = CudaDeviceBuilder::new(0).build().unwrap();
        let cudnn_handle = CudnnHandle::create(&cuda).unwrap();
        let x = Tensor4D::alloc_with(&cuda, [[[[f64::NAN, 2.0]]], [[[-1.0, 0.0]]]]).unwrap();
        let dy = Tensor4D::alloc_with(&cuda, [[[[f64::NAN, 3.0]]], [[[-1.0, 0.0]]]]).unwrap();
        let mut dx = unsafe { Tensor4D::alloc_uninit(&cuda) }.unwrap();
        let mut y = unsafe { Tensor4D::alloc_uninit(&cuda) }.unwrap();

        let activation = Activation::<Relu>::create().unwrap();
        activation.forward(&cudnn_handle, &x, &mut y).unwrap();

        let out = y.get_data().unwrap();
        assert!(out[0][0][0][0].is_nan());
        assert!((out[0][0][0][1] - 2.0).abs() < f64::EPSILON);
        assert!(out[1][0][0][0].abs() < f64::EPSILON);
        assert!(out[1][0][0][1].abs() < f64::EPSILON);
        activation
            .backward(&cudnn_handle, &x, &dy, &y, &mut dx)
            .unwrap();

        let out = dx.get_data().unwrap();
        // NANs aren't backpropagated
        assert!(out[0][0][0][0].abs() < f64::EPSILON);
        assert!((out[0][0][0][1] - 3.0).abs() < f64::EPSILON);
        assert!(out[1][0][0][0].abs() < f64::EPSILON);
        assert!(out[1][0][0][1].abs() < f64::EPSILON);
    }
}
