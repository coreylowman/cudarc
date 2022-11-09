use core::marker::PhantomData;
use core::mem::MaybeUninit;

use alloc::rc::Rc;

use super::sys::*;
use crate::prelude::*;

pub struct Activation<A> {
    pub(crate) descriptor: Rc<ActivationDescriptor>,
    activation_mode: PhantomData<A>,
}
impl<A> Clone for Activation<A> {
    fn clone(&self) -> Self {
        Self {
            activation_mode: PhantomData,
            descriptor:      Rc::clone(&self.descriptor),
        }
    }
}
impl<A: ActivationMode> Activation<A> {
    pub fn create() -> CudnnResult<Self> {
        let descriptor = Rc::new(ActivationDescriptor::create()?);
        unsafe {
            cudnnSetActivationDescriptor(
                descriptor.0,
                A::get_activation_mode(),
                cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
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
    ) -> CudnnResult<()> {
        unsafe {
            cudnnActivationForward(
                cudnn_handle.0,
                self.descriptor.0,
                &T::ONE as *const _ as *const _,
                input.descriptor.descriptor.0,
                input.data.t_cuda.cu_device_ptr as *const _,
                &T::ZERO as *const _ as *const _,
                output.descriptor.descriptor.0,
                output.data.t_cuda.cu_device_ptr as *mut _,
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
    ) -> CudnnResult<()> {
        unsafe {
            cudnnActivationBackward(
                cudnn_handle.0,
                self.descriptor.0,
                &T::ONE as *const _ as *const _,
                input.descriptor.descriptor.0,
                input.data.t_cuda.cu_device_ptr as *const _,
                d_input.descriptor.descriptor.0,
                d_input.data.t_cuda.cu_device_ptr as *const _,
                output.descriptor.descriptor.0,
                output.data.t_cuda.cu_device_ptr as *const _,
                &T::ZERO as *const _ as *const _,
                d_output.descriptor.descriptor.0,
                d_output.data.t_cuda.cu_device_ptr as *mut _,
            )
        }
        .result()
    }
}

pub struct ActivationDescriptor(pub(crate) cudnnActivationDescriptor_t);
impl ActivationDescriptor {
    pub fn create() -> CudnnResult<Self> {
        let mut descriptor = MaybeUninit::uninit();
        unsafe {
            cudnnCreateActivationDescriptor(descriptor.as_mut_ptr()).result()?;
            Ok(Self(descriptor.assume_init()))
        }
    }
}
impl Drop for ActivationDescriptor {
    fn drop(&mut self) {
        unsafe { cudnnDestroyActivationDescriptor(self.0) }
            .result()
            .unwrap();
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn test_create_descriptor() {
        let _descriptor = ActivationDescriptor::create().unwrap();
    }

    #[test]
    fn test_relu_activation_forward_backward() {
        let cuda = CudaDeviceBuilder::new(0).build().unwrap();
        let cudnn_handle = CudnnHandle::create(&cuda).unwrap();
        let x = Tensor2D::alloc_with(&cuda, [[[[f64::NAN, 2.0]]], [[[-1.0, 0.0]]]]).unwrap();
        let dy = Tensor2D::alloc_with(&cuda, [[[[f64::NAN, 3.0]]], [[[-1.0, 0.0]]]]).unwrap();
        let mut dx = unsafe { Tensor2D::alloc_uninit(&cuda) }.unwrap();
        let mut y = unsafe { Tensor2D::alloc_uninit(&cuda) }.unwrap();

        let activation = Activation::<Relu>::create().unwrap();
        activation.forward(&cudnn_handle, &x, &mut y).unwrap();

        let out = y.data.clone().into_host().unwrap();
        assert!(out[0][0][0][0].is_nan());
        assert!((out[0][0][0][1] - 2.0).abs() < f64::EPSILON);
        assert!(out[1][0][0][0].abs() < f64::EPSILON);
        assert!(out[1][0][0][1].abs() < f64::EPSILON);
        activation
            .backward(&cudnn_handle, &x, &dy, &y, &mut dx)
            .unwrap();

        let out = dx.data.into_host().unwrap();
        // NANs aren't backpropagated
        assert!(out[0][0][0][0].abs() < f64::EPSILON);
        assert!((out[0][0][0][1] - 3.0).abs() < f64::EPSILON);
        assert!(out[1][0][0][0].abs() < f64::EPSILON);
        assert!(out[1][0][0][1].abs() < f64::EPSILON);
    }
}
