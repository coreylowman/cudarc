use core::marker::PhantomData;
use core::mem::MaybeUninit;

use super::sys::*;
use crate::prelude::*;

pub struct Activation<A> {
    descriptor:      ActivationDescriptor,
    activation_mode: PhantomData<A>,
}
impl<A: ActivationMode> Activation<A> {
    pub fn create() -> CudnnResult<Self> {
        let descriptor = ActivationDescriptor::create()?;
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
    use alloc::rc::Rc;

    use crate::cudarc::CudaUniquePtr;
    use crate::prelude::*;

    #[test]
    fn test_create_descriptor() {
        let _descriptor = ActivationDescriptor::create().unwrap();
    }

    #[test]
    fn test_relu_activation_forward_backward() {
        let cuda = CudaDeviceBuilder::new(0).build().unwrap();
        let cudnn_handle = CudnnHandle::create(&cuda).unwrap();
        let allocation_in = cuda
            .take(Rc::new([[[[f64::NAN, 2.0]]], [[[-1.0, 0.0]]]]))
            .unwrap();
        let tensor_in = Tensor2D::create(allocation_in.clone()).unwrap();
        let allocation_d_in = cuda
            .take(Rc::new([[[[f64::NAN, 3.0]]], [[[-1.0, 0.0]]]]))
            .unwrap();
        let tensor_d_in = Tensor2D::create(allocation_d_in.clone()).unwrap();
        let allocation_out = CudaRc {
            t_cuda: Rc::new(unsafe { CudaUniquePtr::alloc(&cuda).unwrap() }),
            t_host: None,
        };
        let mut tensor_out = Tensor2D::create(allocation_out.clone()).unwrap();

        let allocation_d_out = CudaRc {
            t_cuda: Rc::new(unsafe { CudaUniquePtr::alloc(&cuda).unwrap() }),
            t_host: None,
        };
        let mut tensor_d_out = Tensor2D::create(allocation_d_out.clone()).unwrap();

        let activation = Activation::<Relu>::create().unwrap();
        activation
            .forward(&cudnn_handle, &tensor_in, &mut tensor_out)
            .unwrap();

        let out = allocation_out.into_host().unwrap();
        assert!(out[0][0][0][0].is_nan());
        assert!((out[0][0][0][1] - 2.0).abs() < f64::EPSILON);
        assert!(out[1][0][0][0].abs() < f64::EPSILON);
        assert!(out[1][0][0][1].abs() < f64::EPSILON);
        activation
            .backward(
                &cudnn_handle,
                &tensor_in,
                &tensor_d_in,
                &tensor_out,
                &mut tensor_d_out,
            )
            .unwrap();

        let out = allocation_d_out.into_host().unwrap();
        // NANs aren't backpropagated
        assert!(out[0][0][0][0].abs() < f64::EPSILON);
        assert!((out[0][0][0][1] - 3.0).abs() < f64::EPSILON);
        assert!(out[1][0][0][0].abs() < f64::EPSILON);
        assert!(out[1][0][0][1].abs() < f64::EPSILON);
    }
}
