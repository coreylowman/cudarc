use core::marker::PhantomData;

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
        cudnn_handle: CudnnHandle,
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

    pub fn forward_inplace<
        T: TensorDataType,
        const N: usize,
        const C: usize,
        const H: usize,
        const W: usize,
    >(
        &self,
        cudnn_handle: CudnnHandle,
        input: &mut Tensor4D<T, N, C, H, W>,
    ) -> CudnnResult<()> {
        self.forward(cudnn_handle, input, unsafe { &mut *(&*input as *const _ as *mut _) })
    }

    pub fn backward<
        T: TensorDataType,
        const N: usize,
        const C: usize,
        const H: usize,
        const W: usize,
    >(
        &self,
        cudnn_handle: CudnnHandle,
        input: &Tensor4D<T, N, C, H, W>,
        d_input: &Tensor4D<T, N, C, H, W>,
        output: &mut Tensor4D<T, N, C, H, W>,
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
                output.data.t_cuda.cu_device_ptr as *mut _,
                &T::ZERO as *const _ as *const _,
                d_output.descriptor.descriptor.0,
                d_output.data.t_cuda.cu_device_ptr as *mut _,
            )
        }
        .result()
    }

    pub fn backward_inplace<
        T: TensorDataType,
        const N: usize,
        const C: usize,
        const H: usize,
        const W: usize,
    >(
        &self,
        cudnn_handle: CudnnHandle,
        input: &mut Tensor4D<T, N, C, H, W>,
        d_input: &mut Tensor4D<T, N, C, H, W>,
    ) -> CudnnResult<()> {
        self.backward(cudnn_handle, input, d_input, unsafe { &mut *(&*input as *const _ as *mut _) }, unsafe { &mut *(&*d_input as *const _ as *mut _)})
    }
}

pub struct ActivationDescriptor(pub(crate) cudnnActivationDescriptor_t);
impl ActivationDescriptor {
    pub fn create() -> CudnnResult<Self> {
        let mut descriptor: Self = unsafe { std::mem::zeroed() };
        unsafe { cudnnCreateActivationDescriptor(&mut descriptor.0 as *mut _) }.result()?;
        Ok(descriptor)
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
    use core::mem::zeroed;

    use alloc::rc::Rc;

    use crate::prelude::*;

    #[test]
    fn test_create_descriptor() {
        let _descriptor = ActivationDescriptor::create().unwrap();
    }

    #[test]
    fn test_relu_activation_f32() {
        let cudnn_handle = CudnnHandle::create().unwrap();
        let cuda = CudaDeviceBuilder::new(0).build().unwrap();
        let allocation_in = cuda.alloc_zeros().unwrap();
        let tensor_in =
        Tensor2D::create_async(allocation_in.clone(), &[[[[f32::NAN, 2.0]]], [[[-1.0, 0.0]]]]).unwrap();
        let allocation_out = cuda.take(Rc::new(unsafe { zeroed() })).unwrap();
        let mut tensor_out = unsafe { Tensor2D::uninit(allocation_out.clone()) }.unwrap();

        let activation = Activation::<Relu>::create().unwrap();
        cuda.synchronize().unwrap();
        activation
            .forward(cudnn_handle, &tensor_in, &mut tensor_out)
            .unwrap();

        let out = allocation_out.sync_release().unwrap().unwrap();
        assert!(out[0][0][0][0].is_nan());
        assert!((out[0][0][0][1] - 2.0).abs() < f32::EPSILON);
        assert!(out[1][0][0][0].abs() < f32::EPSILON);
        assert!(out[1][0][0][1].abs() < f32::EPSILON);
    }

    #[test]
    fn test_relu_activation_f64_inplace() {
        let cudnn_handle = CudnnHandle::create().unwrap();
        let cuda = CudaDeviceBuilder::new(0).build().unwrap();
        let allocation_in = cuda.take(Rc::new(unsafe { zeroed() })).unwrap();
        let mut tensor_in = Tensor2D::create_async(allocation_in.clone(), &[
            [[[f64::NAN, 2.0]]],
            [[[-1.0, 0.0]]],
        ])
        .unwrap();
        let activation = Activation::<Relu>::create().unwrap();
        cuda.synchronize().unwrap();
        activation
            .forward_inplace(cudnn_handle, &mut tensor_in)
            .unwrap();
            
        let out = allocation_in.sync_release().unwrap().unwrap();
        assert!(out[0][0][0][0].is_nan());
        assert!((out[0][0][0][1] - 2.0).abs() < f64::EPSILON);
        assert!(out[1][0][0][0].abs() < f64::EPSILON);
        assert!(out[1][0][0][1].abs() < f64::EPSILON);
    }
}
